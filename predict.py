from cog import BasePredictor, Input, Path
from comfy_utils import *
from typing import List
import logging
import random
import shutil
import torch
import uuid
import os

OUTPUT_DIR = '/tmp/outputs'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LORAS = [
    { 'lora': 'Wan2.1_T2V_14B_FusionX_LoRA.safetensors', 'strength_model': 0.5 },
    { 'lora': 'lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors', 'strength_model': 1.0 }
]

def load_models_with_stack_loras(model_name: str):
    with torch.inference_mode():
        model = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]().load_unet(
            unet_name=model_name,
        )
        model = get_value_at_index(model, 0)

        logger.info('Loading LoRAs ...')
        for l in LORAS:
            lora = NODE_CLASS_MAPPINGS['LoraLoaderModelOnly']().load_lora_model_only(
                model=model,
                lora_name=l['lora'],
                strength_model=l.get('strength_model', 1.0)
            )
            model = get_value_at_index(lora, 0)
            del lora
        
        modelsamplingsd3 = NODE_CLASS_MAPPINGS["ModelSamplingSD3"]().patch(
            shift=8.0, model=model
        )
        del model
        
        model = get_value_at_index(modelsamplingsd3, 0)
        return model
    
def load_clip(model_name: str):
    with torch.inference_mode():
        return NODE_CLASS_MAPPINGS['CLIPLoader']().load_clip(
            clip_name=model_name,
            type='wan',
            device='default',
        )

def load_vae(model_name: str):
    with torch.inference_mode():
        return NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name=model_name)

class Predictor(BasePredictor):
    def setup(self):
        logger.info("Starting setup...")
        logger.info("Loading models...")
        logger.info(f'{os.listdir(".")}')
        self.wan_14b_high = load_models_with_stack_loras('Wan2.2-T2V-A14B-HighNoise.gguf')
        self.wan_14b_low  = load_models_with_stack_loras('Wan2.2-T2V-A14B-LowNoise.gguf')
        self.clip = load_clip('umt5_xxl_fp8_e4m3fn_scaled.safetensors')
        self.wan_vae = load_vae('Wan2.1_VAE.safetensors')
        self.negative = torch.load('negative.pt', map_location='cuda', weights_only=True)
        
        print("Setup completed successfully!")

    def predict(
        self,
        prompt: str = Input(
            description='Input prompt',
            default='',
        ),
        width: int = Input(
            description='Width (default: 480)',
            default=480
        ),
        height: int = Input(
            description='Height (default: 864)',
            default=864
        ),
        length: int = Input(
            description='Length/Frames(default: 81)',
            choices=[17, 33, 49, 65, 81],
            default=81
        ),
        steps: int = Input(
            description='Steps for generation',
            default=8
        ),
        seeds: int = Input(
            description='Seeds (Zero for automatically randomise)',
            default=0,
        ),
    ) -> List[Path]:
        '''Run a single prediction on the model'''
        # Encode input prompt
        with torch.inference_mode():
            positive = NODE_CLASS_MAPPINGS['CLIPTextEncode']().encode(
                text=f'realistic, {prompt}',
                clip=get_value_at_index(self.clip, 0)
            )

            latents = NODE_CLASS_MAPPINGS["Wan22ImageToVideoLatent"]().encode(
                width=width,
                height=height,
                length=length,
                batch_size=1,
                vae=get_value_at_index(self.wan_vae, 0)
            )

            mid_step = steps // 2
            cur_seeds = random.randint(1, 2**64) if seeds == 0 else seeds

            # KSampler HighNoise Model
            latents = NODE_CLASS_MAPPINGS["KSamplerAdvanced"]().sample(
                model=self.wan_14b_high, 
                add_noise='enable', 
                noise_seed=cur_seeds,
                steps=steps,
                cfg=1.0,
                sampler_name='res_multistep',
                scheduler='beta',
                positive=get_value_at_index(positive, 0),
                negative=self.negative,
                latent_image=get_value_at_index(latents, 0),
                start_at_step=0, 
                end_at_step=mid_step, 
                return_with_leftover_noise='enable',
                denoise=1.0,
            )

            # KSampler LowNoise Model
            latents = NODE_CLASS_MAPPINGS["KSamplerAdvanced"]().sample(
                model=self.wan_14b_low, 
                add_noise='disable', 
                noise_seed=cur_seeds,
                steps=steps,
                cfg=1.0,
                sampler_name='res_multistep',
                scheduler='beta',
                positive=positive,
                negative=self.negative,
                latent_image=get_value_at_index(latents, 0),
                start_at_step=mid_step, 
                end_at_step=10000, 
                return_with_leftover_noise='disable',
                denoise=1.0,
            )

            vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]().decode(
                samples=get_value_at_index(latents, 0),
                vae=get_value_at_index(self.wan_vae, 0),
            )

            vhs = NODE_CLASS_MAPPINGS['VHS_VideoCombine']().combine_video(
                frame_rate=16,
                loop_count=0,
                filename_prefix=f"wan",
                format='video/h264-mp4',
                pix_fmt='yuv420p',
                crf=19,
                save_metadata=True,
                trim_to_audio=False,
                pingpong=False,
                save_output=True,
                images=get_value_at_index(vaedecode, 0),
                unique_id=6180912156727070981,
            )
            result = get_value_at_index(vhs, 0)[1]

            new_path = f'{uuid.uuid4()}.mp4'
            shutil.move(result[1], os.path.join(OUTPUT_DIR, new_path))
            os.remove(result[0])
            os.remove(result[1])
            print(new_path)
            return [Path(new_path)]
