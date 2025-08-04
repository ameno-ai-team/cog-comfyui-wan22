from cog import BasePredictor, Input, Path
from comfy_utils import *
from typing import List
import random
import shutil
import torch
import uuid
import os

OUTPUT_DIR = '/tmp/outputs'

class Predictor(BasePredictor):
    def setup(self):
        self.wan_14b_high = load_models_with_stack_loras('Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf')
        self.wan_14b_low  = load_models_with_stack_loras('Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf')
        self.wan_vae = load_vae('Wan2.1_VAE.safetensors')
        self.negative = torch.load('negative.pt', map_location='cuda', weights_only=True)

    def predict(
        self,
        prompt: str = Input(
            description='Input prompt',
            default='',
        ),
        width: int = Input(
            description='Width (default: 432)',
            default=432
        ),
        height: int = Input(
            description='Height (default: 768)',
            default=768
        ),
        length: int = Input(
            description='Length/Frames(default: 81)',
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
            clip = NODE_CLASS_MAPPINGS['CLIPLoader']().load_clip(
                clip_name='umt5_xxl_fp8_e4m3fn_scaled.safetensors',
                type='wan',
                device='default',
            )
            textencode = NODE_CLASS_MAPPINGS['CLIPTextEncode']().encode(
                text=f'realistic, {prompt}',
                clip=get_value_at_index(clip, 0)
            )
            positive = get_value_at_index(textencode, 0)

            latents = NODE_CLASS_MAPPINGS["EmptyHunyuanLatentVideo"]().generate(
                width=width,
                height=height,
                length=length,
                batch_size=1,
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
                sampler_name='res_multi',
                scheduler='beta',
                positive=positive,
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
                sampler_name='res_multi',
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

            # upscale_model = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]().load_model(
            #     model_name='RealESRGAN_x2.pth'
            # )
                
            # vaedecode = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]().upscale(
            #     upscale_model=get_value_at_index(upscale_model, 0),
            #     image=get_value_at_index(vaedecode, 0),
            # )

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
            return [new_path]
