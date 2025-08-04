from typing import Sequence, Mapping, Any, Union, Optional
import torch
import sys
import os

### COMFYUI UTILITIES ###

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except (KeyError, IndexError):
        return obj["result"][index] # type: ignore

def find_path(name: str, path: Optional[str] = None) -> Optional[str]:
    if path is None:
        path = os.getcwd()

    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    parent_directory = os.path.dirname(path)

    if parent_directory == path:
        return None

    return find_path(name, parent_directory)

def add_comfyui_directory_to_sys_path() -> None:
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")

def add_extra_model_paths() -> None:
    try:
        from main import load_extra_path_config # type: ignore
    except ImportError:
        print("Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead.")
        try:
            from utils.extra_config import load_extra_path_config
        except ImportError:
            print("Could not import load_extra_path_config from utils.extra_config either.")
            return

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")

def import_custom_nodes() -> None:
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    init_extra_nodes()

add_comfyui_directory_to_sys_path()
add_extra_model_paths()
from nodes import NODE_CLASS_MAPPINGS
import_custom_nodes()

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
    
def load_vae(model_name: str):
    return NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name=model_name)