from typing import Sequence, Mapping, Any, Union, Optional
from pathlib import Path
import requests
import zipfile
import torch
import sys
import os

REPOS = [
    {
        "url": "https://github.com/comfyanonymous/ComfyUI/archive/refs/heads/master.zip",
        "zip_name": "ComfyUI-master.zip",
        "inner_dir": "ComfyUI-master",
        "target_dir": Path("ComfyUI"),
    },
    {
        "url": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/archive/refs/heads/main.zip",
        "zip_name": "ComfyUI-VideoHelperSuite.zip",
        "inner_dir": "ComfyUI-VideoHelperSuite-main",
        "target_dir": Path("ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite"),
    },
    {
        "url": "https://github.com/city96/ComfyUI-GGUF/archive/refs/heads/main.zip",
        "zip_name": "ComfyUI-GGUF.zip",
        "inner_dir": "ComfyUI-GGUF-main",
        "target_dir": Path("ComfyUI/custom_nodes/ComfyUI-GGUF"),
    },
]

def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return
    print(f"[download] {url} → {dest}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(4096):
            f.write(chunk)
    print(f"[done ] {dest}")

def fetch_repos():
    for repo in REPOS:
        zip_path = Path(repo["zip_name"])
        tgt_dir  = repo["target_dir"]
        inner   = repo["inner_dir"]

        if tgt_dir.exists():
            print(f"[skip] {tgt_dir} already present")
            continue

        # 1) Download .zip
        download_file(repo["url"], zip_path)

        # 2) Extract
        print(f"[extract] {zip_path} → {tgt_dir.parent}/")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tgt_dir.parent)

        # 3) Move & cleanup
        extracted = repo["target_dir"].parent / inner
        extracted.rename(tgt_dir)
        zip_path.unlink()
        print(f"[done ] repo at {tgt_dir}")

fetch_repos()

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

# map local target → URL to download from
MODELS = {
    "ComfyUI/models/unet/Wan2.2-T2V-A14B-HighNoise.gguf":
        "https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF/resolve/main/HighNoise/Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf",
    "ComfyUI/models/unet/Wan2.2-T2V-A14B-LowNoise.gguf":
        "https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF/resolve/main/LowNoise/Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf",
    "ComfyUI/models/vae/Wan2.1_VAE.safetensors":
        "https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF/resolve/main/VAE/Wan2.1_VAE.safetensors",
    "ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors":
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "ComfyUI/models/loras/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors":
        "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors",
    "ComfyUI/models/loras/Wan2.1_T2V_14B_FusionX_LoRA.safetensors":
        "https://huggingface.co/hotdogs/wan_nsfw_lora/resolve/main/Wan2.1_T2V_14B_FusionX_LoRA.safetensors",
}

def _download(url: str, dest: Path):
    """Download `url` to `dest` if `dest` does not already exist."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return
    print(f"[downloading] {url} → {dest}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=4_096):
            f.write(chunk)
    print(f"[done] {dest}")

def download_all_models():
    for rel_path, url in MODELS.items():
        _download(url, Path(rel_path))

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
    
def load_clip(model_name: str):
    return NODE_CLASS_MAPPINGS['CLIPLoader']().load_clip(
        clip_name=model_name,
        type='wan',
        device='default',
    )

def load_vae(model_name: str):
    return NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name=model_name)