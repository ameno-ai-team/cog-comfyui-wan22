from typing import Sequence, Mapping, Any, Union, Optional
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
        from main import load_extra_path_config
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

async def init_extra_nodes_async():
    from nodes import init_extra_nodes
    await init_extra_nodes()

def import_custom_nodes() -> None:
    import asyncio
    import execution
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Run the async init_extra_nodes function
    loop.run_until_complete(init_extra_nodes_async())

add_comfyui_directory_to_sys_path()
add_extra_model_paths()
from nodes import NODE_CLASS_MAPPINGS
import_custom_nodes()