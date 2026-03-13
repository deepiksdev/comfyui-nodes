import importlib
import logging

from .nodes import api_routes
from .nodes.task_nodes import (
    T2TNode, I2TNode, T2INode, I2INode,
    I2I3Node, I2I10Node,
    T2VNode, I2VNode, I2V2Node, I2VRNode,
    V2VNode, V2VRNode,
)
from .nodes.display_node import NODE_CLASS_MAPPINGS as DISPLAY_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_NAMES
from .nodes.video_to_image_node import NODE_CLASS_MAPPINGS as VTI_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as VTI_NAMES

# Node order here controls display order in ComfyUI
NODE_CLASS_MAPPINGS = {
    "DeepGen_T2T": T2TNode,
    "DeepGen_I2T": I2TNode,
    "DeepGen_T2I": T2INode,
    "DeepGen_I2I": I2INode,
    "DeepGen_I2I3": I2I3Node,
    "DeepGen_I2I10": I2I10Node,
    "DeepGen_T2V": T2VNode,
    "DeepGen_I2V": I2VNode,
    "DeepGen_I2V2": I2V2Node,
    "DeepGen_I2VR": I2VRNode,
    "DeepGen_V2V": V2VNode,
    "DeepGen_V2VR": V2VRNode,
    **DISPLAY_MAPPINGS,
    **VTI_MAPPINGS,
}

logging.info("[DeepGen] NODE_CLASS_MAPPINGS insertion order: %s", list(NODE_CLASS_MAPPINGS.keys()))

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepGen_T2T": "Invoke LLM",
    "DeepGen_I2T": "Review Images",
    "DeepGen_T2I": "Generate Image (from Text)",
    "DeepGen_I2I": "Edit Image",
    "DeepGen_I2I3": "Generate Image (from 3 Images)",
    "DeepGen_I2I10": "Generate Image (from 10 Images)",
    "DeepGen_T2V": "Generate Video (from Text)",
    "DeepGen_I2V": "Generate Video (from Start Frame)",
    "DeepGen_I2V2": "Generate Video (from Start and End Frames)",
    "DeepGen_I2VR": "Generate Video (from Images with Elements)",
    "DeepGen_V2V": "Edit Video",
    "DeepGen_V2VR": "Edit Video (with Elements)",
    **DISPLAY_NAMES,
    **VTI_NAMES,
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
