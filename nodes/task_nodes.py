import os
import csv
import json
import torch
import copy
from .deepgen_utils import DeepGenApiHandler as ApiHandler, ImageUtils, ResultProcessor

def load_models_for_task(task_name):
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models.csv")
    models = []
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                tasks = [x.strip() for x in row[2].split(",")]
                if task_name in tasks:
                    models.append(row[0])
    except Exception as e:
        print(f"DeepGen: Failed to load models for task {task_name}: {e}")
    if not models:
        models = ["No models found"]
    return models

def parse_config_json(config_str):
    if not config_str or not config_str.strip():
        return {}
    try:
        return json.loads(config_str)
    except json.JSONDecodeError as e:
        print(f"DeepGen: Failed to parse config_json: {e}")
        return {}

def process_kwargs_for_images(kwargs, unique_id, extra_pnginfo):
    attachments_files = []
    original_names_map = {}

    def get_orig_name(idx, original_names):
        if idx < len(original_names) and original_names[idx]:
            org = str(original_names[idx])
            if org.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.mp4')):
                org = org.rsplit('.', 1)[0]
            return f"_{org}"
        return ""

    for k, v in kwargs.items():
        if v is None:
            continue
        if k in ["model", "prompt", "seed_value", "nb_results", "output_prefix", "config_json", "minimum_resolution", "aspect_ratio", "output_format", "endpoint", "unique_id", "extra_pnginfo"]:
            continue

        prefix_base = k
        if k not in original_names_map and unique_id and extra_pnginfo:
            original_names_map[k] = ImageUtils.resolve_filenames(unique_id, extra_pnginfo, k)
        original_names = original_names_map.get(k, [])

        v_list = v if isinstance(v, list) else [v]
        flattened_items = []
        for item in v_list:
            if hasattr(item, "shape") and len(item.shape) == 4:
                for i in range(item.shape[0]):
                    flattened_items.append(item[i:i+1])
            elif isinstance(item, list):
                flattened_items.extend(item)
            else:
                flattened_items.append(item)

        for i, item in enumerate(flattened_items):
            if hasattr(item, "shape"):
                attach = ImageUtils.get_attachment_file(item, filename=f"{prefix_base}_{i+1}{get_orig_name(i, original_names)}.png")
                if attach:
                    attachments_files.append(attach)

    return attachments_files

class BaseTaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {}}

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def run_generation(self, task_type, **kwargs):
        def unwrap(v):
            return v[0] if isinstance(v, list) and len(v) > 0 else v

        model = unwrap(kwargs.get("model", ""))
        prompt = unwrap(kwargs.get("prompt", ""))
        seed_value = unwrap(kwargs.get("seed_value", 1000))
        nb_results = unwrap(kwargs.get("nb_results", 1))
        output_prefix = unwrap(kwargs.get("output_prefix", ""))
        config_json_str = unwrap(kwargs.get("config_json", ""))
        minimum_resolution = unwrap(kwargs.get("minimum_resolution", ""))
        aspect_ratio = unwrap(kwargs.get("aspect_ratio", ""))
        output_format = unwrap(kwargs.get("output_format", ""))
        endpoint = unwrap(kwargs.get("endpoint"))
        unique_id = unwrap(kwargs.get("unique_id"))
        extra_pnginfo = unwrap(kwargs.get("extra_pnginfo"))

        arguments = {
            "prompt": prompt,
            "seed": seed_value,
        }
        
        if task_type in ["T2I", "I2I", "I2I3", "I2I10", "T2V", "V2VR"]:
            arguments["num_images"] = nb_results # used for both image and video
            
        if minimum_resolution:
            arguments["resolution"] = minimum_resolution
        if aspect_ratio:
            arguments["aspect_ratio"] = aspect_ratio
        if output_format:
            arguments["output_format"] = output_format

        attachments_files = process_kwargs_for_images(kwargs, unique_id, extra_pnginfo)
        if attachments_files:
            arguments["attachments_files"] = attachments_files

        extra_args = parse_config_json(config_json_str)
        arguments.update(extra_args)

        try:
            if task_type in ["T2V", "V2VR"]:
                if nb_results > 1:
                    results = ApiHandler.submit_multiple_and_get_results(model, arguments, nb_results, api_url=endpoint)
                    # For video, polling might be needed. Using submit_and_get_result for now.
                    # As batch video is rare, we default to sequential handling
                    outputs = []
                    credits_out = 0.0
                    for _ in range(nb_results):
                        result = ApiHandler.submit_and_get_result(model, arguments, api_url=endpoint)
                        outputs.append(ResultProcessor.process_video_result(result)[0])
                        obj = result[0] if isinstance(result, list) and len(result) > 0 else result
                        cred = obj.get("total_credits_used") if isinstance(obj, dict) else 0.0
                        if cred is None:
                            cred = obj.get("output", {}).get("total_credits_used", obj.get("aiCredits", 0.0))
                        credits_out += float(cred or 0.0)
                    prefixed_model = f"{output_prefix}_{model}" if output_prefix else model
                    return (outputs[0], prefixed_model, credits_out) # returning first video
                else:
                    result = ApiHandler.submit_and_get_result(model, arguments, api_url=endpoint)
                    res_obj = result[0] if isinstance(result, list) and len(result) > 0 else result
                    video_path = ResultProcessor.process_video_result(result)[0]
                    
                    def _get_attr(obj, key, default=None):
                        if isinstance(obj, dict): return obj.get(key, default)
                        return getattr(obj, key, default)
                        
                    agent_alias = _get_attr(res_obj, "agent_alias", model)
                    if isinstance(res_obj, dict) and "output" in res_obj:
                        agent_alias = res_obj["output"].get("agent_alias", agent_alias)
                        
                    prefixed_model = f"{output_prefix}_{agent_alias}" if output_prefix else agent_alias
                    
                    cred = _get_attr(res_obj, "total_credits_used")
                    if cred is None:
                        out_obj = _get_attr(res_obj, "output", {})
                        cred = _get_attr(out_obj, "total_credits_used", _get_attr(res_obj, "aiCredits", 0.0))
                    credits_out = float(cred or 0.0)
                    return (video_path, prefixed_model, credits_out)

            elif task_type in ["T2T", "I2T"]:
                arguments["stream"] = False
                result = ApiHandler.submit_and_get_result(model, arguments, api_url=endpoint)
                res_obj = result[0] if isinstance(result, list) and len(result) > 0 else result
                text_result = ResultProcessor.process_text_result(result)[0]
                
                def _get_attr(obj, key, default=None):
                    if isinstance(obj, dict): return obj.get(key, default)
                    return getattr(obj, key, default)
                    
                agent_alias = _get_attr(res_obj, "agent_alias", model)
                if isinstance(res_obj, dict) and "output" in res_obj:
                    agent_alias = res_obj["output"].get("agent_alias", agent_alias)
                    
                cred = _get_attr(res_obj, "total_credits_used")
                if cred is None:
                    out_obj = _get_attr(res_obj, "output", {})
                    cred = _get_attr(out_obj, "total_credits_used", _get_attr(res_obj, "aiCredits", 0.0))
                credits_out = float(cred or 0.0)
                # No output prefix for T nodes
                return (text_result, agent_alias, credits_out)

            else:
                # Images
                result = ApiHandler.submit_and_get_result(model, arguments, api_url=endpoint)
                res_obj = result[0] if isinstance(result, list) and len(result) > 0 else result
                img_tensor = ResultProcessor.process_image_result(result)[0]
                
                def _get_attr(obj, key, default=None):
                    if isinstance(obj, dict): return obj.get(key, default)
                    return getattr(obj, key, default)
                    
                agent_alias = _get_attr(res_obj, "agent_alias", model)
                if isinstance(res_obj, dict) and "output" in res_obj:
                    agent_alias = res_obj["output"].get("agent_alias", agent_alias)
                    
                prefixed_model = f"{output_prefix}_{agent_alias}" if output_prefix else agent_alias
                
                cred = _get_attr(res_obj, "total_credits_used")
                if cred is None:
                    out_obj = _get_attr(res_obj, "output", {})
                    cred = _get_attr(out_obj, "total_credits_used", _get_attr(res_obj, "aiCredits", 0.0))
                credits_out = float(cred or 0.0)
                return (img_tensor, prefixed_model, credits_out)

        except Exception as e:
            if task_type in ["T2T", "I2T"]:
                return (f"Error: {e}", model, 0.0)
            elif task_type in ["T2V", "V2VR"]:
                return (f"Error: {e}", model, 0.0)
            else:
                blank_img = ResultProcessor.create_blank_image()[0]
                return (blank_img, model, 0.0)


class T2TNode(BaseTaskNode):
    @classmethod
    def INPUT_TYPES(cls):
        models = load_models_for_task("T2T")
        return {
            "required": {
                "model": (models,),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed_value": ("INT", {"default": 1000}),
                "config_json": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT",)
    RETURN_NAMES = ("output", "model", "total_credits_used",)
    FUNCTION = "generate"
    CATEGORY = "DeepGen/Generators"

    def generate(self, **kwargs):
        return self.run_generation("T2T", **kwargs)

class I2TNode(BaseTaskNode):
    @classmethod
    def INPUT_TYPES(cls):
        models = load_models_for_task("I2T")
        optional_images = {f"image_{i}": ("IMAGE",) for i in range(1, 11)}
        return {
            "required": {
                "model": (models,),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed_value": ("INT", {"default": 1000}),
                "config_json": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                **optional_images,
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT",)
    RETURN_NAMES = ("output", "model", "total_credits_used",)
    FUNCTION = "generate"
    CATEGORY = "DeepGen/Generators"

    def generate(self, **kwargs):
        return self.run_generation("I2T", **kwargs)

class T2INode(BaseTaskNode):
    @classmethod
    def INPUT_TYPES(cls):
        models = load_models_for_task("T2I")
        return {
            "required": {
                "model": (models,),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed_value": ("INT", {"default": 1000}),
                "nb_results": ("INT", {"default": 1, "min": 1, "max": 10}),
                "output_prefix": ("STRING", {"default": ""}),
                "config_json": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "minimum_resolution": ("STRING", {"default": ""}),
                "aspect_ratio": ("STRING", {"default": ""}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT",)
    RETURN_NAMES = ("IMAGE", "output_prefix_and_model", "total_credits_used",)
    FUNCTION = "generate"
    CATEGORY = "DeepGen/Generators"

    def generate(self, **kwargs):
        return self.run_generation("T2I", **kwargs)

class I2INode(BaseTaskNode):
    @classmethod
    def INPUT_TYPES(cls):
        models = load_models_for_task("I2I")
        return {
            "required": {
                "model": (models,),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed_value": ("INT", {"default": 1000}),
                "nb_results": ("INT", {"default": 1, "min": 1, "max": 10}),
                "output_prefix": ("STRING", {"default": ""}),
                "config_json": ("STRING", {"default": "", "multiline": True}),
                "image_1": ("IMAGE",),
            },
            "optional": {
                "minimum_resolution": ("STRING", {"default": ""}),
                "aspect_ratio": ("STRING", {"default": ""}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT",)
    RETURN_NAMES = ("IMAGE", "output_prefix_and_model", "total_credits_used",)
    FUNCTION = "generate"
    CATEGORY = "DeepGen/Generators"

    def generate(self, **kwargs):
        return self.run_generation("I2I", **kwargs)

class I2I3Node(BaseTaskNode):
    @classmethod
    def INPUT_TYPES(cls):
        models = load_models_for_task("I2I3")
        return {
            "required": {
                "model": (models,),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed_value": ("INT", {"default": 1000}),
                "nb_results": ("INT", {"default": 1, "min": 1, "max": 10}),
                "output_prefix": ("STRING", {"default": ""}),
                "config_json": ("STRING", {"default": "", "multiline": True}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
            },
            "optional": {
                "minimum_resolution": ("STRING", {"default": ""}),
                "aspect_ratio": ("STRING", {"default": ""}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT",)
    RETURN_NAMES = ("IMAGE", "output_prefix_and_model", "total_credits_used",)
    FUNCTION = "generate"
    CATEGORY = "DeepGen/Generators"

    def generate(self, **kwargs):
        return self.run_generation("I2I3", **kwargs)

class I2I10Node(BaseTaskNode):
    @classmethod
    def INPUT_TYPES(cls):
        models = load_models_for_task("I2I10")
        required_images = {f"image_{i}": ("IMAGE",) for i in range(1, 11)}
        return {
            "required": {
                "model": (models,),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed_value": ("INT", {"default": 1000}),
                "nb_results": ("INT", {"default": 1, "min": 1, "max": 10}),
                "output_prefix": ("STRING", {"default": ""}),
                "config_json": ("STRING", {"default": "", "multiline": True}),
                **required_images
            },
            "optional": {
                "minimum_resolution": ("STRING", {"default": ""}),
                "aspect_ratio": ("STRING", {"default": ""}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT",)
    RETURN_NAMES = ("IMAGE", "output_prefix_and_model", "total_credits_used",)
    FUNCTION = "generate"
    CATEGORY = "DeepGen/Generators"

    def generate(self, **kwargs):
        return self.run_generation("I2I10", **kwargs)

class T2VNode(BaseTaskNode):
    @classmethod
    def INPUT_TYPES(cls):
        models = load_models_for_task("T2V")
        return {
            "required": {
                "model": (models,),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed_value": ("INT", {"default": 1000}),
                "nb_results": ("INT", {"default": 1, "min": 1, "max": 10}),
                "output_prefix": ("STRING", {"default": ""}),
                "config_json": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "aspect_ratio": ("STRING", {"default": ""}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("VIDEO", "STRING", "FLOAT",)
    RETURN_NAMES = ("VIDEO", "output_prefix_and_model", "total_credits_used",)
    FUNCTION = "generate"
    CATEGORY = "DeepGen/Generators"

    def generate(self, **kwargs):
        return self.run_generation("T2V", **kwargs)

class V2VRNode(BaseTaskNode):
    @classmethod
    def INPUT_TYPES(cls):
        models = load_models_for_task("V2VR")
        images = {}
        for i in range(1, 5):
            images[f"frontal_image_{i}"] = ("IMAGE",)
            for j in range(1, 4):
                images[f"reference_image_{i}_{j}"] = ("IMAGE",)
        
        return {
            "required": {
                "model": (models,),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed_value": ("INT", {"default": 1000}),
                "nb_results": ("INT", {"default": 1, "min": 1, "max": 10}),
                "output_prefix": ("STRING", {"default": ""}),
                "config_json": ("STRING", {"default": "", "multiline": True}),
                "start_image": ("IMAGE",),
                "end_image": ("IMAGE",),
            },
            "optional": {
                **images,
                "aspect_ratio": ("STRING", {"default": ""}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("VIDEO", "STRING", "FLOAT",)
    RETURN_NAMES = ("VIDEO", "output_prefix_and_model", "total_credits_used",)
    FUNCTION = "generate"
    CATEGORY = "DeepGen/Generators"

    def generate(self, **kwargs):
        return self.run_generation("V2VR", **kwargs)

NODE_CLASS_MAPPINGS = {
    "DeepGen_T2T": T2TNode,
    "DeepGen_I2T": I2TNode,
    "DeepGen_T2I": T2INode,
    "DeepGen_I2I": I2INode,
    "DeepGen_I2I3": I2I3Node,
    "DeepGen_I2I10": I2I10Node,
    "DeepGen_T2V": T2VNode,
    "DeepGen_V2VR": V2VRNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepGen_T2T": "LLM",
    "DeepGen_I2T": "Vision",
    "DeepGen_T2I": "Image (from text)",
    "DeepGen_I2I": "Image (from 1 image)",
    "DeepGen_I2I3": "Image (from 3 images)",
    "DeepGen_I2I10": "Image (from 10 images)",
    "DeepGen_T2V": "Video (from text)",
    "DeepGen_V2VR": "Video (from video, images and references)",
}
