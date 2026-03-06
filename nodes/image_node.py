from .deepgen_utils import DeepGenApiHandler as ApiHandler, ImageUtils, ResultProcessor
import torch
import os
import glob
import csv
from PIL import Image, ImageOps

# Initialize DeepGenConfig implicitly via ApiHandler usages or if needed
# deepgen_config = DeepGenConfig()

class ImageNode:
    # Base INPUT_TYPES left blank as this will be a dynamically generated subclass.
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {}}

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT",)
    RETURN_NAMES = ("IMAGE", "output_prefix_and_model", "total_credits_used",)
    FUNCTION = "generate_image"
    CATEGORY = "DeepGen/Image"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Bypass standard ComfyUI validation for dynamic combo boxes"""
        return True

    def generate_image(self, **kwargs):
        # Use properties from the subclass directly
        alias_id = getattr(self, "alias_id", "flux_schnell")
        supported_inputs = getattr(self, "supported_inputs", [])

        def unwrap(v):
            return v[0] if isinstance(v, list) and len(v) > 0 else v

        prompt = unwrap(kwargs.get("prompt", ""))
        negative_prompt = unwrap(kwargs.get("negative_prompt", ""))
        seed_value = unwrap(kwargs.get("seed_value", -1))
        num_images = unwrap(kwargs.get("num_images", 1))
        output_format = unwrap(kwargs.get("output_format", "png"))
        endpoint = unwrap(kwargs.get("endpoint", "https://api.deepgen.app"))
        output_prefix = unwrap(kwargs.get("output_prefix", ""))
        resolution = unwrap(kwargs.get("resolution", ""))
        aspect_ratio = unwrap(kwargs.get("aspect_ratio", "Auto"))
        pixel_size = unwrap(kwargs.get("pixel_size", ""))
        temperature = unwrap(kwargs.get("temperature", 0.7))
        cfg_scale = unwrap(kwargs.get("cfg_scale", 7.0))
        steps = unwrap(kwargs.get("steps", 20))
        loras = unwrap(kwargs.get("loras", ""))
        style = unwrap(kwargs.get("style", ""))
        queue = unwrap(kwargs.get("queue", False))
        auto_fix = unwrap(kwargs.get("auto_fix", False))
        enable_safety_checker = unwrap(kwargs.get("enable_safety_checker", True))
        transparent_background = unwrap(kwargs.get("transparent_background", False))
        partial_images = unwrap(kwargs.get("partial_images", 1))
        quality = unwrap(kwargs.get("quality", "standard"))

        arguments = {
            "prompt": prompt,
            "num_images": num_images,
            "output_format": output_format,
        }

        # Submitting optional inputs conditionally based on what's supported
        if "negative_prompt" in supported_inputs and negative_prompt:
            arguments["negative_prompt"] = negative_prompt
            
        if "aspect_ratio" in supported_inputs and aspect_ratio not in ("", "Auto"):
            arguments["aspect_ratio"] = aspect_ratio
        if "resolution" in supported_inputs and resolution not in ("", "Auto"):
            arguments["resolution"] = resolution
        if "pixel_size" in supported_inputs and pixel_size not in ("", "Auto"):
            arguments["pixel_size"] = pixel_size
            
        if "temperature" in supported_inputs:
            arguments["temperature"] = temperature
        if "cfg_scale" in supported_inputs:
            arguments["cfg_scale"] = cfg_scale
        if "steps" in supported_inputs:
            arguments["steps"] = steps
        if "loras" in supported_inputs and loras:
            arguments["loras"] = loras.split(",") # Assume comma separated values for loras
        if "style" in supported_inputs and style:
            arguments["style"] = style
        if "queue" in supported_inputs:
            arguments["queue"] = queue
        if "auto_fix" in supported_inputs:
            arguments["auto_fix"] = auto_fix
        if "enable_safety_checker" in supported_inputs:
            arguments["enable_safety_checker"] = enable_safety_checker
        if "transparent_background" in supported_inputs:
            arguments["transparent_background"] = transparent_background
        if "partial_images" in supported_inputs:
            arguments["partial_images"] = partial_images
        if "quality" in supported_inputs and quality:
            arguments["quality"] = quality

        if seed_value != -1:
            arguments["seed"] = seed_value

        attachments_files = []
        # Exclude standard optional fields from iterating over kwargs
        standard_kwargs = [
            "prompt", "negative_prompt", "seed_value", "num_images", "output_format", "endpoint", 
            "output_prefix", "aspect_ratio", "resolution", "pixel_size", "temperature", "cfg_scale", 
            "steps", "loras", "style", "queue", "auto_fix", "enable_safety_checker", 
            "transparent_background", "partial_images", "quality", "extra_pnginfo", "unique_id"
        ]
        
        unique_id = unwrap(kwargs.get("unique_id"))
        extra_pnginfo = unwrap(kwargs.get("extra_pnginfo"))
        original_names_map = {}

        for k, v in kwargs.items():
            if v is None:
                continue
            if k in standard_kwargs or k in ["unique_id", "extra_pnginfo"]:
                continue

            limit = 1
            prefix_base = k
            if k in ("image", "images"):
                limit = 9999
                prefix_base = "image"
            elif k in ("video", "videos"):
                limit = 9999
                prefix_base = "video"
            elif k in ("frame", "frames"):
                limit = 9999
                prefix_base = "frame"
            elif k in ("element", "elements"):
                limit = 9999
                prefix_base = "element"
            elif k in ("mask", "masks"):
                limit = 10
                prefix_base = "mask"

            if k not in original_names_map and unique_id and extra_pnginfo:
                original_names_map[k] = ImageUtils.resolve_filenames(unique_id, extra_pnginfo, k)
            original_names = original_names_map.get(k, [])
            
            def get_orig_name(idx):
                if idx < len(original_names) and original_names[idx]:
                    org = str(original_names[idx])
                    if org.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.mp4')):
                        org = org.rsplit('.', 1)[0]
                    return f"_{org}"
                return ""

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

            if prefix_base == "element":
                for i, elem_dict in enumerate(flattened_items[:limit], start=1):
                    if not isinstance(elem_dict, dict):
                        continue
                    
                    for elem_key, elem_val in elem_dict.items():
                        if elem_val is None:
                            continue
                            
                        if elem_key == "frontal_image":
                            if hasattr(elem_val, "shape"):
                                img = elem_val
                                if len(img.shape) == 4:
                                    attach = ImageUtils.get_attachment_file(img[0:1], filename=f"{prefix_base}_{i}_frontal.png")
                                else:
                                    attach = ImageUtils.get_attachment_file(img, filename=f"{prefix_base}_{i}_frontal.png")
                                if attach:
                                    attachments_files.append(attach)
                        elif elem_key == "references":
                            if hasattr(elem_val, "shape"):
                                img = elem_val
                                if len(img.shape) == 4:
                                    for r in range(min(img.shape[0], 3)):
                                        attach = ImageUtils.get_attachment_file(img[r:r+1], filename=f"{prefix_base}_{i}_ref_{r+1}.png")
                                        if attach:
                                            attachments_files.append(attach)
                                else:
                                    attach = ImageUtils.get_attachment_file(img, filename=f"{prefix_base}_{i}_ref_1.png")
                                    if attach:
                                        attachments_files.append(attach)
                continue

            for i, item in enumerate(flattened_items[:limit]):
                if hasattr(item, "shape"):
                    attach = ImageUtils.get_attachment_file(item, filename=f"{prefix_base}_{i+1}{get_orig_name(i)}.png")
                    if attach:
                        attachments_files.append(attach)
                else:
                    vid_path = None
                    if isinstance(item, str):
                        try:
                            if os.path.exists(item): vid_path = item
                        except: pass
                    elif hasattr(item, "filepath") and item.filepath:
                        try:
                            if os.path.exists(item.filepath): vid_path = item.filepath
                        except: pass
                    
                    if vid_path:
                        import base64
                        import mimetypes
                        import os as os_mod
                        mime_type, _ = mimetypes.guess_type(vid_path)
                        mime_type = mime_type or "application/octet-stream"
                        original_name = os_mod.basename(vid_path)
                        orig_n = get_orig_name(i)
                        
                        if orig_n:
                            ext = os_mod.path.splitext(original_name)[1]
                            new_filename = f"{prefix_base}_{i+1}{orig_n}{ext}"
                        else:
                            new_filename = f"{prefix_base}_{i+1}__{original_name}"
                            
                        with open(vid_path, "rb") as vf:
                            b64 = base64.b64encode(vf.read()).decode("utf-8")
                            attachments_files.append({
                                "attachment_bytes": b64,
                                "attachment_mime_type": mime_type,
                                "attachment_file_name": new_filename
                            })

        if attachments_files:
            arguments["attachments_files"] = attachments_files

        try:
            result = ApiHandler.submit_and_get_result(alias_id, arguments, api_url=endpoint)
            # The API returns a list [WebResponse(...)] or a dict depending on endpoint.
            # We better extract properties safely:
            res_obj = result[0] if isinstance(result, list) and len(result) > 0 else result
            img_tensor = ResultProcessor.process_image_result(result)[0]
            
            def _get_attr(obj, key, default=None):
                if isinstance(obj, dict): return obj.get(key, default)
                return getattr(obj, key, default)
                
            agent_alias = _get_attr(res_obj, "agent_alias", "")
            prefixed_model = f"{output_prefix}_{agent_alias}" if output_prefix else agent_alias
            
            cred = _get_attr(res_obj, "total_credits_used")
            if cred is None:
                out_obj = _get_attr(res_obj, "output", {})
                cred = _get_attr(out_obj, "total_credits_used", _get_attr(res_obj, "aiCredits", 0.0))
            credits_out = float(cred or 0.0)
            return (img_tensor, prefixed_model, credits_out)
        except ValueError as ve:
            raise ve
        except Exception as e:
            #rint(f"Error generating image : {str(e)}")
            blank_img = ApiHandler.handle_image_generation_error("ImageNode", e)[0]
            return (blank_img, "", 0.0)


    @staticmethod
    def handle_image_generation_error(model_name, error):
        """Handle image generation errors consistently."""
        import traceback
        error_details = traceback.format_exc()
        #rint(f"Error generating image with {model_name}: {str(error)}")
        #rint(f"Traceback: {error_details}")
        return ResultProcessor.create_blank_image()



