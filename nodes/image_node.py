from .deepgen_utils import DeepGenApiHandler as ApiHandler, ImageUtils, ResultProcessor
import torch
import os
import glob
import csv
from PIL import Image, ImageOps

# Initialize DeepGenConfig implicitly via ApiHandler usages or if needed
# deepgen_config = DeepGenConfig()

class ImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Load models from CSV
        cls.models_list = []
        cls.models_map = {}
        cls.supported_inputs_map = {}

        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models.csv")
        try:
            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 11 and row[10].strip() == "T2I":
                        cls.models_list.append(row[1])
                        cls.models_map[row[1]] = row[0]
                        cls.supported_inputs_map[row[0]] = [x.strip() for x in row[2].split(",")] if row[2].strip() else []
        except Exception as e:
            print(f"DeepGen: Failed to load models.csv: {e}")
            cls.models_list = ["Flux Schnell"]
            cls.models_map = {"Flux Schnell": "flux_schnell"}
            cls.supported_inputs_map = {"flux_schnell": []}

        optional_inputs = {
            "seed_value": ("INT", {"default": -1}),
            "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
            "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
            "endpoint": ("STRING", {"default": "https://api.deepgen.app"}),
            "output_prefix": ("STRING", {"default": ""}),
            
            # Dynamic combo fields
            "resolution": ([""], {"default": ""}),
            "aspect_ratio": (["Auto"], {"default": "Auto"}),
            "pixel_size": ([""], {"default": ""}),
            
            # Additional T2I fields
            "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            "cfg_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.5}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 150}),
            "loras": ("STRING", {"default": ""}),
            "style": ("STRING", {"default": ""}),
            "queue": ("BOOLEAN", {"default": False}),
            "auto_fix": ("BOOLEAN", {"default": False}),
            "enable_safety_checker": ("BOOLEAN", {"default": True}),
            "transparent_background": ("BOOLEAN", {"default": False}),
            "partial_images": ("INT", {"default": 1, "min": 1, "max": 100}),
            "quality": ("STRING", {"default": "standard"}),
        }

        return {
            "required": {
                "model": ([""] + cls.models_list, {"default": ""}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": optional_inputs,
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT",)
    RETURN_NAMES = ("IMAGE", "output_prefix_and_model", "total_credits_used",)
    FUNCTION = "generate_image"
    CATEGORY = "DeepGen/Image"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Bypass standard ComfyUI validation for dynamic combo boxes"""
        return True

    def generate_image(
        self,
        model,
        prompt,
        negative_prompt="",
        seed_value=-1,
        num_images=1,
        output_format="png",
        endpoint="https://api.deepgen.app",
        output_prefix="",
        resolution="",
        aspect_ratio="Auto",
        pixel_size="",
        temperature=0.7,
        cfg_scale=7.0,
        steps=20,
        loras="",
        style="",
        queue=False,
        auto_fix=False,
        enable_safety_checker=True,
        transparent_background=False,
        partial_images=1,
        quality="standard",
        **kwargs
    ):
        # Lookup alias_id from the selected model name
        alias_id = getattr(self, "models_map", {}).get(model, "flux_schnell")
        supported_inputs = getattr(self, "supported_inputs_map", {}).get(alias_id, [])

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
            "transparent_background", "partial_images", "quality"
        ]
        
        for k, v in kwargs.items():
            if v is None:
                continue
            if k in standard_kwargs:
                continue

            if k.startswith("element_") and isinstance(v, dict):
                for elem_key, elem_val in v.items():
                    if elem_val is None:
                        continue
                    
                    vid_path = None
                    if isinstance(elem_val, str):
                        try:
                            if os.path.exists(elem_val):
                                vid_path = elem_val
                        except Exception:
                            pass
                    elif hasattr(elem_val, "filepath") and elem_val.filepath:
                        try:
                            if os.path.exists(elem_val.filepath):
                                vid_path = elem_val.filepath
                        except Exception:
                            pass
                    
                    prefix = f"{k}_{elem_key}"
                    
                    if vid_path:
                        import base64
                        import mimetypes
                        import os as os_mod
                        mime_type, _ = mimetypes.guess_type(vid_path)
                        mime_type = mime_type or "application/octet-stream"
                        original_name = os_mod.basename(vid_path)
                        new_filename = f"{prefix}___{original_name}"
                        with open(vid_path, "rb") as vf:
                            b64 = base64.b64encode(vf.read()).decode("utf-8")
                            attachments_files.append({
                                "attachment_bytes": b64,
                                "attachment_mime_type": mime_type,
                                "attachment_file_name": new_filename
                            })
                        continue
                    
                    if hasattr(elem_val, "shape"):
                        img = elem_val
                        if len(img.shape) == 4:
                            for i in range(img.shape[0]):
                                single_image = img[i:i+1]
                                attach = ImageUtils.get_attachment_file(single_image, filename=f"{prefix}___{i}.png")
                                if attach:
                                    attachments_files.append(attach)
                        else:
                            attach = ImageUtils.get_attachment_file(img, filename=f"{prefix}___image.png")
                            if attach:
                                attachments_files.append(attach)
                continue

            vid_path = None
            if isinstance(v, str):
                try:
                    if os.path.exists(v):
                        vid_path = v
                except Exception:
                    pass
            elif hasattr(v, "filepath") and v.filepath:
                try:
                    if os.path.exists(v.filepath):
                        vid_path = v.filepath
                except Exception:
                    pass
            
            if vid_path:
                import base64
                import mimetypes
                import os as os_mod
                mime_type, _ = mimetypes.guess_type(vid_path)
                mime_type = mime_type or "application/octet-stream"
                original_name = os_mod.basename(vid_path)
                new_filename = f"{k}__{original_name}"
                with open(vid_path, "rb") as vf:
                    b64 = base64.b64encode(vf.read()).decode("utf-8")
                    attachments_files.append({
                        "attachment_bytes": b64,
                        "attachment_mime_type": mime_type,
                        "attachment_file_name": new_filename
                    })
                continue
            
            if hasattr(v, "shape"):
                img = v
                if len(img.shape) == 4:
                    for i in range(img.shape[0]):
                        single_image = img[i:i+1]
                        attach = ImageUtils.get_attachment_file(single_image, filename=f"{k}__{i}.png")
                        if attach:
                            attachments_files.append(attach)
                else:
                    attach = ImageUtils.get_attachment_file(img, filename=f"{k}__image.png")
                    if attach:
                        attachments_files.append(attach)

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


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "Image_deepgen": ImageNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "Image_deepgen": "Image (deepgen)",
}
