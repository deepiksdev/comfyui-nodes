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
        cls.models_map = {} # Map from name to value (alias_id)
        
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models.csv")
        try:
            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 10 and row[9].strip() == "T2I":
                        cls.models_list.append(row[1])
                        cls.models_map[row[1]] = row[0]
        except Exception as e:
            print(f"DeepGen: Failed to load models.csv: {e}")
            cls.models_list = ["Flux Schnell"]
            cls.models_map = {"Flux Schnell": "flux_schnell"}

        optional_inputs = {
            "seed_value": ("INT", {"default": -1}),
            "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
            "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
            "endpoint": ("STRING", {"default": "https://api.deepgen.app"}),
            "output_prefix": ("STRING", {"default": ""}),
            "aspect_ratio": (["Auto"], {"default": "Auto"}),
            "resolution": ([""], {"default": ""}),
            "pixel_size": ([""], {"default": ""}),
        }


        return {
            "required": {
                "model": (cls.models_list, {"default": cls.models_list[0] if cls.models_list else ""}),
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
        aspect_ratio=None,
        resolution=None,
        pixel_size=None,
        **kwargs
    ):
        arguments = {
            "prompt": prompt,
            "num_images": num_images,
            "negative_prompt": negative_prompt,
            "output_format": output_format,
        }

        if aspect_ratio is not None and aspect_ratio not in ("", "Auto"):
            arguments["aspect_ratio"] = aspect_ratio
        if resolution is not None and resolution not in ("", "Auto"):
            arguments["resolution"] = resolution
        if pixel_size is not None and pixel_size not in ("", "Auto"):
            arguments["pixel_size"] = pixel_size

        # Lookup alias_id from the selected model name
        alias_id = self.models_map.get(model, "flux_schnell")


        if seed_value != -1:
            arguments["seed"] = seed_value

        attachments_files = []
        for k, v in kwargs.items():
            if v is None:
                continue
            if k in ["prompt", "negative_prompt", "seed_value", "num_images", "output_format", "endpoint", "output_prefix", "aspect_ratio", "resolution", "pixel_size"]:
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
            if not isinstance(res_obj, dict):
                res_obj = getattr(res_obj, '__dict__', {}) or {}
            
            img_tensor = ResultProcessor.process_image_result(result)[0]
            agent_alias = res_obj.get("agent_alias", "")
            prefixed_model = f"{output_prefix}_{agent_alias}" if output_prefix else agent_alias
            credits_out = float(res_obj.get("total_credits_used", 0.0))
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
