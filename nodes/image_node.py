from .deepgen_utils import DeepGenApiHandler as ApiHandler, ImageUtils, ResultProcessor
import torch
import os
import glob
from PIL import Image, ImageOps

# Initialize DeepGenConfig implicitly via ApiHandler usages or if needed
# deepgen_config = DeepGenConfig()

class ImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 768, "min": 256, "max": 4096, "step": 8}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed_value": ("INT", {"default": -1}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 40}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "image": ("IMAGE",),
                "mask_image": ("IMAGE",),
                "alias_id": ("STRING", {"default": "flux_schnell"}),
                "loras": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "endpoint": ("STRING", {"default": "https://api.deepgen.app"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "DeepGen/Image"

    def generate_image(
        self,
        prompt,
        width,
        height,
        negative_prompt="",
        seed_value=-1,
        steps=28, # Fixed argument name to match parameter
        guidance_scale=3.5,
        num_images=1,
        enable_safety_checker=True,
        output_format="png",
        image=None,
        mask_image=None,
        alias_id="deepgen/flux/dev",
        loras="", # Supports "URL" or "URL, scale" (e.g. "https://..., 0.8") per line
        endpoint="https://api.deepgen.app",
    ):
        arguments = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "enable_safety_checker": enable_safety_checker,
            "output_format": output_format,
        }

        if loras:
            # Parse loras string into a list of dictionaries
            loras_list = []
            for l in loras.splitlines():
                if not l.strip():
                    continue
                parts = l.split(",")
                path = parts[0].strip()
                scale = 1.0
                if len(parts) > 1:
                    try:
                        scale = float(parts[1].strip())
                    except ValueError:
                        print(f"Warning: Invalid scale for LoRA {path}, defaulting to 1.0")
                
                loras_list.append({"path": path, "scale": scale})

            if loras_list:
                arguments["loras"] = loras_list


        if seed_value != -1:
            arguments["seed"] = seed_value

        # Handle image and mask if provided
        if image is not None:
            image_urls = ImageUtils.prepare_images(image)
            if image_urls:
                arguments["image_urls"] = image_urls
        
        if mask_image is not None:
            mask_url = ImageUtils.upload_image(mask_image)
            if mask_url:
                arguments["mask_url"] = mask_url

        try:
            result = ApiHandler.submit_and_get_result(alias_id, arguments, api_url=endpoint)
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("ImageNode", e)


    @staticmethod
    def handle_image_generation_error(model_name, error):
        """Handle image generation errors consistently."""
        import traceback
        error_details = traceback.format_exc()
        print(f"Error generating image with {model_name}: {str(error)}")
        print(f"Traceback: {error_details}")
        return ResultProcessor.create_blank_image()


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "Image_deepgen": ImageNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "Image_deepgen": "Image (deepgen)",
}
