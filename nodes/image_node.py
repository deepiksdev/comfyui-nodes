from .deepgen_utils import DeepGenApiHandler as ApiHandler, ImageUtils, ResultProcessor
import torch

# Initialize DeepGenConfig implicitly via ApiHandler usages or if needed
# deepgen_config = DeepGenConfig()

class ImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (
                    [
                        "square_hd",
                        "square",
                        "portrait_4_3",
                        "portrait_16_9",
                        "landscape_4_3",
                        "landscape_16_9",
                        "custom",
                    ],
                    {"default": "landscape_4_3"},
                ),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 768, "min": 256, "max": 4096, "step": 8}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": -1}),
                "num_inference_steps": ("INT", {"default": 4, "min": 1, "max": 40}),
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
        image_size,
        width,
        height,
        negative_prompt="",
        seed=-1,
        num_inference_steps=28,
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
            "num_inference_steps": num_inference_steps,
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

        # Handle image size
        if image_size == "custom":
            arguments["image_size"] = {"width": width, "height": height}
        else:
            arguments["image_size"] = image_size

        if seed != -1:
            arguments["seed"] = seed

        # Handle image and mask if provided
        if image is not None:
            image_url = ImageUtils.upload_image(image)
            if image_url:
                arguments["image_url"] = image_url
        
        if mask_image is not None:
            mask_url = ImageUtils.upload_image(mask_image)
            if mask_url:
                arguments["mask_url"] = mask_url

        try:
            result = ApiHandler.submit_and_get_result(alias_id, arguments, api_url=endpoint)
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("ImageNode", e)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "Image_deepgen": ImageNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "Image_deepgen": "Image (deepgen)",
}
