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
                reader = csv.DictReader(f)
                for row in reader:
                    cls.models_list.append(row["name"])
                    cls.models_map[row["name"]] = row["value"]
        except Exception as e:
            print(f"DeepGen: Failed to load models.csv: {e}")
            cls.models_list = ["Flux Schnell"]
            cls.models_map = {"Flux Schnell": "flux_schnell"}

        optional_inputs = {
            "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            "seed_value": ("INT", {"default": -1}),
            "steps": ("INT", {"default": 4, "min": 1, "max": 40}),
            "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
            "enable_safety_checker": ("BOOLEAN", {"default": True}),
            "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
            "image": ("IMAGE",),
            "mask_image": ("IMAGE",),
            "model": (cls.models_list, {"default": cls.models_list[0] if cls.models_list else ""}),
            "loras": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            "endpoint": ("STRING", {"default": "https://api.deepgen.app"}),
        }


        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 768, "min": 256, "max": 4096, "step": 8}),
            },
            "optional": optional_inputs,
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
        model="Flux Schnell",
        loras="", # Supports "URL" or "URL, scale" (e.g. "https://..., 0.8") per line
        endpoint="https://api.deepgen.app",
        **kwargs
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

        # Lookup alias_id from the selected model name
        alias_id = self.models_map.get(model, "flux_schnell")


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
                        pass
                        #rint(f"Warning: Invalid scale for LoRA {path}, defaulting to 1.0")
                
                loras_list.append({"path": path, "scale": scale})

            if loras_list:
                arguments["loras"] = loras_list


        if seed_value != -1:
            arguments["seed"] = seed_value

        # Handle images if provided
        images_to_process = []
        if image is not None:
            images_to_process.append(image)

        for k, v in kwargs.items():
            if k.startswith('image_') and v is not None:
                images_to_process.append(v)

        attachments_files = []
        for img in images_to_process:
            if len(img.shape) == 4:
                for i in range(img.shape[0]):
                    single_image = img[i:i+1]
                    attach = ImageUtils.get_attachment_file(single_image, filename=f"image_{len(attachments_files)}.png")
                    if attach:
                        attachments_files.append(attach)
            else:
                attach = ImageUtils.get_attachment_file(img, filename=f"image_{len(attachments_files)}.png")
                if attach:
                    attachments_files.append(attach)

        if attachments_files:
            arguments["attachments_files"] = attachments_files
        
        if mask_image is not None:
            # Note: Mask may also need to be sent differently if /upload is unsupported.
            # Currently fallback to old behavior, but ideally would be attachments_files too
            # or integrated alongside other attachments.
            mask_attach = ImageUtils.get_attachment_file(mask_image, filename="mask.png")
            if mask_attach:
                if "attachments_files" not in arguments:
                    arguments["attachments_files"] = []
                arguments["attachments_files"].append(mask_attach)

        try:
            result = ApiHandler.submit_and_get_result(alias_id, arguments, api_url=endpoint)
            return ResultProcessor.process_image_result(result)
        except ValueError as ve:
            raise ve
        except Exception as e:
            #rint(f"Error generating image : {str(e)}")
            return ApiHandler.handle_image_generation_error("ImageNode", e)


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
