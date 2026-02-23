import os
import tempfile
import zipfile
import torch
from PIL import Image
from .deepgen_utils import DeepGenApiHandler, DeepGenConfig, ImageUtils, ResultProcessor

# Initialize DeepGenConfig
deepgen_config = DeepGenConfig()


def create_zip_from_images(images):
    """Create a zip file from a list of images."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip:
            with zipfile.ZipFile(temp_zip, "w") as zf:
                for idx, img_tensor in enumerate(images):
                    # Convert tensor to PIL Image
                    if isinstance(img_tensor, torch.Tensor):
                        # Convert to numpy and scale to 0-255 range
                        img_np = (img_tensor.cpu().numpy() * 255).astype("uint8")
                        # Handle different tensor formats
                        if img_np.shape[0] == 3:  # If in format (C, H, W)
                            img_np = img_np.transpose(1, 2, 0)
                        img = Image.fromarray(img_np)
                    else:
                        img = img_tensor

                    # Save image to temporary file
                    with tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False
                    ) as temp_img:
                        img.save(temp_img, format="PNG")
                        temp_img_path = temp_img.name

                    # Add to zip file
                    zf.write(temp_img_path, f"image_{idx}.png")
                    os.unlink(temp_img_path)

            # Use ImageUtils.upload_file instead of FalConfig client
            return ImageUtils.upload_file(temp_zip.name)
    except Exception as e:
        #rint(f"Failed to create zip file: {str(e)}")
        return None


class TrainerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 1000, "min": 1, "max": 10000, "step": 10}),
            },
            "optional": {
                "images": ("IMAGE",),
                "training_data_url": ("STRING", {"default": ""}),
                "trigger_word": ("STRING", {"default": ""}),
                "learning_rate": ("FLOAT", {"default": 0.0001, "min": 0.00001, "max": 0.01}),
                "create_masks": ("BOOLEAN", {"default": True}),
                "is_style": ("BOOLEAN", {"default": False}),
                "do_caption": ("BOOLEAN", {"default": True}),
                "auto_scale_input": ("BOOLEAN", {"default": True}),
                "alias_id": ("STRING", {"default": "deepgen/flux-lora-fast-training"}),
                "endpoint": ("STRING", {"default": "https://api.deepgen.app"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_file_url",)
    FUNCTION = "train"
    CATEGORY = "DeepGen/Training"

    def train(
        self,
        steps,
        images=None,
        training_data_url="",
        trigger_word="",
        learning_rate=0.0001,
        create_masks=True,
        is_style=False,
        do_caption=True,
        auto_scale_input=True,
        alias_id="deepgen/flux-lora-fast-training",
        endpoint="https://api.deepgen.app",
    ):
        try:
            # Handle training data
            data_url = training_data_url
            if images is not None and not data_url:
                data_url = create_zip_from_images(images)
            
            if not data_url:
                return ("Error: No training data provided",)

            # Prepare arguments
            arguments = {
                "steps": steps,
                "learning_rate": learning_rate,
                "trigger_word": trigger_word,
                "trigger_phrase": trigger_word, # Some use phrase
                "create_masks": create_masks,
                "is_style": is_style,
                "do_caption": do_caption,
                "auto_scale_input": auto_scale_input,
            }

            # Map to specific field name if needed by certain models
            if "flux-lora" in alias_id:
                arguments["images_data_url"] = data_url
            elif "hunyuan" in alias_id:
                arguments["images_data_url"] = data_url
            else:
                arguments["training_data_url"] = data_url

            # Submit training job
            result = DeepGenApiHandler.submit_and_get_result(alias_id, arguments, api_url=endpoint)
            return ResultProcessor.process_file_result(result)

        except Exception as e:
            return (f"Error: {str(e)}",)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "Trainer_deepgen": TrainerNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "Trainer_deepgen": "Trainer (deepgen)",
}
