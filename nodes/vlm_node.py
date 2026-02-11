from .deepgen_utils import DeepGenApiHandler, DeepGenConfig, ImageUtils, ResultProcessor

# Initialize DeepGenConfig
deepgen_config = DeepGenConfig()


class VLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "alias_id": ("STRING", {"default": "deepgen/openrouter/router/vision"}),
            },
            "optional": {
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "reasoning": ("BOOLEAN", {"default": False}),
                "max_tokens": ("INT", {"default": 1024, "min": 0, "max": 100000}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "DeepGen/VLM"

    def generate_text(self, prompt, image, alias_id, system_prompt="", temperature=1.0, reasoning=False, max_tokens=1024):
        try:
            # Handle multiple images - image can be a batch
            image_urls = []

            # Check if image is a batch (4D tensor) or single image (3D tensor)
            if len(image.shape) == 4:
                # Batch of images
                for i in range(image.shape[0]):
                    single_image = image[i:i+1]
                    image_url = ImageUtils.upload_image(single_image)
                    if not image_url:
                        return DeepGenApiHandler.handle_text_generation_error(
                            alias_id, f"Failed to upload image {i+1}"
                        )
                    image_urls.append(image_url)
            else:
                # Single image
                image_url = ImageUtils.upload_image(image)
                if not image_url:
                    return DeepGenApiHandler.handle_text_generation_error(
                        alias_id, "Failed to upload image"
                    )
                image_urls.append(image_url)

            arguments = {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "image_urls": image_urls,
                "temperature": temperature,
                "reasoning": reasoning,
                "stream": False,
            }

            # Only include max_tokens if it's greater than 0
            if max_tokens > 0:
                arguments["max_tokens"] = max_tokens

            result = DeepGenApiHandler.submit_and_get_result(alias_id, arguments)
            text_result = ResultProcessor.process_text_result(result)
            return (text_result[0],)
        except Exception as e:
            return DeepGenApiHandler.handle_text_generation_error(alias_id, str(e))


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "VLM_deepgen": VLMNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "VLM_deepgen": "VLM (deepgen)",
}
