from .deepgen_utils import DeepGenApiHandler, DeepGenConfig, ImageUtils, ResultProcessor

# Initialize DeepGenConfig
deepgen_config = DeepGenConfig()


class VLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {
            "image": ("IMAGE",),
            "system_prompt": ("STRING", {"default": "", "multiline": True}),
            "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            "reasoning": ("BOOLEAN", {"default": False}),
            "max_tokens": ("INT", {"default": 1024, "min": 0, "max": 100000}),
            "attachments_urls": ("STRING", {"default": "", "multiline": True}),
            "endpoint": ("STRING", {"default": "https://api.deepgen.app"}),
        }
        
        # Add 15 additional image sockets natively
        for i in range(1, 16):
            optional_inputs[f"image_{i}"] = ("IMAGE",)

        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "alias_id": ("STRING", {"default": "deepgen/openrouter/router/vision"}),
            },
            "optional": optional_inputs,
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "DeepGen/VLM"

    def generate_text(self, prompt, alias_id, image=None, system_prompt="", temperature=1.0, reasoning=False, max_tokens=1024, attachments_urls=None, endpoint="https://api.deepgen.app", **kwargs):
        try:
            image_urls = []
            attachments_files = []
            
            # Collect all explicitly named images and any dynamic ones from kwargs
            images_to_process = []
            if image is not None:
                images_to_process.append(image)
                
            for k, v in kwargs.items():
                if k.startswith('image_') and v is not None:
                    images_to_process.append(v)
            
            # Handle list of attachments if provided via URL
            if attachments_urls:
                if isinstance(attachments_urls, list):
                    image_urls.extend([u for u in attachments_urls if isinstance(u, str) and u.strip()])
                elif isinstance(attachments_urls, str):
                    image_urls.extend([u.strip() for u in attachments_urls.split("\n") if u.strip()])

            # Process all images collected
            for img in images_to_process:
                # Check if image is a batch (4D tensor) or single image (3D tensor)
                if len(img.shape) == 4:
                    # Batch of images
                    for i in range(img.shape[0]):
                        single_image = img[i:i+1]
                        attach = ImageUtils.get_attachment_file(single_image, filename=f"image_{len(attachments_files)}.png")
                        if attach:
                            attachments_files.append(attach)
                else:
                    # Single image
                    attach = ImageUtils.get_attachment_file(img, filename=f"image_{len(attachments_files)}.png")
                    if attach:
                        attachments_files.append(attach)

            arguments = {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "attachments_urls": image_urls,
                "attachments_files": attachments_files,
                "temperature": temperature,
                "reasoning": reasoning,
                "stream": False,
            }

            # Only include max_tokens if it's greater than 0
            if max_tokens > 0:
                arguments["max_tokens"] = max_tokens

            result = DeepGenApiHandler.submit_and_get_result(alias_id, arguments, api_url=endpoint)
            text_result = ResultProcessor.process_text_result(result)
            return (text_result[0],)
        except ValueError as ve:
            raise ve
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
