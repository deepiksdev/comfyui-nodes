from .deepgen_utils import DeepGenApiHandler, DeepGenConfig, ResultProcessor

# Initialize DeepGenConfig
deepgen_config = DeepGenConfig()


class LLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "alias_id": ("STRING", {"default": ""}),
            },
            "optional": {
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "reasoning": ("BOOLEAN", {"default": False}),
                "max_tokens": ("INT", {"default": 0, "min": 0, "max": 100000}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("output", "reasoning",)
    FUNCTION = "generate_text"
    CATEGORY = "DeepGen/LLM"

    def generate_text(self, prompt, model, system_prompt, temperature, reasoning, max_tokens=0, custom_model_name="", alias_id=None):
        try:
            # Handle custom model selection
            if model == "Custom":
                if not custom_model_name or custom_model_name.strip() == "":
                    error_result = DeepGenApiHandler.handle_text_generation_error(
                        "Custom", "Custom model name is required when 'Custom' is selected"
                    )
                    return (error_result[0], "")
                model = custom_model_name.strip()

            arguments = {
                "model": model,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "reasoning": reasoning,
                "stream": False,
            }

            # Only include max_tokens if it's greater than 0
            if max_tokens > 0:
                arguments["max_tokens"] = max_tokens

            result = DeepGenApiHandler.submit_and_get_result(alias_id if alias_id else "deepgen/openrouter/router", arguments)

            return ResultProcessor.process_text_result(result)
        except Exception as e:
            error_result = DeepGenApiHandler.handle_text_generation_error(model, str(e))
            return (error_result[0], "")


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "LLM_deepgen": LLMNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM_deepgen": "LLM (deepgen)",
}
