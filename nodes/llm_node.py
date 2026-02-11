from .deepgen_utils import DeepGenApiHandler, DeepGenConfig

# Initialize DeepGenConfig
deepgen_config = DeepGenConfig()


class LLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "google/gemini-2.5-flash",
                        "anthropic/claude-sonnet-4.5",
                        "openai/gpt-4.1",
                        "openai/gpt-oss-120b",
                        "meta-llama/llama-4-maverick",
                        "Custom",
                    ],
                    {"default": "google/gemini-2.5-flash"},
                ),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "reasoning": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "max_tokens": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "custom_model_name": ("STRING", {"default": "", "multiline": False}),
                "alias_id": ("STRING", {"default": ""}),
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
