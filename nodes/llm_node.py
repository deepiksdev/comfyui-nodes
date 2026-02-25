from .deepgen_utils import DeepGenApiHandler, DeepGenConfig, ResultProcessor

# Initialize DeepGenConfig
deepgen_config = DeepGenConfig()


class LLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "alias_id": ("STRING", {"default": "deepgen/openrouter/router"}),
            },
            "optional": {
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "reasoning": ("BOOLEAN", {"default": False}),
                "max_tokens": ("INT", {"default": 1024, "min": 0, "max": 100000}),
                "endpoint": ("STRING", {"default": "https://api.deepgen.app"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("output", "reasoning",)
    FUNCTION = "generate_text"
    CATEGORY = "DeepGen/LLM"

    def generate_text(self, prompt, alias_id, system_prompt="", temperature=1.0, reasoning=False, max_tokens=1024, endpoint="https://api.deepgen.app"):
        try:
            arguments = {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "reasoning": reasoning,
                "stream": False,
            }

            # Only include max_tokens if it's greater than 0
            if max_tokens > 0:
                arguments["max_tokens"] = max_tokens

            result = DeepGenApiHandler.submit_and_get_result(alias_id, arguments, api_url=endpoint)

            return ResultProcessor.process_text_result(result)
        except ValueError as ve:
            raise ve
        except Exception as e:
            error_result = DeepGenApiHandler.handle_text_generation_error(alias_id, str(e))
            return (error_result[0], "")


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "LLM_deepgen": LLMNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM_deepgen": "LLM (deepgen)",
}
