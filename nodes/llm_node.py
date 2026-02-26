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
                "output_prefix": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "FLOAT",)
    RETURN_NAMES = ("output", "reasoning", "output_prefix_and_model", "total_credits_used",)
    FUNCTION = "generate_text"
    CATEGORY = "DeepGen/LLM"

    def generate_text(self, prompt, alias_id, system_prompt="", temperature=1.0, reasoning=False, max_tokens=1024, endpoint="https://api.deepgen.app", output_prefix=""):
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
            
            res_obj = result[0] if isinstance(result, list) and len(result) > 0 else result
            if not isinstance(res_obj, dict):
                res_obj = getattr(res_obj, '__dict__', {}) or {}

            text_result = ResultProcessor.process_text_result(result)
            agent_alias = res_obj.get("agent_alias", "")
            prefixed_model = f"{output_prefix}_{agent_alias}" if output_prefix else agent_alias
            credits_out = float(res_obj.get("total_credits_used", 0.0))
            return (text_result[0], text_result[1], prefixed_model, credits_out)
        except ValueError as ve:
            raise ve
        except Exception as e:
            error_result = DeepGenApiHandler.handle_text_generation_error(alias_id, str(e))
            return (error_result[0], "", "", 0.0)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "LLM_deepgen": LLMNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM_deepgen": "LLM (deepgen)",
}
