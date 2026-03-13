from .task_utils import BaseTaskNode, load_models_for_task

class T2INode(BaseTaskNode):
    @classmethod
    def INPUT_TYPES(cls):
        models = load_models_for_task("T2I")
        return {
            "required": {
                "model": (models,),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed_value": ("INT", {"default": 1000}),
                "nb_results": ("INT", {"default": 1, "min": 1, "max": 10}),
                "output_prefix": ("STRING", {"default": ""}),
                "config_json": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "minimum_resolution": (["500", "1K", "2K", "4K"], {"default": "1K"}),
                "aspect_ratio": (["1:1", "9:16", "16:9", "3:4", "4:3", "3:2", "2:3", "5:4", "4:5", "21:9", "4:1", "1:4", "8:1", "1:8"], {"default": "1:1"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT",)
    RETURN_NAMES = ("IMAGE", "output_prefix_and_model", "total_credits_used",)
    FUNCTION = "generate"
    CATEGORY = "DeepGen/Generators"

    def generate(self, **kwargs):
        return self.run_generation("T2I", **kwargs)
