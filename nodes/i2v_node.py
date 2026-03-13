from .task_utils import BaseTaskNode, load_models_for_task

class I2VNode(BaseTaskNode):
    @classmethod
    def INPUT_TYPES(cls):
        models = load_models_for_task("I2V")
        return {
            "required": {
                "model": (models,),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed_value": ("INT", {"default": 1000}),
                "nb_results": ("INT", {"default": 1, "min": 1, "max": 10}),
                "output_prefix": ("STRING", {"default": ""}),
                "config_json": ("STRING", {"default": "", "multiline": True}),
                "start_image": ("IMAGE",),
            },
            "optional": {
                "aspect_ratio": ("STRING", {"default": ""}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("VIDEO", "STRING", "FLOAT",)
    RETURN_NAMES = ("VIDEO", "output_prefix_and_model", "total_credits_used",)
    FUNCTION = "generate"
    CATEGORY = "DeepGen/Generators"

    def generate(self, **kwargs):
        return self.run_generation("I2V", **kwargs)
