from .task_utils import BaseTaskNode, load_models_for_task

class I2TNode(BaseTaskNode):
    @classmethod
    def INPUT_TYPES(cls):
        models = load_models_for_task("I2T")
        optional_images = {f"image_{i}": ("IMAGE",) for i in range(1, 11)}
        return {
            "required": {
                "model": (models,),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed_value": ("INT", {"default": 1000}),
                "config_json": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                **optional_images,
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT",)
    RETURN_NAMES = ("output", "model", "total_credits_used",)
    FUNCTION = "generate"
    CATEGORY = "DeepGen/Generators"

    def generate(self, **kwargs):
        return self.run_generation("I2T", **kwargs)
