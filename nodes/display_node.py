class DisplayFloatNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float_val": ("FLOAT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_NODE = True
    FUNCTION = "display_float"
    CATEGORY = "DeepGen/Display"

    def display_float(self, float_val):
        text_val = str(float_val)
        return {"ui": {"text": [text_val]}, "result": (text_val,)}

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "DisplayFloat_deepgen": DisplayFloatNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "DisplayFloat_deepgen": "Display Float (deepgen)",
}
