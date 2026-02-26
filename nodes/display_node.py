class DisplayNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "text": ("STRING", {"forceInput": True}),
                "float_val": ("FLOAT", {"forceInput": True}),
                "int_val": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_NODE = True
    FUNCTION = "display_text"
    CATEGORY = "DeepGen/Display"

    def display_text(self, text=None, float_val=None, int_val=None, **kwargs):
        vals = []
        if text is not None: vals.append(str(text))
        if float_val is not None: vals.append(str(float_val))
        if int_val is not None: vals.append(str(int_val))
        
        final_text = " | ".join(vals) if vals else ""
        return {"ui": {"text": [final_text]}, "result": (final_text,)}

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "Display_deepgen": DisplayNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "Display_deepgen": "Display (deepgen)",
}
