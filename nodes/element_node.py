class ElementNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frontal_image": ("IMAGE",),
                "references": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("ELEMENT",)
    RETURN_NAMES = ("ELEMENT",)
    FUNCTION = "create_element"
    CATEGORY = "DeepGen/Elements"

    def create_element(self, frontal_image, references):
        return ({"frontal_image": frontal_image, "references": references},)

NODE_CLASS_MAPPINGS = {
    "Element_deepgen": ElementNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Element_deepgen": "Element (deepgen)"
}
