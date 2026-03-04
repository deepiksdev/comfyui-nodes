class ElementProducerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frontal": ("IMAGE",),
            },
            "optional": {
                "reference1": ("IMAGE",),
                "reference2": ("IMAGE",),
                "reference3": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("ELEMENT",)
    FUNCTION = "produce_element"
    CATEGORY = "DeepGen/Element"

    def produce_element(self, frontal, reference1=None, reference2=None, reference3=None):
        element = {
            "frontal": frontal,
        }
        if reference1 is not None:
            element["reference1"] = reference1
        if reference2 is not None:
            element["reference2"] = reference2
        if reference3 is not None:
            element["reference3"] = reference3
            
        return (element,)

NODE_CLASS_MAPPINGS = {
    "ElementProducer_deepgen": ElementProducerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ElementProducer_deepgen": "Element Producer (deepgen)",
}
