import os
import json
import torch
import numpy as np
from PIL import Image, ImageOps
from .deepgen_utils import ResultProcessor

class UploadMultipleImagesNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_paths": ("STRING", {"multiline": True, "default": "[]"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("IMAGE", "count")
    FUNCTION = "load_images"
    CATEGORY = "DeepGen/Image"

    def load_images(self, image_paths):
        import folder_paths
        
        try:
            image_names = json.loads(image_paths)
        except Exception as e:
            print(f"UploadMultipleImagesNode: Invalid JSON in image_paths. {e}")
            return (ResultProcessor.create_blank_image()[0], 0)
            
        if not image_names or not isinstance(image_names, list):
            print("UploadMultipleImagesNode: No images provided. Returning blank image.")
            return (ResultProcessor.create_blank_image()[0], 0)
            
        input_dir = folder_paths.get_input_directory()
        
        tensors = []
        target_size = None
        
        for name in image_names:
            p = os.path.join(input_dir, name)
            if not os.path.isfile(p):
                print(f"UploadMultipleImagesNode: Warning skipping {name} (not found)")
                continue
                
            try:
                img = Image.open(p)
                img = ImageOps.exif_transpose(img)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                # Resize all subsequent images to the first image's size to ensure they can be stacked
                if target_size is None:
                    target_size = img.size
                elif img.size != target_size:
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                    
                img_array = np.array(img).astype(np.float32) / 255.0
                tensors.append(torch.from_numpy(img_array))
            except Exception as e:
                print(f"UploadMultipleImagesNode: Warning skipping {name}: {e}")
                
        if not tensors:
            return (ResultProcessor.create_blank_image()[0], 0)
            
        batch_tensor = torch.stack(tensors)
        return (batch_tensor, len(tensors))


NODE_CLASS_MAPPINGS = {
    "UploadMultipleImages_deepgen": UploadMultipleImagesNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UploadMultipleImages_deepgen": "Upload Multiple Images (deepgen)",
}
