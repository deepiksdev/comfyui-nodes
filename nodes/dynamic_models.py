import os
import csv
from .image_node import ImageNode
from .video_node import VideoNode
from .llm_node import LLMNode

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def load_models():
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models.csv")
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 11:
                    continue
                
                alias_id = row[0]
                name = row[1]
                supported_inputs = [x.strip() for x in row[2].split(",")] if row[2].strip() else []
                aspect_ratios = [x.strip() for x in row[3].split(",")] if row[3].strip() else []
                resolutions = [x.strip() for x in row[4].split(",")] if row[4].strip() else []
                pixel_sizes = [x.strip() for x in row[5].split(",")] if row[5].strip() else []
                
                try: num_images = int(row[6]) if row[6].strip() else 0
                except ValueError: num_images = 1
                try: num_videos = int(row[7]) if row[7].strip() else 0
                except ValueError: num_videos = 0
                try: num_elements = int(row[8]) if row[8].strip() else 0
                except ValueError: num_elements = 0
                try: num_frames = int(row[9]) if row[9].strip() else 0
                except ValueError: num_frames = 0
                
                model_type = row[10].strip()
                
                # Sanitize class name
                class_name = f"DeepGen_{alias_id.replace('-', '_').replace('.', '_')}"
                
                if model_type == "T2I":
                    base_class = ImageNode
                elif model_type == "T2V":
                    base_class = VideoNode
                elif model_type == "LLM":
                    base_class = LLMNode
                else:
                    continue

                def make_input_types(bt, sup_inps, aspect, res, pix, n_img, n_vid, n_elem, n_frames):
                    @classmethod
                    def INPUT_TYPES(cls):
                        required = {
                            "prompt": ("STRING", {"default": "", "multiline": True}),
                        }
                        
                        optional = {}
                        
                        if bt == ImageNode:
                            optional.update({
                                "seed_value": ("INT", {"default": -1}),
                                "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                                "endpoint": ("STRING", {"default": "https://api.deepgen.app"}),
                                "output_prefix": ("STRING", {"default": ""}),
                            })
                        elif bt == VideoNode:
                            optional.update({
                                "seed_value": ("INT", {"default": -1}),
                                "variations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                                "endpoint": ("STRING", {"default": "https://api.deepgen.app"}),
                                "output_prefix": ("STRING", {"default": ""}),
                            })
                        elif bt == LLMNode:
                            optional.update({
                                "seed_value": ("INT", {"default": -1}),
                                "endpoint": ("STRING", {"default": "https://api.deepgen.app"}),
                                "output_prefix": ("STRING", {"default": ""}),
                            })

                        # Dynamic Combo fields
                        if "aspect_ratio" in sup_inps and aspect:
                            optional["aspect_ratio"] = (aspect, {"default": aspect[0]})
                        if "resolution" in sup_inps and res:
                            optional["resolution"] = (res, {"default": res[0]})
                        if "pixel_size" in sup_inps and pix:
                            optional["pixel_size"] = (pix, {"default": pix[0]})

                        # Other fields
                        all_optional = {
                            "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                            "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                            "cfg_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.5}),
                            "steps": ("INT", {"default": 20, "min": 1, "max": 150}),
                            "loras": ("STRING", {"default": ""}),
                            "style": ("STRING", {"default": ""}),
                            "queue": ("BOOLEAN", {"default": False}),
                            "duration": (["5", "10"], {"default": "5"}),
                            "generate_audio": ("BOOLEAN", {"default": False}),
                            "shot_type": ("STRING", {"default": ""}),
                            "auto_fix": ("BOOLEAN", {"default": False}),
                            "enable_safety_checker": ("BOOLEAN", {"default": True}),
                            "safety_tolerance": ("STRING", {"default": "Auto"}),
                            "transparent_background": ("BOOLEAN", {"default": False}),
                            "partial_images": ("INT", {"default": 1, "min": 1, "max": 100}),
                            "quality": ("STRING", {"default": "standard"}),
                            "loop": ("BOOLEAN", {"default": False}),
                        }

                        for inp in sup_inps:
                            if inp in all_optional:
                                optional[inp] = all_optional[inp]

                        # Sockets
                        if n_img == 1:
                            optional["image"] = ("IMAGE",)
                        elif n_img > 1:
                            optional["images"] = ("IMAGE",)
                            
                        if n_vid == 1:
                            optional["video"] = ("VIDEO",)
                        elif n_vid > 1:
                            optional["videos"] = ("VIDEO",)
                            
                        if n_frames == 1:
                            optional["frame"] = ("IMAGE",)
                        elif n_frames > 1:
                            optional["frames"] = ("IMAGE",)
                            
                        if n_elem == 1:
                            optional["element"] = ("ELEMENT",)
                        elif n_elem > 1:
                            optional["elements"] = ("ELEMENT",)
                            
                        if "masks" in sup_inps:
                            optional["masks"] = ("IMAGE",)
                        elif "mask" in sup_inps or "mask_image" in sup_inps:
                            optional["mask"] = ("IMAGE",)


                        return {"required": required, "optional": optional, "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"}}
                    return INPUT_TYPES

                new_class = type(class_name, (base_class,), {
                    "INPUT_TYPES": make_input_types(base_class, supported_inputs, aspect_ratios, resolutions, pixel_sizes, num_images, num_videos, num_elements, num_frames),
                    "alias_id": alias_id,
                    "supported_inputs": supported_inputs,
                    "num_images": num_images,
                    "num_videos": num_videos,
                    "num_elements": num_elements,
                    "num_frames": num_frames,
                })
                
                NODE_CLASS_MAPPINGS[class_name] = new_class
                NODE_DISPLAY_NAME_MAPPINGS[class_name] = f"{name} (deepgen)"

    except Exception as e:
        print(f"DeepGen: Failed to load dynamic models: {e}")

load_models()
