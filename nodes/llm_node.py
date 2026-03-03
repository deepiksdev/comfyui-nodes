from .deepgen_utils import DeepGenApiHandler, DeepGenConfig, ImageUtils, ResultProcessor
import os
import csv

# Initialize DeepGenConfig
deepgen_config = DeepGenConfig()


class LLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Load models from CSV
        cls.models_list = []
        cls.models_map = {} # Map from name to value (alias_id)
        
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models.csv")
        try:
            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 8 and row[7].strip() == "LLM":
                        cls.models_list.append(row[1])
                        cls.models_map[row[1]] = row[0]
        except Exception as e:
            print(f"DeepGen: Failed to load models.csv for llm_node: {e}")
            cls.models_list = ["Gemini 3 Flash"]
            cls.models_map = {"Gemini 3 Flash": "gemini-3-flash"}

        return {
            "required": {
                "model": (cls.models_list, {"default": cls.models_list[0] if cls.models_list else ""}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "max_tokens": ("INT", {"default": 1024, "min": 0, "max": 100000}),
                "endpoint": ("STRING", {"default": "https://api.deepgen.app"}),
                "output_prefix": ("STRING", {"default": ""}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Bypass standard ComfyUI validation for dynamic combo boxes"""
        return True

    RETURN_TYPES = ("STRING", "STRING", "STRING", "FLOAT",)
    RETURN_NAMES = ("output", "reasoning", "output_prefix_and_model", "total_credits_used",)
    FUNCTION = "generate_text"
    CATEGORY = "DeepGen/LLM"

    def generate_text(self, model, prompt, max_tokens=1024, endpoint="https://api.deepgen.app", output_prefix="", **kwargs):
        try:
            alias_id = self.models_map.get(model, "gemini-3-flash")
            
            image_urls = []
            images_to_process = []
            videos_to_process = []
            for k, v in kwargs.items():
                if k.startswith('image_') and v is not None:
                    images_to_process.append(v)
                elif k.startswith('video_') and v is not None:
                    videos_to_process.append(v)
                    
            attachments_files = []
            for img in images_to_process:
                if len(img.shape) == 4:
                    for i in range(img.shape[0]):
                        single_image = img[i:i+1]
                        attach = ImageUtils.get_attachment_file(single_image, filename=f"image_{len(attachments_files)}.png")
                        if attach:
                            attachments_files.append(attach)
                else:
                    attach = ImageUtils.get_attachment_file(img, filename=f"image_{len(attachments_files)}.png")
                    if attach:
                        attachments_files.append(attach)
                        
            for vid in videos_to_process:
                vid_path = None
                if isinstance(vid, str) and os.path.exists(vid):
                    vid_path = vid
                elif hasattr(vid, "filepath") and vid.filepath and os.path.exists(vid.filepath):
                    vid_path = vid.filepath
                
                if vid_path:
                    import base64
                    with open(vid_path, "rb") as vf:
                        b64 = base64.b64encode(vf.read()).decode("utf-8")
                        attachments_files.append({
                            "attachment_bytes": b64,
                            "attachment_mime_type": "video/mp4",
                            "attachment_file_name": f"video_{len(attachments_files)}.mp4"
                        })
            
            arguments = {
                "prompt": prompt,
                "stream": False,
            }
            if image_urls:
                arguments["attachments_urls"] = image_urls
            if attachments_files:
                arguments["attachments_files"] = attachments_files

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
