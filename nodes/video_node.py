import asyncio
import os
import tempfile
import concurrent.futures
import cv2
import numpy as np
import requests
import torch
from .deepgen_utils import DeepGenApiHandler, DeepGenConfig, ImageUtils, ResultProcessor

# Initialize DeepGenConfig
deepgen_config = DeepGenConfig()

class VideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Load models from CSV
        cls.models_list = []
        cls.models_map = {} # Map from name to value (alias_id)
        
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models.csv")
        try:
            import csv
            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 10 and row[9].strip() == "T2V":
                        cls.models_list.append(row[1])
                        cls.models_map[row[1]] = row[0]
        except Exception as e:
            print(f"DeepGen: Failed to load models.csv for video_node: {e}")
            cls.models_list = ["Kling 2.5 Turbo Pro"]
            cls.models_map = {"Kling 2.5 Turbo Pro": "kling2.5-turbo-pro"}

        return {
            "required": {
                "model": (cls.models_list, {"default": cls.models_list[0] if cls.models_list else ""}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "seed_value": ("INT", {"default": -1}),
                "duration": (["5", "10"], {"default": "5"}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "9:21"], {"default": "16:9"}),
                "loop": ("BOOLEAN", {"default": False}),
                "variations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "endpoint": ("STRING", {"default": "https://api.deepgen.app"}),
                "output_prefix": ("STRING", {"default": ""}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Bypass standard ComfyUI validation for dynamic combo boxes"""
        return True

    RETURN_TYPES = ("VIDEO", "STRING", "FLOAT",)
    RETURN_NAMES = ("VIDEO", "output_prefix_and_model", "total_credits_used",)
    FUNCTION = "generate_video"
    CATEGORY = "DeepGen/VideoGeneration"

    def generate_video(
        self,
        model,
        prompt,
        seed_value=-1,
        duration="5",
        aspect_ratio="16:9",
        loop=False,
        variations=1,
        endpoint="https://api.deepgen.app",
        output_prefix="",
        **kwargs
    ):
        try:
            alias_id = self.models_map.get(model, "kling2.5-turbo-pro")
            arguments = {
                "prompt": prompt,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "loop": loop,
                "queue": True,
            }
            if seed_value != -1:
                arguments["seed"] = seed_value

            attachments_files = []
            for k, v in kwargs.items():
                if v is None:
                    continue
                if k in ["model", "prompt", "seed_value", "duration", "aspect_ratio", "loop", "variations", "endpoint", "output_prefix"]:
                    continue

                vid_path = None
                if isinstance(v, str):
                    try:
                        if os.path.exists(v):
                            vid_path = v
                    except Exception:
                        pass
                elif hasattr(v, "filepath") and v.filepath:
                    try:
                        if os.path.exists(v.filepath):
                            vid_path = v.filepath
                    except Exception:
                        pass
                
                if vid_path:
                    import base64
                    import mimetypes
                    import os as os_mod
                    mime_type, _ = mimetypes.guess_type(vid_path)
                    mime_type = mime_type or "application/octet-stream"
                    original_name = os_mod.basename(vid_path)
                    new_filename = f"{k}__{original_name}"
                    with open(vid_path, "rb") as vf:
                        b64 = base64.b64encode(vf.read()).decode("utf-8")
                        attachments_files.append({
                            "attachment_bytes": b64,
                            "attachment_mime_type": mime_type,
                            "attachment_file_name": new_filename
                        })
                    continue
                
                if hasattr(v, "shape"):
                    img = v
                    if len(img.shape) == 4:
                        for i in range(img.shape[0]):
                            single_image = img[i:i+1]
                            attach = ImageUtils.get_attachment_file(single_image, filename=f"{k}__{i}.png")
                            if attach:
                                attachments_files.append(attach)
                    else:
                        attach = ImageUtils.get_attachment_file(img, filename=f"{k}__image.png")
                        if attach:
                            attachments_files.append(attach)
            
            if attachments_files:
                arguments["attachments_files"] = attachments_files

            if variations > 1:
                results = DeepGenApiHandler.submit_multiple_and_get_results(alias_id, arguments, variations, api_url=endpoint)
                results = self._poll_results(results, endpoint)
                
                video_paths = [ResultProcessor.process_video_result(r)[0] for r in results]
                # Returning first one as primary, though we could return a list if we changed RETURN_TYPES
                
                def get_dict(r):
                    obj = r[0] if isinstance(r, list) and len(r) > 0 else r
                    return obj if isinstance(obj, dict) else getattr(obj, '__dict__', {}) or {}
                
                agent_alias = get_dict(results[0]).get("agent_alias", "") if results else ""
                prefixed_model = f"{output_prefix}_{agent_alias}" if output_prefix else agent_alias
                
                credits_out = 0.0
                for r in results:
                    obj = get_dict(r)
                    cred = obj.get("total_credits_used")
                    if cred is None:
                        cred = obj.get("output", {}).get("total_credits_used", obj.get("aiCredits", 0.0))
                    credits_out += float(cred)
                    
                return (video_paths[0], prefixed_model, credits_out)
            else:
                result = DeepGenApiHandler.submit_and_get_result(alias_id, arguments, api_url=endpoint)
                result = self._poll_results([result], endpoint)[0]
                
                res_obj = result[0] if isinstance(result, list) and len(result) > 0 else result
                if not isinstance(res_obj, dict):
                    res_obj = getattr(res_obj, '__dict__', {}) or {}
                
                video_path = ResultProcessor.process_video_result(result)[0]
                agent_alias = res_obj.get("agent_alias", "")
                prefixed_model = f"{output_prefix}_{agent_alias}" if output_prefix else agent_alias
                
                cred = res_obj.get("total_credits_used")
                if cred is None:
                    cred = res_obj.get("output", {}).get("total_credits_used", res_obj.get("aiCredits", 0.0))
                credits_out = float(cred)
                
                return (video_path, prefixed_model, credits_out)

        except ValueError as ve:
            raise ve
        except Exception as e:
            error_msg = DeepGenApiHandler.handle_video_generation_error(alias_id, str(e))[0]
            return (error_msg, "", 0.0)

    def _poll_results(self, results, endpoint):
        import time
        from .deepgen_utils import DeepGenConfig
        
        config = DeepGenConfig()
        key = config.get_key()
        if not key:
            raise ValueError("DeepGen API Key not found.")
        user_id = key.split("_")[0] if "_" in key else ""
        
        base_url = endpoint
        if base_url.endswith("/"):
            base_url = base_url[:-1]
            
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }

        final_results = [None] * len(results)
        pending = {}
        
        for i, res in enumerate(results):
            res_list = res if isinstance(res, list) else [res]
            res_obj = res_list[0] if len(res_list) > 0 else {}
            if not isinstance(res_obj, dict):
                res_obj = getattr(res_obj, '__dict__', {}) or {}
                
            if res_obj.get("status") == "queued" and "queue_id" in res_obj:
                agent_alias = res_obj.get("agent_alias", "kling-1")
                q_id = res_obj["queue_id"]
                pending[i] = (q_id, agent_alias)
                print(f"DeepGen Video: Queued generation with queue_id: {q_id} (model: {agent_alias})")
            else:
                final_results[i] = res

        while pending:
            print(f"DeepGen Video: Polling {len(pending)} pending generation(s)...")
            time.sleep(15)
            completed_indices = []
            for idx, (queue_id, agent_alias) in pending.items():
                poll_url = f"{base_url}/users/{user_id}/agents/{agent_alias}/turns/{queue_id}"
                try:
                    poll_response = requests.get(poll_url, headers=headers)
                    if poll_response.status_code == 200:
                        poll_data = poll_response.json()
                        if isinstance(poll_data, dict) and "output" in poll_data:
                            final_results[idx] = poll_data
                            completed_indices.append(idx)
                            print(f"DeepGen Video: Generation completed for queue_id: {queue_id}")
                        elif isinstance(poll_data, dict) and poll_data.get("status") in ["failed", "error"]:
                            raise ValueError(f"Video generation failed: {poll_data}")
                    else:
                        raise ValueError(f"Polling failed with status {poll_response.status_code}: {poll_response.text}")
                except Exception as e:
                    raise ValueError(f"Polling error for queue_id {queue_id}: {str(e)}")
            
            for idx in completed_indices:
                del pending[idx]
                
        return final_results


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "Video_deepgen": VideoNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "Video_deepgen": "Video (deepgen)",
}
