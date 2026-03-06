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
        cls.models_map = {}
        cls.supported_inputs_map = {}
        
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models.csv")
        try:
            import csv
            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 11 and row[10].strip() == "T2V":
                        cls.models_list.append(row[1])
                        cls.models_map[row[1]] = row[0]
                        cls.supported_inputs_map[row[0]] = [x.strip() for x in row[2].split(",")] if row[2].strip() else []
        except Exception as e:
            print(f"DeepGen: Failed to load models.csv for video_node: {e}")
            cls.models_list = ["Kling 2.5 Turbo Pro"]
            cls.models_map = {"Kling 2.5 Turbo Pro": "kling2.5-turbo-pro"}
            cls.supported_inputs_map = {"kling2.5-turbo-pro": []}

        return {
            "required": {
                "model": ([""] + cls.models_list, {"default": ""}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "seed_value": ("INT", {"default": -1}),
                "variations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "endpoint": ("STRING", {"default": "https://api.deepgen.app"}),
                "output_prefix": ("STRING", {"default": ""}),
                
                # Dynamic combo fields
                "duration": (["5", "10"], {"default": "5"}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "9:21"], {"default": "16:9"}),
                "resolution": ([""], {"default": ""}),
                
                # Additional T2V fields
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.5}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "queue": ("BOOLEAN", {"default": False}),
                "loop": ("BOOLEAN", {"default": False}),
                "generate_audio": ("BOOLEAN", {"default": False}),
                "shot_type": ("STRING", {"default": ""}),
                "auto_fix": ("BOOLEAN", {"default": False}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "safety_tolerance": ("STRING", {"default": "Auto"}),
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
        variations=1,
        endpoint="https://api.deepgen.app",
        output_prefix="",
        duration="5",
        aspect_ratio="16:9",
        resolution="",
        temperature=0.7,
        cfg_scale=7.0,
        negative_prompt="",
        queue=False,
        loop=False,
        generate_audio=False,
        shot_type="",
        auto_fix=False,
        enable_safety_checker=True,
        safety_tolerance="Auto",
        **kwargs
    ):
        try:
            alias_id = getattr(self, "models_map", {}).get(model, "kling2.5-turbo-pro")
            supported_inputs = getattr(self, "supported_inputs_map", {}).get(alias_id, [])
            
            arguments = {
                "prompt": prompt,
            }
            
            # Submitting optional inputs conditionally based on what's supported
            if "duration" in supported_inputs and duration:
                arguments["duration"] = duration
            if "aspect_ratio" in supported_inputs and aspect_ratio not in ("", "Auto"):
                arguments["aspect_ratio"] = aspect_ratio
            if "resolution" in supported_inputs and resolution not in ("", "Auto"):
                arguments["resolution"] = resolution
                
            if "negative_prompt" in supported_inputs and negative_prompt:
                arguments["negative_prompt"] = negative_prompt
            if "temperature" in supported_inputs:
                arguments["temperature"] = temperature
            if "cfg_scale" in supported_inputs:
                arguments["cfg_scale"] = cfg_scale
            if "queue" in supported_inputs:
                arguments["queue"] = queue
            if "loop" in supported_inputs:
                arguments["loop"] = loop
            if "generate_audio" in supported_inputs:
                arguments["generate_audio"] = generate_audio
            if "shot_type" in supported_inputs and shot_type:
                arguments["shot_type"] = shot_type
            if "auto_fix" in supported_inputs:
                arguments["auto_fix"] = auto_fix
            if "enable_safety_checker" in supported_inputs:
                arguments["enable_safety_checker"] = enable_safety_checker
            if "safety_tolerance" in supported_inputs and safety_tolerance not in ("", "Auto"):
                arguments["safety_tolerance"] = safety_tolerance

            if seed_value != -1:
                arguments["seed"] = seed_value

            attachments_files = []
            
            # Exclude standard optional fields from kwargs
            standard_kwargs = [
                "prompt", "seed_value", "variations", "endpoint", "output_prefix", "duration", 
                "aspect_ratio", "resolution", "temperature", "cfg_scale", "negative_prompt", 
                "queue", "loop", "generate_audio", "shot_type", "auto_fix", 
                "enable_safety_checker", "safety_tolerance"
            ]
            
            for k, v in kwargs.items():
                if v is None:
                    continue
                if k in ["model"] + standard_kwargs:
                    continue

                # The new separate image sockets for elements (element_i_frontal, etc.) 
                # will hit the standard attachment code block below and automatically inherit 
                # their required prefix prefix via the kwargs key `k`.

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
                
                def _get_attr(obj, key, default=None):
                    if isinstance(obj, dict): return obj.get(key, default)
                    return getattr(obj, key, default)

                def _get_res_obj(r):
                    return r[0] if isinstance(r, list) and len(r) > 0 else r

                res0 = _get_res_obj(results[0]) if results else {}
                agent_alias = _get_attr(res0, "agent_alias", "")
                prefixed_model = f"{output_prefix}_{agent_alias}" if output_prefix else agent_alias
                
                credits_out = 0.0
                for r in results:
                    obj = _get_res_obj(r)
                    cred = _get_attr(obj, "total_credits_used")
                    if cred is None:
                        out_obj = _get_attr(obj, "output", {})
                        cred = _get_attr(out_obj, "total_credits_used", _get_attr(obj, "aiCredits", 0.0))
                    credits_out += float(cred or 0.0)
                    
                return (video_paths[0], prefixed_model, credits_out)
            else:
                result = DeepGenApiHandler.submit_and_get_result(alias_id, arguments, api_url=endpoint)
                result = self._poll_results([result], endpoint)[0]
                
                res_dic = result[0] if isinstance(result, list) and len(result) > 0 else result
                video_path = ResultProcessor.process_video_result(result)[0]
                
                out_dic = res_dic.get("output", {})
                agent_alias = out_dic.get("agent_alias", "")
                print("AGENT ALIAS",agent_alias,out_dic)
                prefixed_model = f"{output_prefix}_{agent_alias}" if output_prefix else agent_alias
                
                cred = out_dic.get("total_credits_used")
                print("WILL RETURN",prefixed_model, cred)
                return (video_path, prefixed_model, cred)

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
                agent_alias = res_obj.get("agent_alias", "_")
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
                        print("POLL DATA",poll_data)
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
