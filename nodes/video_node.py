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
    # Base INPUT_TYPES left blank as this will be a dynamically generated subclass.
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {}}

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Bypass standard ComfyUI validation for dynamic combo boxes"""
        return True

    RETURN_TYPES = ("VIDEO", "STRING", "FLOAT",)
    RETURN_NAMES = ("VIDEO", "output_prefix_and_model", "total_credits_used",)
    FUNCTION = "generate_video"
    CATEGORY = "DeepGen/VideoGeneration"

    def generate_video(self, **kwargs):
        try:
            alias_id = getattr(self, "alias_id", "kling2.5-turbo-pro")
            supported_inputs = getattr(self, "supported_inputs", [])
            
            def unwrap(v):
                return v[0] if isinstance(v, list) and len(v) > 0 else v

            prompt = unwrap(kwargs.get("prompt", ""))
            seed_value = unwrap(kwargs.get("seed_value", -1))
            variations = unwrap(kwargs.get("variations", 1))
            endpoint = unwrap(kwargs.get("endpoint", "https://api.deepgen.app"))
            output_prefix = unwrap(kwargs.get("output_prefix", ""))
            duration = unwrap(kwargs.get("duration", "5"))
            aspect_ratio = unwrap(kwargs.get("aspect_ratio", "16:9"))
            resolution = unwrap(kwargs.get("resolution", ""))
            temperature = unwrap(kwargs.get("temperature", 0.7))
            cfg_scale = unwrap(kwargs.get("cfg_scale", 7.0))
            negative_prompt = unwrap(kwargs.get("negative_prompt", ""))
            queue = unwrap(kwargs.get("queue", False))
            loop = unwrap(kwargs.get("loop", False))
            generate_audio = unwrap(kwargs.get("generate_audio", False))
            shot_type = unwrap(kwargs.get("shot_type", ""))
            auto_fix = unwrap(kwargs.get("auto_fix", False))
            enable_safety_checker = unwrap(kwargs.get("enable_safety_checker", True))
            safety_tolerance = unwrap(kwargs.get("safety_tolerance", "Auto"))

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
                "enable_safety_checker", "safety_tolerance", "extra_pnginfo", "unique_id"
            ]
            
            unique_id = unwrap(kwargs.get("unique_id"))
            extra_pnginfo = unwrap(kwargs.get("extra_pnginfo"))
            original_names_map = {}
            
            for k, v in kwargs.items():
                if v is None:
                    continue
                if k in ["model"] + standard_kwargs:
                    continue

                limit = 1
                prefix_base = k
                if k in ("image", "images"):
                    limit = 9999
                    prefix_base = "image"
                elif k in ("video", "videos"):
                    limit = 9999
                    prefix_base = "video"
                elif k in ("frame", "frames"):
                    limit = 9999
                    prefix_base = "frame"
                elif k in ("element", "elements"):
                    limit = 9999
                    prefix_base = "element"
                elif k in ("mask", "masks"):
                    limit = 10
                    prefix_base = "mask"

                if k not in original_names_map and unique_id and extra_pnginfo:
                    original_names_map[k] = ImageUtils.resolve_filenames(unique_id, extra_pnginfo, k)
                original_names = original_names_map.get(k, [])
                
                def get_orig_name(idx):
                    if idx < len(original_names) and original_names[idx]:
                        org = str(original_names[idx])
                        if org.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.mp4')):
                            org = org.rsplit('.', 1)[0]
                        return f"_{org}"
                    return ""

                v_list = v if isinstance(v, list) else [v]
                flattened_items = []
                for item in v_list:
                    if hasattr(item, "shape") and len(item.shape) == 4:
                        for i in range(item.shape[0]):
                            flattened_items.append(item[i:i+1])
                    elif isinstance(item, list):
                        flattened_items.extend(item)
                    else:
                        flattened_items.append(item)

                if prefix_base == "element":
                    for i, elem_dict in enumerate(flattened_items[:limit], start=1):
                        if not isinstance(elem_dict, dict):
                            continue
                        
                        for elem_key, elem_val in elem_dict.items():
                            if elem_val is None:
                                continue
                                
                            if elem_key == "frontal_image":
                                if hasattr(elem_val, "shape"):
                                    img = elem_val
                                    if len(img.shape) == 4:
                                        attach = ImageUtils.get_attachment_file(img[0:1], filename=f"{prefix_base}_{i}_frontal.png")
                                    else:
                                        attach = ImageUtils.get_attachment_file(img, filename=f"{prefix_base}_{i}_frontal.png")
                                    if attach:
                                        attachments_files.append(attach)
                            elif elem_key == "references":
                                if hasattr(elem_val, "shape"):
                                    img = elem_val
                                    if len(img.shape) == 4:
                                        for r in range(min(img.shape[0], 3)):
                                            attach = ImageUtils.get_attachment_file(img[r:r+1], filename=f"{prefix_base}_{i}_ref_{r+1}.png")
                                            if attach:
                                                attachments_files.append(attach)
                                    else:
                                        attach = ImageUtils.get_attachment_file(img, filename=f"{prefix_base}_{i}_ref_1.png")
                                        if attach:
                                            attachments_files.append(attach)
                    continue

                for i, item in enumerate(flattened_items[:limit]):
                    if hasattr(item, "shape"):
                        attach = ImageUtils.get_attachment_file(item, filename=f"{prefix_base}_{i+1}{get_orig_name(i)}.png")
                        if attach:
                            attachments_files.append(attach)
                    else:
                        vid_path = None
                        if isinstance(item, str):
                            try:
                                if os.path.exists(item): vid_path = item
                            except: pass
                        elif hasattr(item, "filepath") and item.filepath:
                            try:
                                if os.path.exists(item.filepath): vid_path = item.filepath
                            except: pass
                        
                        if vid_path:
                            import base64
                            import mimetypes
                            import os as os_mod
                            mime_type, _ = mimetypes.guess_type(vid_path)
                            mime_type = mime_type or "application/octet-stream"
                            original_name = os_mod.basename(vid_path)
                            orig_n = get_orig_name(i)
                            
                            if orig_n:
                                ext = os_mod.path.splitext(original_name)[1]
                                new_filename = f"{prefix_base}_{i+1}{orig_n}{ext}"
                            else:
                                new_filename = f"{prefix_base}_{i+1}__{original_name}"
                                
                            with open(vid_path, "rb") as vf:
                                b64 = base64.b64encode(vf.read()).decode("utf-8")
                                attachments_files.append({
                                    "attachment_bytes": b64,
                                    "attachment_mime_type": mime_type,
                                    "attachment_file_name": new_filename
                                })
            
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



