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
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "start_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "duration": (["5", "10"], {"default": "5"}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "9:21"], {"default": "16:9"}),
                "loop": ("BOOLEAN", {"default": False}),
                "variations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "alias_id": ("STRING", {"default": "deepgen/minimax/video-01-live/image-to-video"}),
                "endpoint": ("STRING", {"default": "https://api.deepgen.app"}),
                "output_prefix": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT",)
    RETURN_NAMES = ("video_url", "output_prefix_and_model", "total_credits_used",)
    FUNCTION = "generate_video"
    CATEGORY = "DeepGen/VideoGeneration"

    def generate_video(
        self,
        prompt,
        image=None,
        start_image=None,
        end_image=None,
        duration="5",
        aspect_ratio="16:9",
        loop=False,
        variations=1,
        alias_id="deepgen/minimax/video-01-live/image-to-video",
        endpoint="https://api.deepgen.app",
        output_prefix="",
    ):
        try:
            arguments = {
                "prompt": prompt,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "loop": loop,
                "queue": True,
            }

            # Handle images
            if image is not None:
                img_url = ImageUtils.upload_image(image)
                if img_url:
                    # Some models use 'image_url', some 'start_image_url'
                    arguments["image_url"] = img_url
                    arguments["start_image_url"] = img_url
            
            if start_image is not None:
                img_url = ImageUtils.upload_image(start_image)
                if img_url:
                    arguments["start_image_url"] = img_url
                    if "image_url" not in arguments:
                        arguments["image_url"] = img_url

            if end_image is not None:
                img_url = ImageUtils.upload_image(end_image)
                if img_url:
                    arguments["end_image_url"] = img_url

            if variations > 1:
                results = DeepGenApiHandler.submit_multiple_and_get_results(alias_id, arguments, variations, api_url=endpoint)
                results = self._poll_results(results, endpoint)
                
                video_urls = [ResultProcessor.process_video_result(r)[0] for r in results]
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
                    
                return (video_urls[0], prefixed_model, credits_out)
            else:
                result = DeepGenApiHandler.submit_and_get_result(alias_id, arguments, api_url=endpoint)
                result = self._poll_results([result], endpoint)[0]
                
                res_obj = result[0] if isinstance(result, list) and len(result) > 0 else result
                if not isinstance(res_obj, dict):
                    res_obj = getattr(res_obj, '__dict__', {}) or {}
                
                video_url = ResultProcessor.process_video_result(result)[0]
                agent_alias = res_obj.get("agent_alias", "")
                prefixed_model = f"{output_prefix}_{agent_alias}" if output_prefix else agent_alias
                
                cred = res_obj.get("total_credits_used")
                if cred is None:
                    cred = res_obj.get("output", {}).get("total_credits_used", res_obj.get("aiCredits", 0.0))
                credits_out = float(cred)
                
                return (video_url, prefixed_model, credits_out)

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
                pending[i] = (res_obj["queue_id"], agent_alias)
            else:
                final_results[i] = res

        while pending:
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
