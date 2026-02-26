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
                video_urls = [ResultProcessor.process_video_result(r)[0] for r in results]
                # Returning first one as primary, though we could return a list if we changed RETURN_TYPES
                
                def get_dict(r):
                    obj = r[0] if isinstance(r, list) and len(r) > 0 else r
                    return obj if isinstance(obj, dict) else getattr(obj, '__dict__', {}) or {}
                
                agent_alias = get_dict(results[0]).get("agent_alias", "") if results else ""
                prefixed_model = f"{output_prefix}_{agent_alias}" if output_prefix else agent_alias
                credits_out = sum(float(get_dict(r).get("total_credits_used", 0.0)) for r in results)
                return (video_urls[0], prefixed_model, credits_out)
            else:
                result = DeepGenApiHandler.submit_and_get_result(alias_id, arguments, api_url=endpoint)
                
                res_obj = result[0] if isinstance(result, list) and len(result) > 0 else result
                if not isinstance(res_obj, dict):
                    res_obj = getattr(res_obj, '__dict__', {}) or {}
                
                video_url = ResultProcessor.process_video_result(result)[0]
                agent_alias = res_obj.get("agent_alias", "")
                prefixed_model = f"{output_prefix}_{agent_alias}" if output_prefix else agent_alias
                credits_out = float(res_obj.get("total_credits_used", 0.0))
                return (video_url, prefixed_model, credits_out)

        except ValueError as ve:
            raise ve
        except Exception as e:
            error_msg = DeepGenApiHandler.handle_video_generation_error(alias_id, str(e))[0]
            return (error_msg, "", 0.0)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "Video_deepgen": VideoNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "Video_deepgen": "Video (deepgen)",
}
