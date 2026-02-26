from .deepgen_utils import DeepGenApiHandler, DeepGenConfig, ImageUtils, ResultProcessor

# Initialize DeepGenConfig
deepgen_config = DeepGenConfig()


class UpscalerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "input_video_url": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"default": "masterpiece, best quality, highres", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "(worst quality, low quality, normal quality:2)", "multiline": True}),
                "seed": ("INT", {"default": -1}),
                "creativity": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "resemblance": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "num_inference_steps": ("INT", {"default": 18, "min": 1, "max": 100}),
                "num_inference_steps": ("INT", {"default": 18, "min": 1, "max": 100}),
                "alias_id": ("STRING", {"default": "deepgen/clarity-upscaler"}),
                "endpoint": ("STRING", {"default": "https://api.deepgen.app"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "FLOAT",)
    RETURN_NAMES = ("image", "video_url", "agent_alias", "total_credits_used",)
    FUNCTION = "upscale"
    CATEGORY = "DeepGen/Upscaling"

    def upscale(
        self,
        upscale_factor,
        image=None,
        video=None,
        input_video_url="",
        prompt="masterpiece, best quality, highres",
        negative_prompt="",
        seed=-1,
        creativity=0.35,
        resemblance=0.6,
        guidance_scale=4.0,
        num_inference_steps=18,
        alias_id="deepgen/clarity-upscaler",
        endpoint="https://api.deepgen.app",
    ):
        try:
            arguments = {
                "upscale_factor": upscale_factor,
            }

            if seed != -1:
                arguments["seed"] = seed

            # Determine if it's a video or image based on provided inputs or alias_id
            is_video = False
            if "video" in (alias_id or "").lower() or video is not None or input_video_url:
                is_video = True

            if is_video:
                video_url = input_video_url
                if video is not None:
                    video_url = ImageUtils.upload_file(video.get_stream_source())
                
                if not video_url:
                    return (None, "Error: No video provided for upscaling", "", 0.0)
                
                arguments["video_url"] = video_url
                # Add other video specific arguments if any (like from topas or bria)
                # For now keeping it generic
                
                result = DeepGenApiHandler.submit_and_get_result(alias_id, arguments, api_url=endpoint)
                video_url_res = result.get("video", {}).get("url") or result.get("url")
                agent_alias_out = result.get("agent_alias", "")
                credits_out = float(result.get("total_credits_used", 0.0))
                return (None, video_url_res, agent_alias_out, credits_out)
            else:
                if image is None:
                    return (ResultProcessor.create_blank_image()[0], "Error: No image provided", "", 0.0)
                
                image_url = ImageUtils.upload_image(image)
                if not image_url:
                    return (ResultProcessor.create_blank_image()[0], "Error: Failed to upload image", "", 0.0)
                
                arguments.update({
                    "image_url": image_url,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "creativity": creativity,
                    "resemblance": resemblance,
                    "guidance_scale": guidance_scale,
                    "num_inference_steps": num_inference_steps,
                })
                
                result = DeepGenApiHandler.submit_and_get_result(alias_id, arguments, api_url=endpoint)
                processed = ResultProcessor.process_image_result(result)
                agent_alias_out = result.get("agent_alias", "")
                credits_out = float(result.get("total_credits_used", 0.0))
                return (processed[0], "", agent_alias_out, credits_out)

        except ValueError as ve:
            raise ve
        except Exception as e:
            return (ResultProcessor.create_blank_image()[0], str(e), "", 0.0)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "Upscaler_deepgen": UpscalerNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "Upscaler_deepgen": "Upscaler (deepgen)",
}
