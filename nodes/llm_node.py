from .deepgen_utils import DeepGenApiHandler, DeepGenConfig, ImageUtils, ResultProcessor
import os
import csv

# Initialize DeepGenConfig
deepgen_config = DeepGenConfig()


class LLMNode:
    # Base INPUT_TYPES left blank as this will be a dynamically generated subclass.
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {}}

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Bypass standard ComfyUI validation for dynamic combo boxes"""
        return True

    RETURN_TYPES = ("STRING", "STRING", "FLOAT",)
    RETURN_NAMES = ("output", "output_prefix_and_model", "total_credits_used",)
    FUNCTION = "generate_text"
    CATEGORY = "DeepGen/LLM"

    def generate_text(self, **kwargs):
        try:
            alias_id = getattr(self, "alias_id", "gemini-3-flash")
            
            def unwrap(v):
                return v[0] if isinstance(v, list) and len(v) > 0 else v

            prompt = unwrap(kwargs.get("prompt", ""))
            seed_value = unwrap(kwargs.get("seed_value", -1))
            endpoint = unwrap(kwargs.get("endpoint", "https://api.deepgen.app"))
            output_prefix = unwrap(kwargs.get("output_prefix", ""))
            unique_id = unwrap(kwargs.get("unique_id"))
            extra_pnginfo = unwrap(kwargs.get("extra_pnginfo"))
            
            image_urls = []
            attachments_files = []
            original_names_map = {}
            
            for k, v in kwargs.items():
                if v is None:
                    continue
                if k in ["prompt", "seed_value", "endpoint", "output_prefix", "unique_id", "extra_pnginfo"]:
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
            
            arguments = {
                "prompt": prompt,
                "stream": False,
            }
            if seed_value != -1:
                arguments["seed"] = seed_value

            if image_urls:
                arguments["attachments_urls"] = image_urls
            if attachments_files:
                arguments["attachments_files"] = attachments_files

            result = DeepGenApiHandler.submit_and_get_result(alias_id, arguments, api_url=endpoint)
            
            res_obj = result[0] if isinstance(result, list) and len(result) > 0 else result
            text_result = ResultProcessor.process_text_result(result)
            
            def _get_attr(obj, key, default=None):
                if isinstance(obj, dict): return obj.get(key, default)
                return getattr(obj, key, default)
                
            agent_alias = _get_attr(res_obj, "agent_alias", "")
            prefixed_model = f"{output_prefix}_{agent_alias}" if output_prefix else agent_alias
            
            cred = _get_attr(res_obj, "total_credits_used")
            if cred is None:
                out_obj = _get_attr(res_obj, "output", {})
                cred = _get_attr(out_obj, "total_credits_used", _get_attr(res_obj, "aiCredits", 0.0))
            credits_out = float(cred or 0.0)
            return (text_result[0], prefixed_model, credits_out)
        except ValueError as ve:
            raise ve
        except Exception as e:
            error_result = DeepGenApiHandler.handle_text_generation_error(alias_id, str(e))
            return (error_result[0], "", 0.0)



