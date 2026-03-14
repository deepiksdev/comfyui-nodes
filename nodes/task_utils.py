import os
import csv
import json
import torch
import copy
from .deepgen_utils import DeepGenApiHandler as ApiHandler, ImageUtils, ResultProcessor

def load_models_for_task(task_name):
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models.csv")
    models = []
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                tasks = [x.strip() for x in row[2].split(",")]
                if task_name in tasks:
                    models.append(row[0])
    except Exception as e:
        print(f"DeepGen: Failed to load models for task {task_name}: {e}")
    if not models:
        models = ["No models found"]
    return models

def parse_config_json(config_str):
    if not config_str or not config_str.strip():
        return {}
    try:
        return json.loads(config_str)
    except json.JSONDecodeError as e:
        print(f"DeepGen: Failed to parse config_json: {e}")
        return {}

def process_kwargs_for_images(kwargs, unique_id, extra_pnginfo):
    attachments_files = []
    original_names_map = {}

    def get_orig_name(idx, original_names):
        if idx < len(original_names) and original_names[idx]:
            org = str(original_names[idx])
            if org.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.mp4')):
                org = org.rsplit('.', 1)[0]
            return f"_{org}"
        return ""

    for k, v in kwargs.items():
        if v is None:
            continue
        if k in ["model", "prompt", "seed_value", "nb_results", "output_prefix", "config_json", "minimum_resolution", "aspect_ratio", "output_format", "endpoint", "unique_id", "extra_pnginfo"]:
            continue

        prefix_base = k
        if k not in original_names_map and unique_id and extra_pnginfo:
            original_names_map[k] = ImageUtils.resolve_filenames(unique_id, extra_pnginfo, k)
        original_names = original_names_map.get(k, [])

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

        for i, item in enumerate(flattened_items):
            if hasattr(item, "shape"):
                attach = ImageUtils.get_attachment_file(item, filename=f"{prefix_base}_{i+1}{get_orig_name(i, original_names)}.png")
                if attach:
                    attachments_files.append(attach)

    return attachments_files

def parse_ratio(r_str):
    if r_str.lower() == 'auto':
        return 1.0
    try:
        w, h = map(float, r_str.split(':'))
        return w / h
    except:
        return 1.0

def parse_res_k(res_str):
    res_str = str(res_str).lower().replace('k', '')
    try:
        if res_str == '500':
            return 500
        return float(res_str) * 1024
    except:
        return 1024

def get_best_pixel_size(pixel_sizes, target_size, target_ratio):
    parsed = []
    for ps in pixel_sizes:
        try:
            w, h = map(int, ps.split('x'))
            ratio = w / h
            size = w * h
            parsed.append({'ps': ps, 'ratio': ratio, 'size': size})
        except:
            pass
    if not parsed:
        return None
        
    valid_ratios = [p for p in parsed if p['ratio'] >= target_ratio]
    if valid_ratios:
        best_ratio = min(valid_ratios, key=lambda x: x['ratio'])['ratio']
    else:
        best_ratio = max(parsed, key=lambda x: x['ratio'])['ratio']
        
    best_ratio_items = [p for p in parsed if abs(p['ratio'] - best_ratio) < 0.01]
    
    target_area = target_size * target_size
    valid_sizes = [p for p in best_ratio_items if p['size'] >= target_area]
    if valid_sizes:
        best_item = min(valid_sizes, key=lambda x: x['size'])
    else:
        best_item = max(best_ratio_items, key=lambda x: x['size'])
        
    return best_item['ps']

def get_best_resolution_and_ratio(resolutions, aspect_ratios, target_size, target_ratio):
    parsed_ar = []
    for ar in aspect_ratios:
        parsed_ar.append({'ar': ar, 'ratio': parse_ratio(ar)})
    
    if parsed_ar:
        valid_ar = [p for p in parsed_ar if p['ratio'] >= target_ratio]
        if valid_ar:
            best_ar = min(valid_ar, key=lambda x: x['ratio'])['ar']
        else:
            best_ar = max(parsed_ar, key=lambda x: x['ratio'])['ar']
    else:
        best_ar = None

    parsed_res = []
    for r in resolutions:
        parsed_res.append({'res': r, 'size': parse_res_k(r)})
        
    if parsed_res:
        valid_res = [p for p in parsed_res if p['size'] >= target_size]
        if valid_res:
            best_res = min(valid_res, key=lambda x: x['size'])['res']
        else:
            best_res = max(parsed_res, key=lambda x: x['size'])['res']
    else:
        best_res = None
        
    return best_res, best_ar

class BaseTaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {}}

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def _poll_video_results(self, results):
        import time
        import requests
        from .deepgen_utils import DeepGenConfig
        
        config = DeepGenConfig()
        key = config.get_key()
        if not key:
            raise ValueError("DeepGen API Key not found.")
        user_id = key.split("_")[0] if "_" in key else ""
        
        base_url = config.get_base_url()
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

    def run_generation(self, task_type, **kwargs):
        def unwrap(v):
            return v[0] if isinstance(v, list) and len(v) > 0 else v

        model = unwrap(kwargs.get("model", ""))
        prompt = unwrap(kwargs.get("prompt", ""))
        seed_value = unwrap(kwargs.get("seed_value", 1000))
        nb_results = unwrap(kwargs.get("nb_results", 1))
        output_prefix = unwrap(kwargs.get("output_prefix", ""))
        config_json_str = unwrap(kwargs.get("config_json", ""))
        minimum_resolution = unwrap(kwargs.get("minimum_resolution", ""))
        aspect_ratio = unwrap(kwargs.get("aspect_ratio", ""))
        output_format = unwrap(kwargs.get("output_format", ""))
        unique_id = unwrap(kwargs.get("unique_id"))
        extra_pnginfo = unwrap(kwargs.get("extra_pnginfo"))

        arguments = {
            "task": task_type,
            "prompt": prompt,
            "seed": seed_value,
        }
        
        if task_type in ["T2I", "I2I", "I2I3", "I2I10", "T2V", "I2V", "I2V2", "I2VR", "V2V", "V2VR"]:
            arguments["num_images"] = nb_results # used for both image and video
            
        if task_type in ["T2V", "I2V", "I2V2", "I2VR", "V2V", "V2VR"]:
            arguments["queue"] = True
            
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models.csv")
        resolutions_supported, aspect_ratios_supported, pixel_sizes_supported = [], [], []
        try:
            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0] == model:
                        if len(row) > 3 and row[3].strip():
                            aspect_ratios_supported = [x.strip() for x in row[3].split(",")]
                        if len(row) > 4 and row[4].strip():
                            resolutions_supported = [x.strip() for x in row[4].split(",")]
                        if len(row) > 5 and row[5].strip():
                            pixel_sizes_supported = [x.strip() for x in row[5].split(",")]
                        break
        except Exception:
            pass

        if minimum_resolution and aspect_ratio and (resolutions_supported or pixel_sizes_supported):
            target_size = parse_res_k(minimum_resolution)
            target_ratio = parse_ratio(aspect_ratio)
            
            if pixel_sizes_supported:
                best_ps = get_best_pixel_size(pixel_sizes_supported, target_size, target_ratio)
                if best_ps:
                    arguments["pixel_size"] = best_ps
                    # When pixel size is found, we don't submit aspect_ratio and resolution
            elif resolutions_supported:
                best_res, best_ar = get_best_resolution_and_ratio(
                    resolutions_supported, aspect_ratios_supported, target_size, target_ratio
                )
                if best_res:
                    arguments["resolution"] = best_res
                if best_ar:
                    arguments["aspect_ratio"] = best_ar
        else:
            if minimum_resolution:
                arguments["resolution"] = minimum_resolution
            if aspect_ratio:
                arguments["aspect_ratio"] = aspect_ratio
                
        if output_format:
            arguments["output_format"] = output_format

        attachments_files = process_kwargs_for_images(kwargs, unique_id, extra_pnginfo)
        if attachments_files:
            arguments["attachments_files"] = attachments_files

        extra_args = parse_config_json(config_json_str)
        arguments.update(extra_args)

        try:
            if task_type in ["T2V", "I2V", "I2V2", "I2VR", "V2V", "V2VR"]:
                if nb_results > 1:
                    results = ApiHandler.submit_multiple_and_get_results(model, arguments, nb_results)
                    results = self._poll_video_results(results)
                    
                    outputs = []
                    credits_out = 0.0
                    for r in results:
                        outputs.append(ResultProcessor.process_video_result(r)[0])
                        obj = r[0] if isinstance(r, list) and len(r) > 0 else r
                        cred = obj.get("total_credits_used") if isinstance(obj, dict) else 0.0
                        if cred is None:
                            cred = obj.get("output", {}).get("total_credits_used", obj.get("aiCredits", 0.0))
                        credits_out += float(cred or 0.0)
                    prefixed_model = f"{output_prefix}_{model}" if output_prefix else model
                    return (outputs[0], prefixed_model, credits_out) # returning first video
                else:
                    result = ApiHandler.submit_and_get_result(model, arguments)
                    result = self._poll_video_results([result])[0]
                    
                    res_obj = result[0] if isinstance(result, list) and len(result) > 0 else result
                    video_path = ResultProcessor.process_video_result(result)[0]
                    
                    def _get_attr(obj, key, default=None):
                        if isinstance(obj, dict): return obj.get(key, default)
                        return getattr(obj, key, default)
                        
                    agent_alias = _get_attr(res_obj, "agent_alias", model)
                    if isinstance(res_obj, dict) and "output" in res_obj:
                        agent_alias = res_obj["output"].get("agent_alias", agent_alias)
                        
                    prefixed_model = f"{output_prefix}_{agent_alias}" if output_prefix else agent_alias
                    
                    cred = _get_attr(res_obj, "total_credits_used")
                    if cred is None:
                        out_obj = _get_attr(res_obj, "output", {})
                        cred = _get_attr(out_obj, "total_credits_used", _get_attr(res_obj, "aiCredits", 0.0))
                    credits_out = float(cred or 0.0)
                    return (video_path, prefixed_model, credits_out)

            elif task_type in ["T2T", "I2T"]:
                arguments["stream"] = False
                result = ApiHandler.submit_and_get_result(model, arguments)
                res_obj = result[0] if isinstance(result, list) and len(result) > 0 else result
                text_result = ResultProcessor.process_text_result(result)[0]
                
                def _get_attr(obj, key, default=None):
                    if isinstance(obj, dict): return obj.get(key, default)
                    return getattr(obj, key, default)
                    
                agent_alias = _get_attr(res_obj, "agent_alias", model)
                if isinstance(res_obj, dict) and "output" in res_obj:
                    agent_alias = res_obj["output"].get("agent_alias", agent_alias)
                    
                cred = _get_attr(res_obj, "total_credits_used")
                if cred is None:
                    out_obj = _get_attr(res_obj, "output", {})
                    cred = _get_attr(out_obj, "total_credits_used", _get_attr(res_obj, "aiCredits", 0.0))
                credits_out = float(cred or 0.0)
                # No output prefix for T nodes
                return (text_result, agent_alias, credits_out)

            else:
                # Images
                result = ApiHandler.submit_and_get_result(model, arguments)
                res_obj = result[0] if isinstance(result, list) and len(result) > 0 else result
                img_tensor = ResultProcessor.process_image_result(result)[0]
                
                def _get_attr(obj, key, default=None):
                    if isinstance(obj, dict): return obj.get(key, default)
                    return getattr(obj, key, default)
                    
                agent_alias = _get_attr(res_obj, "agent_alias", model)
                if isinstance(res_obj, dict) and "output" in res_obj:
                    agent_alias = res_obj["output"].get("agent_alias", agent_alias)
                    
                prefixed_model = f"{output_prefix}_{agent_alias}" if output_prefix else agent_alias
                
                cred = _get_attr(res_obj, "total_credits_used")
                if cred is None:
                    out_obj = _get_attr(res_obj, "output", {})
                    cred = _get_attr(out_obj, "total_credits_used", _get_attr(res_obj, "aiCredits", 0.0))
                credits_out = float(cred or 0.0)
                return (img_tensor, prefixed_model, credits_out)

        except Exception as e:
            print(f"DeepGen task generation error: {e}")
            raise e
