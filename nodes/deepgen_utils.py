import configparser
import io
import os
import json
import tempfile
import asyncio
import concurrent.futures
import time

import numpy as np
import requests
import torch
from PIL import Image


class DeepGenConfig:
    """Singleton class to handle DeepGen configuration and client setup."""

    _instance = None
    _key = None
    _base_url = "https://api.deepgen.app"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeepGenConfig, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize configuration and API key."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        old_env_path = os.path.join(parent_dir, ".env")
        old_config_path = os.path.join(parent_dir, "config.ini")

        # Determine user directory for the config
        try:
            import folder_paths
            user_dir = os.path.join(folder_paths.base_path, "user", "deepgen")
        except ImportError:
            # Fallback if imported outside ComfyUI environment
            comfy_path = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
            user_dir = os.path.join(comfy_path, "user", "deepgen")

        if not os.path.exists(user_dir):
            try:
                os.makedirs(user_dir, exist_ok=True)
            except Exception as e:
                #rint(f"Warning: Could not create user directory at {user_dir}: {e}")

        user_config_path = os.path.join(user_dir, "config.json")
        default_config = {
            "DEEPGEN_API_KEY": "<your_deepgen_api_key_here>",
            "DEEPGEN_API_URL": "https://api.deepgen.app"
        }

        # 1. Provide a template if not exists
        if not os.path.exists(user_config_path) and os.path.exists(user_dir):
            # Try to migrate from old .env or config.ini if they exist
            migrated = False
            if os.path.exists(old_env_path):
                try:
                    with open(old_env_path, "r") as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("DEEPGEN_API_KEY="):
                                key = line.split("=", 1)[1].strip("\"'")
                                default_config["DEEPGEN_API_KEY"] = key
                                migrated = True
                            elif line.startswith("DEEPGEN_API_URL="):
                                url = line.split("=", 1)[1].strip("\"'")
                                default_config["DEEPGEN_API_URL"] = url
                                migrated = True
                except Exception as e:
                    #rint(f"Warning: failed reading old .env: {e}")

            if not migrated and os.path.exists(old_config_path):
                try:
                    config = configparser.ConfigParser()
                    config.read(old_config_path)
                    if "API" in config and "DEEPGEN_API_KEY" in config["API"]:
                        key = config["API"]["DEEPGEN_API_KEY"]
                        if key and key != "<your_deepgen_api_key_here>":
                            default_config["DEEPGEN_API_KEY"] = key
                            migrated = True
                except Exception as e:
                    #rint(f"Warning: failed reading old config.ini: {e}")

            try:
                with open(user_config_path, "w") as f:
                    json.dump(default_config, f, indent=4)
                #rint(f"\n[!] DeepGen Nodes: Created a config file at {user_config_path}")
                if migrated:
                    #rint(f"[!] Migrated old API keys into the new config file.")
                else:
                    #rint(f"[!] Please add your DEEPGEN_API_KEY to this file and restart ComfyUI.\n")
            except Exception as e:
                #rint(f"Warning: could not write config file at {user_config_path}: {e}")

        # 2. Load the user config
        user_config = {}
        self._config_error = None
        if os.path.exists(user_config_path):
            try:
                import re
                with open(user_config_path, "r") as f:
                    content = f.read()
                    # Strip out trailing commas to prevent annoying JSON syntax errors
                    content = re.sub(r',\s*([\]}])', r'\1', content)
                    user_config = json.loads(content)
            except Exception as e:
                self._config_error = f"Malformed JSON in {user_config_path}: {e}"
                #rint(f"Error reading config from {user_config_path}: {e}")

        try:
            # 3. Apply configurations (Env overrides User Config)
            self._key = os.environ.get("DEEPGEN_API_KEY")
            if not self._key:
                key = user_config.get("DEEPGEN_API_KEY")
                if key and key != "<your_deepgen_api_key_here>":
                    self._key = key
                    os.environ["DEEPGEN_API_KEY"] = self._key
                    #rint(f"DEEPGEN_API_KEY loaded from {user_config_path}")

            if not self._key:
                #rint(f"Error: DEEPGEN_API_KEY not found in {user_config_path} or environment variables")
                #rint(f"Please configure it at: {user_config_path}")
            elif self._key == "<your_deepgen_api_key_here>":
                #rint(f"WARNING: You are using the default DEEPGEN_API_KEY placeholder in {user_config_path}!")
                
            # Allow overriding base URL from env
            env_url = os.environ.get("DEEPGEN_API_URL")
            if env_url:
                self._base_url = env_url
            else:
                config_url = user_config.get("DEEPGEN_API_URL")
                if config_url and config_url != "https://api.deepgen.app":
                    self._base_url = config_url
                    os.environ["DEEPGEN_API_URL"] = self._base_url
                    #rint(f"DEEPGEN_API_URL loaded from {user_config_path}: {self._base_url}")
                
        except Exception as e:
            #rint(f"Error initializing DeepGenConfig: {str(e)}")

    def get_key(self):
        """Get the DeepGen API key."""
        return self._key
        
    def get_base_url(self):
        """Get the DeepGen API base URL."""
        return self._base_url

    @staticmethod
    def check_key(key):
        """Raise an informative error if the API key is not configured."""
        # First check if there was a JSON parse error to warn the user about syntax
        config_inst = DeepGenConfig()
        if hasattr(config_inst, '_config_error') and config_inst._config_error:
            raise ValueError(f"Syntax Error in config.json: {config_inst._config_error}")

        if not key or key == "<your_deepgen_api_key_here>":
            try:
                import folder_paths
                user_dir = os.path.join(folder_paths.base_path, "user", "deepgen")
            except ImportError:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                comfy_path = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
                user_dir = os.path.join(comfy_path, "user", "deepgen")
            user_config_path = os.path.join(user_dir, "config.json")
            raise ValueError(f"DEEPGEN_API_KEY is missing. Please click the ComfyUI Settings gear icon and enter your Deepgen API Key to use the nodes.")

    def set_key(self, api_key):
        """Set the DeepGen API key and save it to config.json."""
        self._key = api_key
        os.environ["DEEPGEN_API_KEY"] = api_key
        
        try:
            import folder_paths
            user_dir = os.path.join(folder_paths.base_path, "user", "deepgen")
        except ImportError:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            comfy_path = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
            user_dir = os.path.join(comfy_path, "user", "deepgen")

        user_config_path = os.path.join(user_dir, "config.json")
        
        user_config = {}
        if os.path.exists(user_config_path):
            try:
                with open(user_config_path, "r") as f:
                    import re
                    content = f.read()
                    content = re.sub(r',\s*([\]}])', r'\1', content)
                    user_config = json.loads(content)
            except Exception:
                pass
                
        user_config["DEEPGEN_API_KEY"] = api_key
        
        os.makedirs(user_dir, exist_ok=True)
        try:
            with open(user_config_path, "w") as f:
                json.dump(user_config, f, indent=4)
        except Exception as e:
            #rint(f"Warning: could not write config file at {user_config_path}: {e}")



class ImageUtils:
    """Utility functions for image processing."""

    @staticmethod
    def tensor_to_pil(image):
        """Convert image tensor to PIL Image."""
        try:
            # Convert the image tensor to a numpy array
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
            else:
                image_np = np.array(image)

            # Ensure the image is in the correct format (H, W, C)
            if image_np.ndim == 4:
                image_np = image_np.squeeze(0)  # Remove batch dimension if present
            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)  # Convert grayscale to RGB
            elif image_np.shape[0] == 3:
                image_np = np.transpose(
                    image_np, (1, 2, 0)
                )  # Change from (C, H, W) to (H, W, C)

            # Normalize the image data to 0-255 range
            if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                image_np = (image_np * 255).astype(np.uint8)

            # Convert to PIL Image
            return Image.fromarray(image_np)
        except Exception as e:
            #rint(f"Error converting tensor to PIL: {str(e)}")
            return None

    @staticmethod
    def upload_image(image):
        """Upload image tensor to DeepGen and return URL."""
        try:
            pil_image = ImageUtils.tensor_to_pil(image)
            if not pil_image:
                return None

            # Save the image to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                pil_image.save(temp_file, format="PNG")
                temp_file_path = temp_file.name

            return ImageUtils.upload_file(temp_file_path)
        except Exception as e:
            #rint(f"Error uploading image: {str(e)}")
            return None
        finally:
            # Clean up the temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                
    @staticmethod
    def get_attachment_file(image, filename="image.png"):
        """Convert image tensor to AttachmentFile dict with base64 encoded bytes."""
        import base64
        import io
        try:
            pil_image = ImageUtils.tensor_to_pil(image)
            if not pil_image:
                return None
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            # Send as base64 string for JSON compatibility; server-side Pydantic bytes field will decode it
            base64_str = base64.b64encode(image_bytes).decode('utf-8')
            
            return {
                "attachment_bytes": base64_str,
                "attachment_mime_type": "image/png",
                "attachment_file_name": filename
            }
        except Exception as e:
            #rint(f"Error creating attachment file: {str(e)}")
            return None

    @staticmethod
    def upload_file(file_path):
        """Upload a file to DeepGen and return URL."""
        try:
            config = DeepGenConfig()
            key = config.get_key()
            DeepGenConfig.check_key(key)
                
            url = f"{config.get_base_url()}/upload" # Assumption: /upload endpoint
            
            with open(file_path, 'rb') as f:
                files = {'file': f}
                headers = {'Authorization': f'Bearer {key}'}
                response = requests.post(url, headers=headers, files=files)
                
            if response.status_code == 200:
                data = response.json()
                if "url" in data:
                    return data["url"]
                # Adjust based on actual response structure if known
                #rint(f"Upload response: {data}")
                return data.get("file_url") or data.get("url")
            else:
                #rint(f"Upload failed with status {response.status_code}: {response.text}")
                return None
        except Exception as e:
            #rint(f"Error uploading file: {str(e)}")
            return None
        
    @staticmethod
    def mask_to_image(mask):
        """Convert mask tensor to image tensor."""
        result = (
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
            .movedim(1, -1)
            .expand(-1, -1, -1, 3)
        )
        return result
    
    @staticmethod
    def prepare_images(images):
        """Preprocess images for use with DeepGen."""
        image_urls = []
        if images is not None:

                if isinstance(images, torch.Tensor):
                    if images.ndim == 4 and images.shape[0] > 1:
                        for i in range(images.shape[0]):
                            single_img = images[i:i+1]
                            img_url = ImageUtils.upload_image(single_img)
                            if img_url:
                                image_urls.append(img_url)
                    else:
                        img_url = ImageUtils.upload_image(images)
                        if img_url:
                            image_urls.append(img_url)

                elif isinstance(images, (list, tuple)):
                    for img in images:
                        img_url = ImageUtils.upload_image(img)
                        if img_url:
                            image_urls.append(img_url)
        return image_urls


class ResultProcessor:
    """Utility functions for processing API results."""

    @staticmethod
    def _extract_image_urls(result):
        """Helper to extract image URLs from various possible response structures."""
        urls = []
        
        def search(obj):
            if isinstance(obj, dict):
                # Check for direct URL in this dict (DeepGen/FAL common pattern)
                url = obj.get("url")
                if isinstance(url, str) and (url.startswith("http") or url.startswith("data:image")):
                    # Validate it's likely an image if mimeType is present
                    mime = obj.get("mimeType", "")
                    if not mime or "image" in mime:
                        urls.append(url)
                        return # Found a leaf, stop searching this branch

                # Search common container keys
                for key in ["images", "attachments", "results", "image", "output", "data"]:
                    if key in obj:
                        search(obj[key])
                
                # If we still haven't found much, maybe it's nested in a message-like structure
                for key, value in obj.items():
                    if key not in ["images", "attachments", "results", "image", "output", "data"]:
                        if isinstance(value, (dict, list)):
                            search(value)

            elif isinstance(obj, list):
                for item in obj:
                    search(item)
            elif isinstance(obj, str):
                # Direct string URL
                if obj.startswith("http") and any(ext in obj.lower() for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]):
                    urls.append(obj)

        search(result)
        # Deduplicate while preserving order
        seen = set()
        return [x for x in urls if not (x in seen or seen.add(x))]

    @staticmethod
    def process_image_result(result):
        """Process image generation result and return tensor."""
        try:
            image_urls = ResultProcessor._extract_image_urls(result)
            
            images = []
            for img_url in image_urls:
                try:
                    img_response = requests.get(img_url, timeout=30)
                    if img_response.status_code == 200:
                        img = Image.open(io.BytesIO(img_response.content))
                        # Handle RGBA or other formats
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img_array = np.array(img).astype(np.float32) / 255.0
                        images.append(img_array)
                except Exception as e:
                    #rint(f"Failed to download/process image from {img_url}: {str(e)}")

            if not images:
                #rint(f"No images found in result: {result}")
                return ResultProcessor.create_blank_image()

            # Stack the images along a new first dimension
            stacked_images = np.stack(images, axis=0)

            # Convert to PyTorch tensor
            img_tensor = torch.from_numpy(stacked_images)

            return (img_tensor,)
        except Exception as e:
            #rint(f"Error processing image result: {str(e)}")
            return ResultProcessor.create_blank_image()

    @staticmethod
    def process_single_image_result(result):
        """Process single image result and return tensor."""
        # Just use the general processor, it returns a 4D tensor (N, H, W, C)
        # which is what ComfyUI expects for a single image anyway (N=1)
        return ResultProcessor.process_image_result(result)

    @staticmethod
    def process_text_result(result):
        """Process text result and return (output, reasoning)."""
        try:
            # Handle list of results
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
            if not isinstance(result, dict):
                return (str(result), "")

            output = result.get("output")
            # DeepGen sometimes uses 'text' or 'response'
            if not output:
                output = result.get("text") or result.get("response")
            
            # If still nothing, maybe it's in a list of choices (OpenAI style)
            if not output and "choices" in result:
                choices = result["choices"]
                if isinstance(choices, list) and len(choices) > 0:
                    choice = choices[0]
                    if isinstance(choice, dict):
                        message = choice.get("message", {})
                        if isinstance(message, dict):
                            output = message.get("content")
                        if not output:
                            output = choice.get("text")
            
            # Use general searching as fallback
            if not output:
                # Look for longest string in dict that isn't a known metadata key
                candidate = ""
                for k, v in result.items():
                    if k not in ["conversation_id", "group_id", "input_message_id", "output_message_id", "agent_alias"]:
                        if isinstance(v, str) and len(v) > len(candidate):
                            candidate = v
                output = candidate

            reasoning = result.get("reasoning", "")
            
            return (output or "", reasoning or "")
        except Exception as e:
            #rint(f"Error processing text result: {str(e)}")
            return (f"Error: {str(e)}", "")

    @staticmethod
    def process_file_result(result):
        """Process result that should contain a file URL (e.g. LoRA)."""
        try:
            # Handle list of results
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
            if not isinstance(result, dict):
                return (str(result),)

            # Look for explicit URL
            urls = []
            def search(obj):
                if isinstance(obj, dict):
                    url = obj.get("url")
                    if isinstance(url, str) and url.startswith("http"):
                        urls.append(url)
                    for v in obj.values():
                         if isinstance(v, (dict, list)):
                             search(v)
                elif isinstance(obj, list):
                    for item in obj:
                        search(item)
            
            search(result)
            if urls:
                return (urls[0],)

            return (f"Error: No file URL found in result: {result}",)
        except Exception as e:
            #rint(f"Error processing file result: {str(e)}")
            return (f"Error: {str(e)}",)

    @staticmethod
    def _extract_video_urls(result):
        """Helper to extract video URLs from various possible response structures."""
        urls = []
        
        def search(obj):
            if isinstance(obj, dict):
                # Check for direct URL in this dict
                url = obj.get("url")
                if isinstance(url, str) and url.startswith("http"):
                    # Validate it's likely a video if mimeType or extension suggests it
                    mime = obj.get("mimeType", "").lower()
                    if "video" in mime or not mime:
                        if any(ext in url.lower() for ext in [".mp4", ".mov", ".webm", ".m4v", ".mkv"]):
                            urls.append(url)
                            return # Found a leaf

                # Search common container keys
                for key in ["videos", "attachments", "results", "video", "output", "data"]:
                    if key in obj:
                        search(obj[key])
                
                # Recursive search for others
                for key, value in obj.items():
                    if key not in ["videos", "attachments", "results", "video", "output", "data"]:
                        if isinstance(value, (dict, list)):
                            search(value)

            elif isinstance(obj, list):
                for item in obj:
                    search(item)
            elif isinstance(obj, str):
                # Direct string URL
                if obj.startswith("http") and any(ext in obj.lower() for ext in [".mp4", ".mov", ".webm", ".m4v", ".mkv"]):
                    urls.append(obj)

        search(result)
        # Deduplicate while preserving order
        seen = set()
        return [x for x in urls if not (x in seen or seen.add(x))]

    @staticmethod
    def process_video_result(result):
        """Process video generation result and return tuple of URLs."""
        try:
            video_urls = ResultProcessor._extract_video_urls(result)
            if not video_urls:
                #rint(f"No videos found in result: {result}")
                return ("Error: No video found in result",)
            
            # Return as tuple of strings (first one if only one expected by most nodes)
            return (video_urls[0],)
        except Exception as e:
            #rint(f"Error processing video result: {str(e)}")
            return (f"Error: Processing video result failed: {str(e)}",)

    @staticmethod
    def create_blank_image():
        """Create a blank black image tensor."""
        blank_img = Image.new("RGB", (512, 512), color="black")
        img_array = np.array(blank_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        return (img_tensor,)


class DeepGenApiHandler:
    """Utility functions for API interactions."""

    @staticmethod
    def _map_arguments(arguments):
        """Map arguments to the expected DeepGen API format."""
        mapped = arguments.copy()

        # Mapping rules
        field_mapping = {
            "prompt": "question",
            "system_prompt": "forced_system_prompt",
            "num_inference_steps": "steps",
            # number_of_steps is also used in trainers
            "number_of_steps": "steps",
            "guidance_scale": "cfg_scale",
            "num_images": "num_results",
        }

        for old_key, new_key in field_mapping.items():
            if old_key in mapped:
                val = mapped.pop(old_key)
                # Don't overwrite if new_key already exists unless val is more significant
                if new_key not in mapped or val:
                    mapped[new_key] = val

        # Handle image_size
        if "image_size" in mapped:
            size = mapped.pop("image_size")
            if isinstance(size, dict):
                if "width" in size: mapped["width"] = size.get("width")
                if "height" in size: mapped["height"] = size.get("height")
            elif isinstance(size, str):
                mapped["aspect_ratio"] = size

        # Handle URLs -> attachments_urls
        attachments_urls = mapped.get("attachments_urls", [])
        if not isinstance(attachments_urls, list):
            attachments_urls = [attachments_urls]

        url_fields = [
            "image_url", "video_url", "images_zip_url",
            "images_data_url", "training_data_url"
        ]
        for url_key in url_fields:
             if url_key in mapped:
                 val = mapped.pop(url_key)
                 if isinstance(val, str) and val:
                     attachments_urls.append(val)
                 elif isinstance(val, list):
                     attachments_urls.extend([v for v in val if v])

        if "image_urls" in mapped:
            val = mapped.pop("image_urls")
            if isinstance(val, list):
                attachments_urls.extend([v for v in val if v])
            elif isinstance(val, str) and val:
                attachments_urls.append(val)

        if attachments_urls:
            # Filter out empty strings and ensure uniqueness
            mapped["attachments_urls"] = list(dict.fromkeys([u for u in attachments_urls if u]))

        # Add default type if not present
        if "type" not in mapped:
            mapped["type"] = "Chat"

        return mapped

    @staticmethod
    def submit_and_get_result(endpoint, arguments, api_url=None):
        """Submit job to DeepGen API and get result."""
        try:
            config = DeepGenConfig()
            print("CONFIG:", config)
            key = config.get_key()
            print("KEY:", key)

            DeepGenConfig.check_key(key)
            
            base_url = api_url if api_url else config.get_base_url()
            
            # Map arguments to DeepGen Gateway format
            mapped_arguments = DeepGenApiHandler._map_arguments(arguments)
            
            # Construct URL: base_url + / + endpoint (alias_id) + /api
            # Handle potential double slashes if base_url ends with /
            if base_url.endswith("/"):
                base_url = base_url[:-1]
                
            url = f"{base_url}/{endpoint}/api"
            
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            }
            
            #rint(f"Submitting to {url}")
            #rint(f"Mapped Arguments: {mapped_arguments}")
            response = requests.post(url, json=mapped_arguments, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                #rint(f"DeepGen API Response: {result}")
                return result
            elif response.status_code == 201: # Accepted/Async?
                # Handle polling if needed, but assuming sync for simple endpoints for now
                # Or if it returns a request_id to poll
                result = response.json()
                #rint(f"DeepGen API Async Response: {result}")
                if "request_id" in result:
                    return DeepGenApiHandler._poll_result(result["request_id"], api_url=base_url)
                return result
            else:
                if response.status_code in [401, 403]:
                    raise ValueError(f"DeepGen API Authentication Error ({response.status_code}). Please verify your Deepgen API Key in ComfyUI Settings.")
                raise Exception(f"API Error {response.status_code}: {response.text}")

        except Exception as e:
            #rint(f"Error submitting to {endpoint}: {str(e)}")
            raise e
            
    @staticmethod
    def _poll_result(request_id, api_url=None):
        """Poll for result."""
        config = DeepGenConfig()
        key = config.get_key()
        base_url = api_url if api_url else config.get_base_url()
        
        # Handle potential double slashes if base_url ends with /
        if base_url.endswith("/"):
            base_url = base_url[:-1]

        url = f"{base_url}/requests/{request_id}" # Assumption
        headers = {"Authorization": f"Bearer {key}"}
        
        while True:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                raise Exception(f"Polling failed: {response.text}")
            
            data = response.json()
            #rint(f"DeepGen Poll Response: {data}")
            
            status = data.get("status")
            
            if status == "COMPLETED":
                return data.get("result", data)
            elif status == "FAILED":
                raise Exception(f"Job failed: {data.get('error')}")
            
            time.sleep(1)

    @staticmethod
    def submit_multiple_and_get_results(endpoint, arguments, variations, api_url=None):
        """Submit multiple jobs concurrently to DeepGen API and get results."""
        try:
            # Simple serial implementation for now, or use ThreadPoolExecutor
            results = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for i in range(variations):
                    # Create copy of args
                    args = arguments.copy()
                    if "seed" in args:
                        args["seed"] = args["seed"] + i
                    futures.append(executor.submit(DeepGenApiHandler.submit_and_get_result, endpoint, args, api_url))
                
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
            return results
        except Exception as e:
            #rint(f"Error in submit_multiple_and_get_results: {str(e)}")
            raise e

    @staticmethod
    def handle_video_generation_error(model_name, error):
        """Handle video generation errors consistently."""
        #rint(f"Error generating video with {model_name}: {str(error)}")
        return ("Error: Unable to generate video.",)

    @staticmethod
    def handle_image_generation_error(model_name, error):
        """Handle image generation errors consistently."""
        import traceback
        error_details = traceback.format_exc()
        #rint(f"Error generating image with {model_name}: {str(error)}")
        #rint(f"Traceback: {error_details}")
        return ResultProcessor.create_blank_image()

    @staticmethod
    def handle_text_generation_error(model_name, error):
        """Handle text generation errors consistently."""
        #rint(f"Error generating text with {model_name}: {str(error)}")
        return ("Error: Unable to generate text.",)
