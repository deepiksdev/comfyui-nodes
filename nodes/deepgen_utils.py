import configparser
import io
import os
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
        config_path = os.path.join(parent_dir, "config.ini")

        config = configparser.ConfigParser()
        config.read(config_path)

        # Try to load from .env file directly if it exists in parent directory
        env_path = os.path.join(parent_dir, ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith("DEEPGEN_API_KEY="):
                        self._key = line.strip().split("=", 1)[1]
                        # Remove quotes if present
                        if (self._key.startswith('"') and self._key.endswith('"')) or \
                           (self._key.startswith("'") and self._key.endswith("'")):
                            self._key = self._key[1:-1]
                        os.environ["DEEPGEN_API_KEY"] = self._key
                        print(f"DEEPGEN_API_KEY loaded from {env_path}")
                        break

        try:
            if not self._key and os.environ.get("DEEPGEN_API_KEY") is not None:
                print("DEEPGEN_API_KEY found in environment variables")
                self._key = os.environ["DEEPGEN_API_KEY"]
            
            if not self._key:
                # Fallback to checking config.ini just in case, though structure might differ
                if "API" in config and "DEEPGEN_API_KEY" in config["API"]:
                    self._key = config["API"]["DEEPGEN_API_KEY"]
                    if self._key != "<your_deepgen_api_key_here>":
                        print("DEEPGEN_API_KEY found in config.ini")
                        os.environ["DEEPGEN_API_KEY"] = self._key
                    else:
                        self._key = None

            if not self._key:
                print("Error: DEEPGEN_API_KEY not found in .env, config.ini or environment variables")
            elif self._key == "<your_deepgen_api_key_here>":
                print("WARNING: You are using the default DeepGen API key placeholder!")
                
            # Allow overriding base URL from env
            if os.environ.get("DEEPGEN_API_URL"):
                self._base_url = os.environ["DEEPGEN_API_URL"]
                
        except Exception as e:
            print(f"Error initializing DeepGenConfig: {str(e)}")

    def get_key(self):
        """Get the DeepGen API key."""
        return self._key
        
    def get_base_url(self):
        """Get the DeepGen API base URL."""
        return self._base_url


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
            print(f"Error converting tensor to PIL: {str(e)}")
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
            print(f"Error uploading image: {str(e)}")
            return None
        finally:
            # Clean up the temporary file
            if "temp_file_path" in locals():
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                
    @staticmethod
    def upload_file(file_path):
        """Upload a file to DeepGen and return URL."""
        try:
            config = DeepGenConfig()
            key = config.get_key()
            if not key:
                print("Cannot upload: API key missing")
                return None
                
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
                print(f"Upload response: {data}")
                return data.get("file_url") or data.get("url")
            else:
                print(f"Upload failed with status {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"Error uploading file: {str(e)}")
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
    def process_image_result(result):
        """Process image generation result and return tensor."""
        try:
            images = []
            # DeepGen response structure might differ. Assuming similar to FAL for now:
            # {"images": [{"url": "..."}]}
            
            image_list = result.get("images", [])
            # Also handle single "image" key
            if "image" in result and not image_list:
                image_list = [result["image"]]
                
            for img_info in image_list:
                img_url = img_info.get("url")
                if not img_url:
                    continue
                    
                img_response = requests.get(img_url)
                img = Image.open(io.BytesIO(img_response.content))
                img_array = np.array(img).astype(np.float32) / 255.0
                images.append(img_array)

            if not images:
                print("No images found in result")
                return ResultProcessor.create_blank_image()

            # Stack the images along a new first dimension
            stacked_images = np.stack(images, axis=0)

            # Convert to PyTorch tensor
            img_tensor = torch.from_numpy(stacked_images)

            return (img_tensor,)
        except Exception as e:
            print(f"Error processing image result: {str(e)}")
            return ResultProcessor.create_blank_image()

    @staticmethod
    def process_single_image_result(result):
        """Process single image result and return tensor."""
        try:
            img_info = result.get("image")
            if not img_info:
                # try finding in images list
                if "images" in result and len(result["images"]) > 0:
                    img_info = result["images"][0]
            
            if not img_info:
                print("No image found in result")
                return ResultProcessor.create_blank_image()

            img_url = img_info.get("url")
            img_response = requests.get(img_url)
            img = Image.open(io.BytesIO(img_response.content))
            img_array = np.array(img).astype(np.float32) / 255.0

            # Stack the images along a new first dimension
            stacked_images = np.stack([img_array], axis=0)

            # Convert to PyTorch tensor
            img_tensor = torch.from_numpy(stacked_images)
            return (img_tensor,)
        except Exception as e:
            print(f"Error processing single image result: {str(e)}")
            return ResultProcessor.create_blank_image()

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
    def submit_and_get_result(endpoint, arguments):
        """Submit job to DeepGen API and get result."""
        try:
            config = DeepGenConfig()
            key = config.get_key()
            base_url = config.get_base_url()
            
            # Construct URL: base_url + / + endpoint (alias_id) + /api
            url = f"{base_url}/{endpoint}/api"
            
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            }
            
            print(f"Submitting to {url}")
            response = requests.post(url, json=arguments, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                print(f"DeepGen API Response: {result}")
                return result
            elif response.status_code == 201: # Accepted/Async?
                # Handle polling if needed, but assuming sync for simple endpoints for now
                # Or if it returns a request_id to poll
                result = response.json()
                print(f"DeepGen API Async Response: {result}")
                if "request_id" in result:
                    return DeepGenApiHandler._poll_result(result["request_id"])
                return result
            else:
                raise Exception(f"API Error {response.status_code}: {response.text}")

        except Exception as e:
            print(f"Error submitting to {endpoint}: {str(e)}")
            raise e
            
    @staticmethod
    def _poll_result(request_id):
        """Poll for result."""
        config = DeepGenConfig()
        key = config.get_key()
        base_url = config.get_base_url()
        url = f"{base_url}/requests/{request_id}" # Assumption
        headers = {"Authorization": f"Bearer {key}"}
        
        while True:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                raise Exception(f"Polling failed: {response.text}")
            
            data = response.json()
            print(f"DeepGen Poll Response: {data}")
            
            status = data.get("status")
            
            if status == "COMPLETED":
                return data.get("result", data)
            elif status == "FAILED":
                raise Exception(f"Job failed: {data.get('error')}")
            
            time.sleep(1)

    @staticmethod
    def submit_multiple_and_get_results(endpoint, arguments, variations):
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
                    futures.append(executor.submit(DeepGenApiHandler.submit_and_get_result, endpoint, args))
                
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
            return results
        except Exception as e:
            print(f"Error in submit_multiple_and_get_results: {str(e)}")
            raise e

    @staticmethod
    def handle_video_generation_error(model_name, error):
        """Handle video generation errors consistently."""
        print(f"Error generating video with {model_name}: {str(error)}")
        return ("Error: Unable to generate video.",)

    @staticmethod
    def handle_image_generation_error(model_name, error):
        """Handle image generation errors consistently."""
        import traceback
        error_details = traceback.format_exc()
        print(f"Error generating image with {model_name}: {str(error)}")
        print(f"Traceback: {error_details}")
        return ResultProcessor.create_blank_image()

    @staticmethod
    def handle_text_generation_error(model_name, error):
        """Handle text generation errors consistently."""
        print(f"Error generating text with {model_name}: {str(error)}")
        return ("Error: Unable to generate text.",)
