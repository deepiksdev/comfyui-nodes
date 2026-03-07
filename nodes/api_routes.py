import json
import csv
import os
from aiohttp import web
from server import PromptServer
from .deepgen_utils import DeepGenConfig

@PromptServer.instance.routes.get("/deepgen/get_settings")
async def get_settings(request):
    config = DeepGenConfig()
    key = config.get_key()
    url = config.get_base_url()
    return web.json_response({"api_key": key or "", "api_url": url or "https://api.deepgen.app"})

@PromptServer.instance.routes.post("/deepgen/set_settings")
async def set_settings(request):
    try:
        data = await request.json()
        api_key = data.get("api_key", "").strip()
        api_url = data.get("api_url", "").strip()
        config = DeepGenConfig()
        config.set_key_and_url(api_key, api_url)
        return web.json_response({"status": "success"})
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)}, status=500)

@PromptServer.instance.routes.get("/deepgen/models")
async def get_deepgen_models(request):
    """Returns the parsed configurations from models.csv to the frontend."""
    models_info = []
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models.csv")
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 11:
                    continue
                try:
                    num_images = int(row[6]) if row[6].strip() else 0
                except ValueError:
                    num_images = 1
                try:
                    num_videos = int(row[7]) if row[7].strip() else 0
                except ValueError:
                    num_videos = 0
                try:
                    num_elements = int(row[8]) if row[8].strip() else 0
                except ValueError:
                    num_elements = 0
                try:
                    num_frames = int(row[9]) if row[9].strip() else 0
                except ValueError:
                    num_frames = 0
                    
                models_info.append({
                    "value": row[0],
                    "name": row[1],
                    "optional_inputs": [x.strip() for x in row[2].split(",")] if row[2].strip() else [],
                    "aspect_ratios": [x.strip() for x in row[3].split(",")] if row[3].strip() else [],
                    "resolutions": [x.strip() for x in row[4].split(",")] if row[4].strip() else [],
                    "pixel_sizes": [x.strip() for x in row[5].split(",")] if row[5].strip() else [],
                    "nb_of_images": num_images,
                    "nb_of_videos": num_videos,
                    "nb_of_elements": num_elements,
                    "nb_of_frames": num_frames,
                    "type": row[10].strip()
                })
        return web.json_response({"models": models_info})
    except Exception as e:
        print(f"DeepGen: Failed to fetch models for frontend: {e}")
        return web.json_response({"status": "error", "message": str(e)}, status=500)

