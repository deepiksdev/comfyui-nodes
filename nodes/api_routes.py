import json
import csv
import os
from aiohttp import web
from server import PromptServer
from .deepgen_utils import DeepGenConfig

@PromptServer.instance.routes.get("/deepgen/get_api_key")
async def get_api_key(request):
    config = DeepGenConfig()
    key = config.get_key()
    return web.json_response({"api_key": key or ""})

@PromptServer.instance.routes.post("/deepgen/set_api_key")
async def set_api_key(request):
    try:
        data = await request.json()
        api_key = data.get("api_key", "").strip()
        config = DeepGenConfig()
        config.set_key(api_key)
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
                if len(row) < 6:
                    continue
                try:
                    num_images = int(row[5])
                except ValueError:
                    num_images = 1
                    
                models_info.append({
                    "value": row[0],
                    "name": row[1],
                    "aspect_ratios": [x.strip() for x in row[2].split(",")] if row[2].strip() else [],
                    "resolutions": [x.strip() for x in row[3].split(",")] if row[3].strip() else [],
                    "pixel_sizes": [x.strip() for x in row[4].split(",")] if row[4].strip() else [],
                    "nb_of_images": num_images
                })
        return web.json_response({"models": models_info})
    except Exception as e:
        print(f"DeepGen: Failed to fetch models for frontend: {e}")
        return web.json_response({"status": "error", "message": str(e)}, status=500)

