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
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    num_images = int(row.get("nb_of_images", 1))
                except ValueError:
                    num_images = 1
                    
                models_info.append({
                    "name": row["name"],
                    "value": row["value"],
                    "nb_of_images": num_images
                })
        return web.json_response({"models": models_info})
    except Exception as e:
        print(f"DeepGen: Failed to fetch models for frontend: {e}")
        return web.json_response({"status": "error", "message": str(e)}, status=500)

