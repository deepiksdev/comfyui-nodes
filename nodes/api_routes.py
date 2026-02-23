import json
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
