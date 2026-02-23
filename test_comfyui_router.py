from aiohttp import web

def register_routes(server):
    @server.routes.post('/deepgen/set_api_key')
    async def set_api_key(request):
        data = await request.json()
        #rint("API key set:", data)
        return web.json_response({"status": "ok"})
