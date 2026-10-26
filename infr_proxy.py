# infr_proxy.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

app = FastAPI()

INFR_BASE = "https://infr.ad"
INFR_KEY = "sk-infr-t3zv913v.gqorrbqippby8lhc6fefofir5zyyph4q"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.body()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {INFR_KEY}",
    }

    # Forward accept để hỗ trợ streaming
    accept = request.headers.get("accept")
    if accept:
        headers["Accept"] = accept

    async def stream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{INFR_BASE}/v1/chat/completions",
                content=body,
                headers=headers,
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/v1/models")
async def list_models():
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{INFR_BASE}/v1/models",
            headers={"Authorization": f"Bearer {INFR_KEY}"},
        )
        return JSONResponse(content=r.json(), status_code=r.status_code)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5002)
