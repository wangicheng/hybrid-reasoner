from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any
import sys
import os

# Add the project root to sys.path to resolve 'src' imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.engine import HybridEngine
from src.core.model_catalog import DEFAULT_MODEL_ID, normalize_model_id
import uvicorn


# Input Model
class SearchRequest(BaseModel):
    query: str
    model_id: str = DEFAULT_MODEL_ID

# Engine Instance (Lazy Load)
engine = None

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("Initializing Hybrid Engine... (This may take a few seconds)")
    try:
        # 初始化引擎 (預設使用實驗最佳配置)
        engine = HybridEngine()
        print("Hybrid Engine initialized successfully!")
        print("Search mode: fixed retrieval path")
    except RuntimeError as e:
        if "already accessed by another instance" in str(e):
            print(f"\n{'='*60}")
            print(f"ERROR: Qdrant storage is locked by another process!")
            print(f"Please kill other Python processes first:")
            print(f"  taskkill /F /IM python.exe")
            print(f"Then restart this application.")
            print(f"{'='*60}\n")
        raise
    except Exception as e:
        print(f"FATAL: Failed to initialize Engine: {e}")
        raise
    yield

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow requests from Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development, we can allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import StreamingResponse
import json
import asyncio

@app.get("/api/search/stream")
async def search_stream(query: str, model_id: str = DEFAULT_MODEL_ID):
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine is still initializing")

    async def event_generator():
        queue = asyncio.Queue()

        async def on_progress(step: str, data: Any):
            await queue.put({"event": step, "data": data})

        # Run search in a way that we can capture its progress
        search_task = asyncio.create_task(engine.search(
            query,
            limit=10,
            model_id=normalize_model_id(model_id),
            progress_callback=on_progress
        ))

        while not search_task.done() or not queue.empty():
            try:
                # Use a short timeout to keep checking if the task is done
                item = await asyncio.wait_for(queue.get(), timeout=0.2)
                yield f"event: {item['event']}\ndata: {json.dumps(item['data'])}\n\n"
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"
                break

        if search_task.done():
            try:
                result = await search_task
                yield f"event: complete\ndata: {json.dumps(result)}\n\n"
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/search")
async def search(request: SearchRequest):
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine is still initializing")
    
    try:
        results = await engine.search(
            request.query,
            limit=10,
            model_id=normalize_model_id(request.model_id)
        )
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
# We'll create the web folder next
if not os.path.exists("src/web"):
    os.makedirs("src/web")

app.mount("/", StaticFiles(directory="src/web", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
