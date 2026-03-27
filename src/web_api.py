from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
import os

# Add the project root to sys.path to resolve 'src' imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.engine import HybridEngine
import uvicorn


# Input Model
class SearchRequest(BaseModel):
    query: str
    model_id: str = "gemma-3-27b-it"

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
        print("Search mode: Embedded tags + prompt (semantic:attribute = 0.3:0.7)")
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

app = FastAPI(lifespan=lifespan)

@app.post("/api/search")
async def search(request: SearchRequest):
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine is still initializing")
    
    try:
        results = await engine.search(
            request.query,
            limit=10,
            model_id=request.model_id
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
