from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
import os

# Add the project root to sys.path to resolve 'src' imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
import sys
sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI()

# Input Model
class SearchRequest(BaseModel):
    query: str
    model_id: str = "gemma-3-27b-it"
    rerank_strategy: str = "score_only"
    rerank_alpha: float = 0.3

# Engine Instance (Lazy Load)
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    
    # 印出明確的 localhost 網址方便使用者點擊
    print("\n" + "="*50)
    print("🚀 伺服器已啟動！")
    print("👉 請點擊前往：http://localhost:8000")
    print("="*50 + "\n")

    # 將笨重的 AI 套件載入移到這裡，讓 uvicorn 可以馬上印出啟動文字，避免畫面卡住
    print("Loading heavy machine learning libraries... (This may take a few seconds)")
    from src.core.engine import HybridEngine
    
    print("Initializing Hybrid Engine... (This may take a few seconds)")
    try:
        engine = HybridEngine()
        print("Hybrid Engine initialized successfully!")
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

@app.post("/api/search")
async def search(request: SearchRequest):
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine is still initializing")
    
    try:
        results = await engine.search(
            request.query,
            limit=10,
            model_id=request.model_id,
            rerank_strategy=request.rerank_strategy,
            rerank_alpha=request.rerank_alpha,
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
