from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
import os

# Add the project root to sys.path to resolve 'src' imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.engine import HybridEngine
import uvicorn

app = FastAPI()

# Input Model
class SearchRequest(BaseModel):
    query: str
    model_id: str = "gemini-2.0-flash"

# Engine Instance (Lazy Load)
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    print("Initializing Hybrid Engine... (This may take a few seconds)")
    try:
        engine = HybridEngine()
        print("Hybrid Engine initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize Engine: {e}")

@app.post("/api/search")
async def search(request: SearchRequest):
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine is still initializing")
    
    # Set the model ID environment variable dynamically based on user selection
    os.environ["LLM_MODEL_ID"] = request.model_id
        
    try:
        results = engine.search(request.query, limit=10)
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
