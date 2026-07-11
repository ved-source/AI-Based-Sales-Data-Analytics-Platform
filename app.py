# app.py
import os
import json
import uuid
import uvicorn
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Dict

# Import components
from database_manager import DatabaseManager
from pipeline_engine import PipelineEngine
from sde_components import LRUCache, TokenBucketRateLimiter

# ── CONFIG ─────────────────────────────────────────────────────────────────────
PORT = int(os.getenv("PORT", 8000))
DATA_DIR = "uploaded_data"
os.makedirs(DATA_DIR, exist_ok=True)

# ── INSTANTIATIONS ──────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db = DatabaseManager.get_instance("sales_analytics.db")
rate_limiter = TokenBucketRateLimiter(capacity=10.0, fill_rate=1.0/5.0) # 10 tokens max, refills 1 token every 5s

# ── REST API ROUTES ─────────────────────────────────────────────────────────────
@app.get("/")
async def get_index():
    return FileResponse("index.html")

@app.get("/download/train.csv")
async def download_train_csv():
    csv_path = "train.csv"
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Sample dataset train.csv not found.")
    return FileResponse(csv_path, media_type="text/csv", filename="train.csv")

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    # Rate Limiting
    if not rate_limiter.allow_request():
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment.")
        
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    
    run_id = str(uuid.uuid4())
    csv_path = os.path.join(DATA_DIR, f"{run_id}.csv")
    
    with open(csv_path, "wb") as f:
        f.write(await file.read())
        
    db.create_run(run_id, file.filename)
    return {"status": "ok", "run_id": run_id, "file_name": file.filename}

@app.get("/api/runs")
async def get_runs():
    return {"status": "ok", "runs": db.get_all_runs()}

@app.get("/api/run/{run_id}")
async def get_run(run_id: str):
    results = db.get_run_results(run_id)
    if not results:
        raise HTTPException(status_code=404, detail="Run not found.")
    return {"status": "ok", "data": results}


# ── WEBSOCKET PIPELINE STREAMING ────────────────────────────────────────────────
@app.websocket("/ws/pipeline/{run_id}")
async def ws_pipeline(websocket: WebSocket, run_id: str):
    await websocket.accept()
    csv_path = os.path.join(DATA_DIR, f"{run_id}.csv")
    
    if not os.path.exists(csv_path):
        await websocket.send_json({"type": "error", "message": "Uploaded CSV file not found on disk."})
        await websocket.close()
        return
        
    try:
        # Define progress callback inside the WebSocket handler
        def send_progress(msg: str, pct: int):
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    websocket.send_json({"type": "progress", "message": msg, "percent": pct}), 
                    loop
                )
        
        # Instantiate pipeline engine with progress logging
        engine = PipelineEngine(progress_callback=send_progress)
        
        # Execute the pipeline steps (runs very fast, under 5 seconds)
        results = engine.run(csv_path)
        
        # Save results in SQLite database
        db.save_run_results(
            run_id,
            results["kpi_sales"],
            results["kpi_orders"],
            results["kpi_profit"],
            results["kpi_margin"],
            results["schema"],
            results["eda"],
            results["forecast"],
            results["anomaly"],
            results["shap"]
        )
        
        db.update_run_status(run_id, "COMPLETED")
        
        # Notify WebSocket of completion
        await websocket.send_json({"type": "completed", "message": "Pipeline completed successfully!"})
        
    except Exception as e:
        db.update_run_status(run_id, "FAILED")
        await websocket.send_json({"type": "error", "message": f"Pipeline execution failed: {str(e)}"})
    finally:
        await websocket.close()


# Mount static dashboard resources if folder exists
if os.path.exists("dashboards"):
    app.mount("/dashboards", StaticFiles(directory="dashboards"), name="dashboards")

if __name__ == "__main__":
    print("\n==================================================")
    print("      Sales Analytics Intelligence Platform       ")
    print("==================================================")
    print(f" Uvicorn running on http://127.0.0.1:{PORT}")
    print("==================================================\n")
    uvicorn.run("app:app", host="127.0.0.1", port=PORT, reload=False)
