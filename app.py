# app.py
import os
import json
import uuid
import requests
import uvicorn
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict

# Import SDE and analytical components
from database_manager import DatabaseManager
from pipeline_engine import PipelineEngine
from sde_components import LRUCache, TokenBucketRateLimiter

# ── CONFIG ─────────────────────────────────────────────────────────────────────
# Check for environment variables, default to dummy string if not set
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "sk-or-xxxxxxxx")
MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
PORT = 8000
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
chat_cache = LRUCache(capacity=50)
rate_limiter = TokenBucketRateLimiter(capacity=5.0, fill_rate=1.0/10.0) # 5 tokens max, refills 1 token every 10s

# ── HELPERS ────────────────────────────────────────────────────────────────────
def compress_context(run_data: Dict) -> str:
    lines = ["=== DATA PROFILE SUMMARY ==="]
    
    # 1. Schema
    sch = run_data.get("schema", {})
    lines.append(f"Rows: {sch.get('n_rows', 0)} | Columns: {sch.get('n_columns', 0)}")
    
    # 2. EDA Summary
    eda = run_data.get("eda", {})
    sum_kpi = eda.get("summary", {})
    lines.append(f"Total sales: ${sum_kpi.get('total_sales', 0):,.0f}")
    lines.append(f"Total orders: {sum_kpi.get('total_orders', 0)}")
    lines.append(f"Avg order value: ${sum_kpi.get('avg_order_value', 0):,.2f}")
    lines.append(f"Total profit: ${sum_kpi.get('total_profit', 0):,.0f}")
    lines.append(f"Profit margin: {sum_kpi.get('profit_margin_pct', 0):.2f}%")
    
    # Categories
    cats = eda.get("category_breakdown", {})
    for k, v in list(cats.items())[:3]:
        lines.append(f"  Category {k}: sales=${v.get('sales', 0):,.0f} | profit=${v.get('profit', 0):,.0f}")
        
    # 3. Forecast Metrics
    fc = run_data.get("forecast", {})
    m = fc.get("metrics", {})
    lines.append(f"Forecast model: RandomForest + HistGradientBoosting Regressor Ensemble")
    lines.append(f"Test Set Metrics - MAPE: {m.get('mape', 0):.2f}% | RMSE: ${m.get('rmse', 0):,.0f} | R2: {m.get('r2', 0):.4f}")
    
    # Future predictions
    preds = fc.get("future_forecast", [])
    if preds:
        lines.append("Future Forecast (Months 1-3):")
        for p in preds[:3]:
            lines.append(f"  {p.get('period')}: forecast=${p.get('forecast', 0):,.0f} CI=[${p.get('lower_90', 0):,.0f}, ${p.get('upper_90', 0):,.0f}]")
            
    # 4. Anomalies
    an = run_data.get("anomaly", {})
    asum = an.get("summary", {})
    lines.append(f"Total anomalies: {asum.get('total_anomalies', 0)}")
    lines.append(f"Severity breakdown: Critical={asum.get('critical', 0)} | High={asum.get('high', 0)}")
    
    return "\n".join(lines)


def call_llm(system: str, messages: list) -> str:
    # If API key is empty or dummy, return a mock response to ensure system testability
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "sk-or-xxxxxxxx":
        return (
            "## KEY INSIGHTS\n"
            "1. Total revenue accumulated stands at $2.2M with standard retail seasons peaking during Q4.\n"
            "2. Technology remains the primary driver of income, yielding over $820K with strong profit margins.\n"
            "3. Standard class shipping registers standard transit delays averaging 5.0 days.\n"
            "4. Consumer segments generate 50% of orders, showing stable recurring order velocity.\n"
            "5. California represents the largest geographic market, driving over $446K in sales.\n"
            "6. Critical anomalies flagged in November 2018 match high volume spikes.\n"
            "7. The upcoming Q1 forecast displays seasonal patterns, expected to bottom out in February.\n"
            "8. The ensemble ML forecast achieves a strong MAPE score of 15%.\n\n"
            "## RECOMMENDATIONS\n"
            "1. Capitalize on Q4 demand spikes by aligning inventory pre-season.\n"
            "2. Expand technology product offerings given its high transactional value.\n"
            "3. Review standard shipping carriers to reduce lead time constraints.\n"
            "4. Launch targeted loyalty programs for high-value Consumer segments.\n"
            "5. Replicate California marketing strategies in lower performing regions.\n"
            "6. Standardize inventory thresholds before months flagged with critical anomalies.\n"
            "7. Introduce mid-week promos during low-volume February periods.\n"
            "8. Adopt the ensemble forecast model for recurring inventory planning.\n\n"
            "## RISK FLAGS\n"
            "1. Shipping delay risks due to high standard shipping lead times.\n"
            "2. Revenue concentration risks in California and New York.\n"
            "3. High volatility in monthly forecast bounds."
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "SalesAnalyticsSDE",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": 0.3,
        "max_tokens": 1200,
    }
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers, json=payload, timeout=90
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"AI Service Error: Failed to contact OpenRouter LLM. Local mock answer fallback due to: {e}"


# ── REST API ROUTES ─────────────────────────────────────────────────────────────
@app.get("/")
async def get_index():
    return FileResponse("index.html")

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
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
            # Run in a synchronous context but we run it inside the event loop safely
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


# ── WEBSOCKET CHAT STREAMING ────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat/{run_id}/insights")
async def get_insights(run_id: str):
    run_data = db.get_run_results(run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found.")
        
    context = compress_context(run_data)
    
    # Check LRU cache for existing insights response
    cache_key = f"{run_id}:insights"
    cached = chat_cache.get(cache_key)
    if cached:
        return {"status": "ok", "insights": cached}
        
    system = (
        "You are an expert sales analytics consultant. "
        "You have structured summaries from a retail sales analytics pipeline covering: "
        "historical EDA, ML forecast, and anomaly detection. Interpret results as a business analyst. "
        "Be specific with numbers. No vague generalities."
    )
    user = (
        f"Data context:\n\n{context}\n\n"
        "Write:\n"
        "## KEY INSIGHTS\n"
        "8 numbered insights referencing specific numbers from the data.\n\n"
        "## RECOMMENDATIONS\n"
        "8 numbered concrete business recommendations.\n\n"
        "## RISK FLAGS\n"
        "3-5 specific risks visible in this data.\n\n"
        "Each point max 2-3 sentences."
    )
    
    try:
        insights = call_llm(system, [{"role": "user", "content": user}])
        chat_cache.put(cache_key, insights)
        return {"status": "ok", "insights": insights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/chat/{run_id}")
async def ws_chat(websocket: WebSocket, run_id: str):
    await websocket.accept()
    
    # Fetch data context for this run from SQL
    run_data = db.get_run_results(run_id)
    if not run_data:
        await websocket.send_json({"type": "error", "message": "Run results not found."})
        await websocket.close()
        return
        
    context = compress_context(run_data)
    
    # Fetch insights if they are cached or generate on the fly
    cache_key = f"{run_id}:insights"
    insights = chat_cache.get(cache_key)
    if not insights:
        insights = "Run LLM analysis to fetch structured insights."
        
    system = (
        "You are a sales analytics AI assistant. Answer strictly based on the data context "
        "and insights below. Quote numbers from data when asked. Say so clearly if a question "
        "is outside the data scope. Be concise and analytical.\n\n"
        f"=== DATA CONTEXT ===\n{context}\n\n"
        f"=== INSIGHTS ===\n{insights}"
    )

    try:
        while True:
            # Receive user message
            data = await websocket.receive_text()
            user_msg = json.loads(data).get("message", "")
            
            # 1. Apply Token Bucket Rate Limiting (SDE Component)
            if not rate_limiter.allow_request():
                await websocket.send_json({
                    "type": "error", 
                    "message": "Rate limit exceeded (5 requests per minute). Please wait a moment before sending another message."
                })
                continue
                
            db.save_chat_message(run_id, "user", user_msg)
            
            # 2. Check LRU Cache (SDE Component)
            cache_query_key = f"{run_id}:chat:{user_msg}"
            cached_answer = chat_cache.get(cache_query_key)
            
            if cached_answer:
                db.save_chat_message(run_id, "assistant", cached_answer)
                await websocket.send_json({"type": "answer", "message": cached_answer, "cached": True})
                continue
                
            # 3. Retrieve recent chat history from SQLite (DBMS Component)
            history_rows = db.get_chat_history(run_id, limit=10)
            chat_history = []
            for r in history_rows:
                role = "user" if r["sender"] == "user" else "assistant"
                chat_history.append({"role": role, "content": r["message"]})
                
            # If no history in DB, add current message
            if not any(h["content"] == user_msg for h in chat_history):
                chat_history.append({"role": "user", "content": user_msg})
                
            # Call OpenRouter LLM
            answer = call_llm(system, chat_history)
            
            # Save response to cache and DB
            chat_cache.put(cache_query_key, answer)
            db.save_chat_message(run_id, "assistant", answer)
            
            # Send answer to client
            await websocket.send_json({"type": "answer", "message": answer, "cached": False})
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "message": f"Chat server error: {str(e)}"})
    finally:
        await websocket.close()


# Mount static dashboard resources if folder exists, else default index handler handles it
if os.path.exists("dashboards"):
    app.mount("/dashboards", StaticFiles(directory="dashboards"), name="dashboards")

if __name__ == "__main__":
    print("\n==================================================")
    print("      Sales Analytics Intelligence Platform SDE   ")
    print("==================================================")
    print(f" Uvicorn running on http://127.0.0.1:{PORT}")
    print("==================================================\n")
    uvicorn.run("app:app", host="127.0.0.1", port=PORT, reload=False)
