# app.py
# Run: python app.py
# Install: pip install fastapi uvicorn python-multipart requests

import json
import requests
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

# ── CONFIG ─────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = "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"
MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
PORT  = 8000
# ───────────────────────────────────────────────────────────────────────────────

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = {
    "loaded":   {},
    "raw":      {},
    "context":  "",
    "insights": "",
    "history":  [],
}


# ── HELPERS ────────────────────────────────────────────────────────────────────
def compress(name: str, d: dict) -> str:
    lines = [f"=== {name} ==="]
    if name == "report_data.json":
        s = d.get("summary", {})
        lines.append(f"Total sales: ${s.get('total_sales', 0):,.0f}")
        lines.append(f"Total orders: {s.get('total_orders', 0)}")
        lines.append(f"Avg order value: ${s.get('avg_order_value', 0):,.2f}")
        lines.append(f"Total profit: ${s.get('total_profit', 0):,.0f}")
        lines.append(f"Profit margin: {s.get('profit_margin_pct', 0):.2f}%")
        for k, v in list(d.get("category_breakdown", {}).items())[:5]:
            lines.append(f"  Category {k}: sales=${v.get('sales',0):,.0f} profit=${v.get('profit',0):,.0f}")
        monthly = d.get("monthly_trend", [])
        if monthly:
            lines.append(f"Monthly trend: {len(monthly)} months, first={monthly[0]}, last={monthly[-1]}")
    elif name == "forecast_data.json":
        m = d.get("metrics", {})
        lines.append(f"MAPE: {m.get('mape',0):.2f}%  RMSE: ${m.get('rmse',0):,.0f}  R2: {m.get('r2',0):.4f}")
        for r in d.get("future_forecast", [])[:12]:
            lines.append(f"  {r.get('period','?')}: forecast=${r.get('forecast',0):,.0f} CI=[{r.get('lower_90',0):,.0f},{r.get('upper_90',0):,.0f}]")
        cv = d.get("cv_summary", {})
        lines.append(f"CV mean MAPE: {cv.get('mean_mape',0):.2f}%  mean MASE: {cv.get('mean_mase',0):.4f}")
    elif name == "anomaly_data.json":
        s = d.get("summary", {})
        lines.append(f"Total months: {s.get('total_months',0)}")
        lines.append(f"Confirmed anomalies: {s.get('total_anomalies',0)}")
        lines.append(f"Critical: {s.get('critical',0)}  High: {s.get('high',0)}")
        lines.append(f"Spikes: {s.get('spikes',0)}  Drops: {s.get('drops',0)}")
        lines.append(f"Anomaly rate: {s.get('anomaly_rate_pct',0)}%")
        for r in d.get("confirmed_list", [])[:5]:
            lines.append(f"  {r.get('period','?')}: sales=${r.get('sales',0):,.0f} severity={r.get('severity','?')} dir={r.get('direction','?')}")
    elif name == "schema.json":
        lines.append(f"Rows: {d.get('n_rows',0)}  Columns: {d.get('n_columns',0)}")
        cols = d.get("columns", [])
        lines.append("Columns: " + ", ".join(c["name"] for c in cols[:18]))
    else:
        for k, v in list(d.items())[:20]:
            if isinstance(v, (str, int, float, bool)):
                lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def llm_call(system: str, messages: list) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "SalesAnalytics",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": 0.3,
        "max_tokens": 1200,
    }
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers, json=payload, timeout=90
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ── ROUTES ─────────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    STATE["loaded"].clear()
    STATE["raw"].clear()
    STATE["insights"] = ""
    STATE["history"]  = []
    names = []
    for f in files:
        raw = await f.read()
        try:
            d = json.loads(raw.decode("utf-8"))
            STATE["raw"][f.filename]    = d
            STATE["loaded"][f.filename] = compress(f.filename, d)
            names.append(f.filename)
        except Exception as e:
            names.append(f"{f.filename} (parse error: {e})")
    STATE["context"] = "\n\n".join(STATE["loaded"].values())
    return {"status": "ok", "files": names}


@app.post("/analyze")
async def analyze():
    if not STATE["loaded"]:
        return JSONResponse({"error": "no_files_uploaded"}, status_code=400)

    system = (
        "You are an expert sales analytics consultant. "
        "You have structured summaries from a retail sales analytics pipeline covering: "
        "historical EDA (report_data.json), ML forecast (forecast_data.json), and "
        "anomaly detection (anomaly_data.json). Interpret results as a business analyst. "
        "Be specific with numbers. No vague generalities."
    )
    user = (
        f"Data context:\n\n{STATE['context']}\n\n"
        "Write:\n"
        "## KEY INSIGHTS\n"
        "8 numbered insights referencing specific numbers.\n\n"
        "## RECOMMENDATIONS\n"
        "8 numbered concrete business recommendations.\n\n"
        "## RISK FLAGS\n"
        "3-5 specific risks visible in this data.\n\n"
        "Each point max 2-3 sentences."
    )
    try:
        result = llm_call(system, [{"role": "user", "content": user}])
        STATE["insights"] = result
        STATE["history"]  = []
        return {"status": "ok", "insights": result}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


class ChatMsg(BaseModel):
    message: str


@app.post("/chat")
async def chat(req: ChatMsg):
    if not STATE["loaded"]:
        return JSONResponse({"error": "no_files_uploaded"}, status_code=400)
    if not STATE["insights"]:
        return JSONResponse({"error": "run_analyze_first"}, status_code=400)

    system = (
        "You are a sales analytics AI assistant. Answer strictly based on the data context "
        "and insights below. Quote numbers from data when asked. Say so clearly if a question "
        "is outside the data scope. Be concise and analytical.\n\n"
        f"=== DATA CONTEXT ===\n{STATE['context']}\n\n"
        f"=== INSIGHTS ===\n{STATE['insights']}"
    )
    STATE["history"].append({"role": "user", "content": req.message})
    try:
        answer = llm_call(system, STATE["history"][-10:])
        STATE["history"].append({"role": "assistant", "content": answer})
        return {"answer": answer}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/status")
async def status():
    return {
        "files_loaded":  list(STATE["loaded"].keys()),
        "insights_ready": bool(STATE["insights"]),
        "chat_turns":    len(STATE["history"]) // 2,
    }


# ── THIS IS WHAT WAS MISSING — without this python app.py does nothing ─────────
if __name__ == "__main__":
    print("\n Sales Analytics API starting...")
    print(" Open insights_chat.html in Chrome after you see 'Uvicorn running' below.\n")
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
