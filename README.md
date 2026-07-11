# AI-Powered Sales Analytics Intelligence Platform (SDE Flagship)

An end-to-end retail sales analytics platform demonstrating production-grade software engineering (SDE) wrappers integrated around a machine learning and data science pipeline.

---

## 1. Project Overview
This project takes a raw retail transaction CSV (US Superstore Sales) and runs it through an automated, real-time analytical pipeline in a single window interface.

Instead of running separate script fragments and manually dragging JSON files between multiple HTML dashboards, this version unifies the entire flow into a single full-stack web application powered by **FastAPI**, **WebSockets**, **SQLite**, and **Chart.js**.

---

## 2. Architecture & Pipeline Flow

```
   [ Upload CSV via UI ]
            │
            ▼
┌───────────────────────────────────────┐
│ FastAPI Ingestion Endpoints           │
└───────────┬───────────────────────────┘
            │
            ├─► Connects WebSockets (/ws/pipeline)
            │
            ▼
┌───────────────────────────────────────┐
│ PipelineEngine (Execution Pipeline)   │
│ 1. Schema Detection                   │ ◄── Streams Real-Time Progress 
│ 2. Automated EDA Report               │     & Console Logs to UI
│ 3. STL Seasonality Decompositions     │
│ 4. Ensemble ML Forecast (RF + HGB)    │
│ 5. SHAP Feature Explainability        │
│ 6. Multi-Algorithm Outlier Anomaly    │
└───────────┬───────────────────────────┘
            │
            ▼ (Saves Completed Analytics)
┌───────────────────────────────────────┐
│ SQLite Database (sales_analytics.db)  │
└───────────┬───────────────────────────┘
            │
            ├─► Fetches Data to Render Unified Charts (Chart.js)
            │
            ▼ (Grounded AI Chat Support)
┌───────────────────────────────────────┐
│ WebSockets Chat (/ws/chat)            │
│ 1. Custom Token-Bucket Rate Limiter   │
│ 2. Custom LRU Prompt Cache            │
│ 3. SQL Chat History Persistence       │
└───────────────────────────────────────┘
```

---

## 3. SDE Components Integrated

### 1. Relational DBMS (SQLite3)
*   **Where it lives:** [database_manager.py](file:///C:/Users/saive/AI-Based-Sales-Data-Analytics-Platform/database_manager.py)
*   **Purpose:** All processed data is persisted durably. The schema consists of:
    *   `runs`: Session details, status tracks, and timestamps.
    *   `run_data`: Full schema, EDA statistics, forecast predictions, anomaly details, and SHAP vectors stored in serialized records.
    *   `chat_history`: Historical AI chatbot conversation logs.

### 2. Real-Time WebSockets
*   **Where it lives:** [app.py](file:///C:/Users/saive/AI-Based-Sales-Data-Analytics-Platform/app.py) (`/ws/pipeline` and `/ws/chat`)
*   **Purpose:**
    *   `Pipeline WS`: Streams terminal execution logs (e.g. "Training HGB Regression...", "Detecting Isolation Forest Outliers...") with percentage coordinates in real time.
    *   `Chat WS`: Manages the conversational dialogue channel between the client and the OpenRouter LLM.

### 3. Data Structures & Algorithms (DSA)
*   **Where it lives:** [sde_components.py](file:///C:/Users/saive/AI-Based-Sales-Data-Analytics-Platform/sde_components.py)
*   **Custom LRU Cache:** Implemented using a hashing-based `OrderedDict` to cache expensive LLM insights and chat responses. Returns cached answers in $O(1)$ time for repeat queries.
*   **Custom Token Bucket Rate Limiter:** Protects the AI server by rate-limiting chat requests (throttled to 5 requests per minute per connection).

### 4. Single Window Interface (SPA)
*   **Where it lives:** [index.html](file:///C:/Users/saive/AI-Based-Sales-Data-Analytics-Platform/index.html)
*   **Purpose:** A modern glassmorphic dashboard built using Vanilla HTML/CSS/JS and Chart.js. Steps are navigated using clean sidebar buttons without reloading the browser.

---

## 4. File Inventory
*   [app.py](file:///C:/Users/saive/AI-Based-Sales-Data-Analytics-Platform/app.py): FastAPI backend server, WebSockets endpoints, and OpenRouter API connections.
*   [database_manager.py](file:///C:/Users/saive/AI-Based-Sales-Data-Analytics-Platform/database_manager.py): SQLite3 database Singleton managers.
*   [pipeline_engine.py](file:///C:/Users/saive/AI-Based-Sales-Data-Analytics-Platform/pipeline_engine.py): Analytical pipeline executing schema checks, EDA, ensemble regressions, and anomalies.
*   [sde_components.py](file:///C:/Users/saive/AI-Based-Sales-Data-Analytics-Platform/sde_components.py): Custom LRU Cache and Token Bucket classes.
*   [index.html](file:///C:/Users/saive/AI-Based-Sales-Data-Analytics-Platform/index.html): Glassmorphic Single Page Application visualizer.
*   [train.csv](file:///C:/Users/saive/AI-Based-Sales-Data-Analytics-Platform/train.csv): Baseline dataset (9,800 rows).

---

## 5. Compile & Run Instructions

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Start the Backend
Set your OpenRouter API Key as an environment variable or edit `app.py` directly:
```bash
# Set OpenRouter Key (Optional, fallback mock responses will be used if unset)
$env:OPENROUTER_API_KEY="your_api_key_here"

# Start FastAPI server
python app.py
```

### Step 3 — Open the Client Visualizer
Open your browser and navigate to:
```text
http://127.0.0.1:8000
```
Drag and drop `train.csv` into the connect zone to execute the SDE pipeline!
