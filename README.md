# AI-Powered Sales Analytics Intelligence Platform

An end-to-end retail sales analytics platform combining time-series forecasting, seasonality profiling, explainable AI (SHAP), and outlier anomaly detection.

---

## 1. Project Overview
This project takes a raw retail transaction CSV (US Superstore Sales) and runs it through an automated, real-time analytical pipeline in a single window interface.

The entire flow is unified into a single full-stack web application powered by **FastAPI**, **WebSockets**, **SQLite**, and **Chart.js**.

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
```

---

## 3. Core Technical Components

### 1. Database Persistence Layer (SQLite3)
*   **Where it lives:** [database_manager.py](file:///C:/Users/saive/AI-Based-Sales-Data-Analytics-Platform/database_manager.py)
*   **Purpose:** All processed data is persisted. The schema consists of:
    *   `runs`: Session details, status tracks, and timestamps.
    *   `run_data`: Full schema, EDA statistics, forecast predictions, anomaly details, and SHAP vectors stored in serialized records.

### 2. Real-Time WebSockets Progress Logs
*   **Where it lives:** [app.py](file:///C:/Users/saive/AI-Based-Sales-Data-Analytics-Platform/app.py) (`/ws/pipeline`)
*   **Purpose:** Streams terminal execution logs (e.g. "Training HGB Regression...", "Detecting Isolation Forest Outliers...") with percentage coordinates in real time.

### 3. Data Structures & Algorithms
*   **Where it lives:** [sde_components.py](file:///C:/Users/saive/AI-Based-Sales-Data-Analytics-Platform/sde_components.py)
*   **LRU Cache:** Implemented using a hashing-based `OrderedDict` to cache computed model metrics. Returns values in $O(1)$ time.
*   **Token Bucket Rate Limiter:** Protects the ingestion server by rate-limiting upload requests (throttled to 10 requests per minute).

### 4. Single Window Interface (SPA)
*   **Where it lives:** [index.html](file:///C:/Users/saive/AI-Based-Sales-Data-Analytics-Platform/index.html)
*   **Purpose:** A clean, standard dashboard built using Vanilla HTML/CSS/JS and Chart.js. Steps are navigated using sidebar buttons.

---

## 4. File Inventory
*   [app.py](file:///C:/Users/saive/AI-Based-Sales-Data-Analytics-Platform/app.py): FastAPI backend server and WebSockets endpoints.
*   [database_manager.py](file:///C:/Users/saive/AI-Based-Sales-Data-Analytics-Platform/database_manager.py): SQLite3 database Singleton managers.
*   [pipeline_engine.py](file:///C:/Users/saive/AI-Based-Sales-Data-Analytics-Platform/pipeline_engine.py): Analytical pipeline executing schema checks, EDA, ensemble regressions, and anomalies.
*   [sde_components.py](file:///C:/Users/saive/AI-Based-Sales-Data-Analytics-Platform/sde_components.py): Cache structures and rate limiters.
*   [index.html](file:///C:/Users/saive/AI-Based-Sales-Data-Analytics-Platform/index.html): Clean Single Page Application visualizer.
*   [train.csv](file:///C:/Users/saive/AI-Based-Sales-Data-Analytics-Platform/train.csv): Baseline dataset (9,800 rows).

---

## 5. Compile & Run Instructions

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Start the Backend
```bash
# Start FastAPI server
python app.py
```

### Step 3 — Open the Client Visualizer
Open your browser and navigate to:
```text
http://127.0.0.1:8000
```
Drag and drop `train.csv` (or use the download option on the upload tab to get the sample dataset) to run the pipeline!
