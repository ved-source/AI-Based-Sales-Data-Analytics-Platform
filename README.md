# Sales Analytics Intelligence Platform

> End-to-end retail sales analytics pipeline — from raw CSV to ML forecasting, anomaly detection, SHAP explainability, and an AI-powered chat interface. Built as a placement project demonstrating production-grade data science and full-stack deployment.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Live Demo](#live-demo)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Pipeline Stages](#pipeline-stages)
  - [Stage 1 — Data Cleaning & Schema Detection](#stage-1--data-cleaning--schema-detection)
  - [Stage 2 — Exploratory Data Analysis & Report](#stage-2--exploratory-data-analysis--report)
  - [Stage 3 — ML Forecasting](#stage-3--ml-forecasting)
  - [Stage 4 — SHAP Explainability](#stage-4--shap-explainability)
  - [Stage 5 — Anomaly Detection](#stage-5--anomaly-detection)
  - [Stage 6 — AI Insights & Chat Interface](#stage-6--ai-insights--chat-interface)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Key Results & Metrics](#key-results--metrics)
- [Dashboards](#dashboards)
- [API Reference](#api-reference)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Project Overview

This project is a **full-stack, end-to-end sales analytics intelligence platform** built on a real-world US retail superstore dataset (9,800 orders, 2015–2018). It covers every stage of a production data science pipeline:

1. Automated schema detection and data quality profiling
2. Deep exploratory analysis with interactive browser dashboards
3. Ensemble ML forecasting (HistGradientBoosting + Random Forest) with walk-forward cross-validation
4. SHAP-based model explainability (global importance, dependence plots, per-prediction waterfall charts)
5. Multi-method anomaly detection (Z-score, IQR, Isolation Forest, STL residuals) with severity scoring
6. LLM-powered insights generation and conversational chat interface via OpenRouter API

Every stage outputs a structured JSON file that feeds the next stage, and all dashboards run entirely in the browser with no server required — just open an HTML file and upload the JSON.

---

## Live Demo

| Component | How to run |
|---|---|
| Forecast Dashboard | Open `forecast_dashboard.html` in Chrome, upload `forecast_data.json` |
| Anomaly Dashboard | Open `anomaly_dashboard.html` in Chrome, upload `anomaly_data.json` |
| AI Chat + Insights | Run `python app.py`, open `insights_chat.html` |

---

## Architecture

```
train.csv
    │
    ▼
┌─────────────────────────────┐
│  data_cleaning.py           │  Schema detection, type inference,
│  schema_detector.py         │  missing value audit → schema.json
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  eda_report.py              │  Monthly trends, breakdowns by category /
│  report_dashboard.html      │  segment / region / state → report_data.json
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  ml_forecast.py             │  Feature engineering (lags, rolling stats,
│  forecast_dashboard.html    │  STL decomposition, calendar features),
│                             │  HGB + RF ensemble, walk-forward CV,
│                             │  recursive 12-month future forecast
│                             │  → forecast_data.json
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  ml_shap.py                 │  Exact SHAP values (RF), TreeExplainer
│  (SHAP section in           │  (HGB), group importance, beeswarm,
│   forecast_dashboard.html)  │  dependence plots, waterfall per month
│                             │  → shap_data.json
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  anomaly_detection.py       │  Z-score, IQR, Isolation Forest, STL
│  anomaly_dashboard.html     │  residuals, ensemble voting, severity
│                             │  scoring, future anomaly flags
│                             │  → anomaly_data.json
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  app.py  (FastAPI)          │  Ingests all JSON outputs, compresses
│  insights_chat.html         │  context, calls OpenRouter LLM for
│                             │  insights + recommendations, serves
│                             │  multi-turn chat API
└─────────────────────────────┘
```

---

## Dataset

**Source:** US Superstore Sales Dataset (Kaggle)

| Property | Value |
|---|---|
| Rows | 9,800 orders |
| Columns | 18 |
| Date range | January 2015 — December 2018 |
| Target variable | `sales` (transaction-level, aggregated to monthly) |
| Geography | 49 US states, 529 cities |
| Categories | Furniture, Office Supplies, Technology |
| Segments | Consumer, Corporate, Home Office |
| Unique customers | 793 |
| Unique products | 1,861 |
| Missing values | 11 rows in `postal_code` only (0.11%) |

**Key statistics:**

| Metric | Value |
|---|---|
| Total sales (4 years) | $2,261,537 |
| Mean transaction | $230.77 |
| Median transaction | $54.49 |
| Std deviation | $626.65 |
| Skewness | 12.98 (highly right-skewed) |
| Kurtosis | 304.29 |
| YoY growth 2015→2018 | +50.5% |

---

## Pipeline Stages

### Stage 1 — Data Cleaning & Schema Detection

**Script:** `data_cleaning.py`, `schema_detector.py`  
**Output:** `schema.json`

- Automatically infers column roles: `identifier`, `time`, `categorical_feature`, `numeric_feature`, `high_cardinality_string`, `text`
- Detects candidate target columns and candidate time columns
- Computes per-column statistics: null rates, unique ratios, sample values, average string length
- Flags data quality issues (missing values, type mismatches)
- Outputs a full schema contract used by all downstream scripts

**Schema detected:**
- Target: `sales`
- Primary time: `order_date`
- 3 categorical features: `segment`, `category`, `region`, `ship_mode`
- 2 high-cardinality strings: `customer_id`, `product_id`
- 1 numeric with missing: `postal_code`

---

### Stage 2 — Exploratory Data Analysis & Report

**Script:** `eda_report.py`  
**Output:** `report_data.json`  
**Dashboard:** `report_dashboard.html`

Full EDA with the following computed sections:

- **Summary KPIs:** total sales, total orders, average order value, profit margin, date range
- **Monthly trend:** 48-month time series (2015-01 to 2018-12) with order volume overlay
- **Yearly aggregates:** 2015 ($479,856), 2016 ($459,436), 2017 ($600,193), 2018 ($722,052)
- **Segment breakdown:** Consumer 50.8%, Corporate 30.4%, Home Office 18.8%
- **Category breakdown:** Technology 36.6% ($827K), Furniture 32.2% ($729K), Office Supplies 31.2% ($705K)
- **Regional breakdown:** West 31.4%, East 29.6%, Central 21.8%, South 17.2%
- **Top states:** California ($446K), New York ($306K), Texas ($169K), Washington ($135K)
- **Top cities:** New York City ($252K), Los Angeles ($173K), Seattle ($116K)
- **Top customers:** Sean Miller ($25K), Tamara Chand ($19K), Raymond Buch ($15K)
- **Top products:** Canon imageCLASS 2200 ($61.6K), Fellowes PB500 ($27.5K)
- **Top subcategories:** Phones ($328K), Chairs ($323K), Storage ($219K), Tables ($203K)
- **Shipping lead time:** Same Day (0 days), First Class (2.2 days), Second Class (3.2 days), Standard Class (5.0 days)
- **Category × Region cross-matrix:** 4×3 heatmap of sales distribution
- **Sales distribution:** Histogram with log scale (heavily right-skewed, 96.2% of transactions under $1,132)

---

### Stage 3 — ML Forecasting

**Script:** `ml_forecast.py`  
**Output:** `forecast_data.json`  
**Dashboard:** `forecast_dashboard.html`

#### Feature Engineering

Monthly time series assembled from raw transaction data. Features engineered:

| Feature Group | Features |
|---|---|
| Lag features | Lag 1, 2, 3, 6, 12 months |
| Rolling statistics | 3-month mean, std; 6-month mean, std; 12-month mean |
| EMA features | EMA-3, EMA-6, EMA-12 |
| Calendar | Month, quarter, year, is_q4 flag |
| STL decomposition | Trend component, seasonal component, residual |
| Growth rates | MoM change, YoY change |
| Interaction | Lag1 × seasonal, trend × quarter |

#### Model

**Ensemble: HistGradientBoostingRegressor + RandomForestRegressor**

- Target transformed to `log1p(sales)` to handle right skew
- Both models trained independently and averaged at prediction time
- Recursive multi-step forecasting for the 12-month future horizon

#### Walk-Forward Cross-Validation

5-fold time-series cross-validation with expanding window:

| Fold | Train months | Test period |
|---|---|---|
| 1 | 24 | 2017-01 to 2017-06 |
| 2 | 30 | 2017-07 to 2017-12 |
| 3 | 36 | 2018-01 to 2018-06 |
| 4 | 42 | 2018-07 to 2018-09 |
| 5 | 45 | 2018-10 to 2018-12 |

#### Model Performance (Test Set)

| Metric | Value | Interpretation |
|---|---|---|
| MAPE | ~15% | Good for monthly retail forecasting |
| RMSE | Computed per run | Root mean squared error on holdout |
| R² | ~0.85+ | Variance explained |
| MASE | < 1.0 | Beats naive lag-1 benchmark |
| CV Mean MAPE | Stable across folds | No significant fold-to-fold degradation |

#### Future Forecast (2019)

12-month recursive forecast with 90% confidence intervals:

| Month | Forecast | Lower 90% | Upper 90% |
|---|---|---|---|
| 2019-01 | $20,185 | — | — |
| 2019-03 | $42,174 | — | — |
| 2019-09 | $69,482 | — | — |
| 2019-11 | $74,660 | — | — |
| 2019-12 | $74,914 | — | — |

Strong Q4 peak ($72K–$75K) consistent with historical seasonality pattern.

---

### Stage 4 — SHAP Explainability

**Script:** `ml_shap.py`  
**Output:** `shap_data.json`  
**Section:** SHAP tab inside `forecast_dashboard.html`

- **Exact SHAP values** computed for RandomForestRegressor using `shap.TreeExplainer`
- **TreeExplainer** used for HistGradientBoostingRegressor
- **Global importance:** Mean absolute SHAP per feature, ranked for both models
- **Feature group importance:** Lag features, Rolling/EMA, STL components, Calendar, Interactions — donut chart + grouped bar
- **SHAP vs Gini alignment:** Scatter plot confirming both methods agree on top features
- **Stacked SHAP per month:** Beeswarm-style stacked bar showing which features drove each month's prediction up or down
- **Dependence plots:** SHAP value vs raw feature value for top features
- **Waterfall charts:** Per-prediction explanation for every test month showing individual feature contributions
- **Full SHAP table:** All features ranked with RF SHAP, HGB SHAP, Gini score, and contribution bar

---

### Stage 5 — Anomaly Detection

**Script:** `anomaly_detection.py`  
**Output:** `anomaly_data.json`  
**Dashboard:** `anomaly_dashboard.html`

#### Methods Used

| Method | What it detects |
|---|---|
| Z-score (threshold 2.0) | Months far from the mean in standard deviation units |
| IQR (1.5× fence) | Months outside the interquartile range fence |
| Isolation Forest | Structural outliers based on tree partitioning |
| STL Residuals | Deviations from trend+seasonal decomposition |

**Ensemble voting:** A month is flagged as a confirmed anomaly only when at least 2 of the 4 methods agree.

#### Severity Scoring

Each confirmed anomaly is scored on three dimensions:
- **Z-score magnitude** (how many standard deviations away)
- **Direction** (spike vs drop)
- **Consensus** (how many methods flagged it)

Combined into a `severity` label: `critical`, `high`, `medium`.

#### Results Summary

| Metric | Value |
|---|---|
| Total months analysed | 48 |
| Confirmed anomalies | Computed per run |
| Critical | Flagged for immediate review |
| High severity | Requires investigation |
| Anomaly rate | ~10–15% of months |
| Future months flagged | Q4 2019 (forecast variance-based) |

---

### Stage 6 — AI Insights & Chat Interface

**Backend:** `app.py` (FastAPI + Uvicorn)  
**Frontend:** `insights_chat.html`  
**LLM:** OpenRouter API (`meta-llama/llama-3.3-70b-instruct:free` or `nvidia/nemotron-3-super-120b-a12b:free`)

#### How it works

1. User uploads all pipeline JSON outputs via drag-and-drop in the browser
2. Backend compresses each file into a structured text summary (preserving all key numbers)
3. LLM is called with a detailed system prompt instructing it to act as a sales analytics consultant
4. LLM generates:
   - **8 numbered Key Insights** (each referencing specific numbers from the data)
   - **8 numbered Recommendations** (concrete, actionable, tied to the insights)
   - **3–5 Risk Flags** (specific risks visible in the data)
5. The full compressed context + generated insights are stored in memory as the chat system prompt
6. User can then ask any question in the chat interface — the LLM answers grounded strictly in the uploaded data with multi-turn conversation memory (last 10 turns)

#### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload JSON files, compress and store in memory |
| `POST` | `/analyze` | Call LLM to generate insights + recommendations |
| `POST` | `/chat` | Multi-turn chat grounded in data context |
| `GET` | `/status` | Health check: files loaded, insights ready, chat turns |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data processing | Python 3.10, Pandas, NumPy |
| ML models | Scikit-learn (HistGradientBoosting, RandomForest, IsolationForest) |
| Time series | statsmodels (STL decomposition), custom walk-forward CV |
| Explainability | SHAP (TreeExplainer, Exact) |
| Backend API | FastAPI, Uvicorn, Pydantic |
| LLM integration | OpenRouter API (Llama 3.3 70B / Nemotron 120B) |
| Frontend | Vanilla HTML/CSS/JS, Chart.js 4.4.3 |
| Data exchange | JSON (structured pipeline contracts) |
| Development | VS Code, PowerShell, Git |

---

## Project Structure

```
project/
│
├── data/
│   └── train.csv                    # Raw superstore dataset (9,800 rows)
│
├── scripts/
│   ├── data_cleaning.py             # Data cleaning and validation
│   ├── schema_detector.py           # Automated schema inference
│   ├── eda_report.py                # EDA and report generation
│   ├── ml_forecast.py               # Feature engineering + ML forecast
│   ├── ml_shap.py                   # SHAP explainability
│   └── anomaly_detection.py         # Multi-method anomaly detection
│
├── outputs/                         # All pipeline JSON outputs
│   ├── schema.json
│   ├── report_data.json
│   ├── forecast_data.json
│   ├── shap_data.json
│   └── anomaly_data.json
│
├── dashboards/                      # Browser-based dashboards (no server needed)
│   ├── report_dashboard.html
│   ├── forecast_dashboard.html      # Also contains SHAP section
│   └── anomaly_dashboard.html
│
├── app.py                           # FastAPI backend for LLM integration
├── insights_chat.html               # AI insights + chat frontend
│
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- pip
- A modern browser (Chrome recommended)
- An OpenRouter API key (free at [openrouter.ai](https://openrouter.ai))

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/sales-analytics-platform.git
cd sales-analytics-platform
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
statsmodels>=0.13.0
shap>=0.42.0
fastapi>=0.100.0
uvicorn[standard]>=0.22.0
python-multipart>=0.0.6
requests>=2.28.0
pydantic>=2.0.0
```

### Step 3 — Run the full pipeline

```bash
# 1. Schema detection
python scripts/schema_detector.py --input data/train.csv --output outputs/schema.json

# 2. EDA report
python scripts/eda_report.py --input data/train.csv --output outputs/report_data.json

# 3. ML Forecast
python scripts/ml_forecast.py --input data/train.csv --output outputs/forecast_data.json

# 4. SHAP explainability
python scripts/ml_shap.py --input data/train.csv --output outputs/shap_data.json

# 5. Anomaly detection
python scripts/anomaly_detection.py --input data/train.csv --output outputs/anomaly_data.json
```

### Step 4 — Start the AI backend

```bash
# Add your OpenRouter key to app.py line 12 first
python app.py
```

You will see:
```
Sales Analytics API starting...
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Step 5 — Open the dashboards

```bash
# Dashboards — open directly in Chrome (no server needed)
start dashboards/forecast_dashboard.html
start dashboards/anomaly_dashboard.html

# AI chat interface — requires app.py to be running
start insights_chat.html
```

---

## Usage Guide

### Running the Forecast Dashboard

1. Open `forecast_dashboard.html` in Chrome
2. Drop `forecast_data.json` onto the upload zone
3. Explore: KPI cards, cross-validation folds, backtest chart, future forecast table, group forecasts, feature importance
4. Optionally upload `shap_data.json` in the SHAP section for explainability

### Running the Anomaly Dashboard

1. Open `anomaly_dashboard.html` in Chrome
2. Drop `anomaly_data.json` onto the upload zone
3. View: anomaly timeline, severity breakdown, method agreement matrix, future risk flags

### Running the AI Chat Interface

1. Start `python app.py` in your terminal — keep it running
2. Open `insights_chat.html` in Chrome
3. Drag and drop all JSON files (`report_data.json`, `forecast_data.json`, `anomaly_data.json`, `schema.json`) into the upload zone
4. Click **Upload to backend**
5. Go to the **Insights** tab, click **Run LLM Analysis** — wait ~10–15 seconds
6. Read the generated insights and recommendations
7. Go to the **Chat** tab and ask anything:
   - *"Which months had the highest anomaly risk and why?"*
   - *"What is the forecast accuracy and is it reliable?"*
   - *"Which product category should we prioritise in Q4 2019?"*
   - *"What drove the spike in November 2018?"*

---

## Key Results & Metrics

### Sales Growth

| Year | Total Sales | Orders | YoY Growth |
|---|---|---|---|
| 2015 | $479,856 | 1,953 | Baseline |
| 2016 | $459,436 | 2,055 | -4.3% |
| 2017 | $600,193 | 2,534 | +30.6% |
| 2018 | $722,052 | 3,258 | +20.3% |

### Forecast Model Performance

| Metric | Value |
|---|---|
| MAPE (test set) | ~15% |
| MASE (vs naive) | < 1.0 (model beats naive benchmark) |
| CV stability | Consistent across 5 time-series folds |
| Forecast horizon | 12 months recursive |

### Top Business Insights (from data)

- **Technology is the revenue leader** at $827K (36.6%) despite having only 1,813 line items — highest average transaction value at $456
- **Q4 dominates** every year: September, November, December consistently account for 40%+ of annual sales
- **Standard Class shipping** handles 59.3% of all orders but averages 5 days lead time — significant customer experience risk
- **California + New York** alone account for 33.3% of total revenue ($752K)
- **Machines and Copiers** have the highest average transaction values ($1,646 and $2,216 respectively) but low volumes — high-value, low-frequency risk
- **February is the worst month** every year — targeted promotions needed
- **Consumer segment** at 50.8% of revenue but Home Office has the highest average transaction ($243) — underserved premium segment

---

## Dashboards

### Forecast Dashboard (`forecast_dashboard.html`)

- Model performance KPI cards (MAPE, RMSE, R², MASE, CV metrics)
- Walk-forward cross-validation charts (MAPE per fold, MAE per fold, fold detail table)
- Backtest: actual vs predicted monthly chart + scatter plot + residual distribution
- Recursive future forecast chart with 90% confidence band + full table with MoM change
- Group-level forecasts by Category, Segment, Region
- Feature importance (Gini) — top 20 bar chart + ranked table
- SHAP section (upload `shap_data.json`):
  - Global SHAP importance (RF vs HGB comparison)
  - Feature group donut chart + group bar chart
  - SHAP vs Gini alignment scatter
  - Stacked SHAP per month (top 8 features)
  - Dependence plots for top features
  - Waterfall per-prediction chart for every test month
  - Full SHAP table with all features

### Anomaly Dashboard (`anomaly_dashboard.html`)

- Summary KPI cards (total anomalies, critical count, anomaly rate, future flags)
- Timeline chart with anomaly markers by severity
- Method agreement matrix (which methods flagged which months)
- Severity distribution chart
- Full confirmed anomalies table with Z-score, direction, and severity
- Future forecast anomaly risk section

### AI Insights & Chat (`insights_chat.html`)

- 4-tab interface: Setup, Insights, Chat, Dashboards
- Drag-and-drop file upload with live backend status indicator
- LLM-generated insights section with formatted headings
- Full multi-turn chat with conversation history
- Embedded iframe view of all previous dashboards
- Dark/light theme toggle

---

## API Reference

The FastAPI backend (`app.py`) runs on `http://localhost:8000`.

### `POST /upload`

Upload one or more JSON files from the pipeline.

**Request:** `multipart/form-data` with `files[]`

**Response:**
```json
{
  "status": "ok",
  "files": ["report_data.json", "forecast_data.json", "anomaly_data.json", "schema.json"]
}
```

### `POST /analyze`

Trigger LLM analysis of uploaded files. Generates insights, recommendations, and risk flags.

**Response:**
```json
{
  "status": "ok",
  "insights": "## KEY INSIGHTS
1. ...

## RECOMMENDATIONS
1. ..."
}
```

### `POST /chat`

Send a chat message. Returns LLM response grounded in data context.

**Request body:**
```json
{ "message": "Which months had anomalies?" }
```

**Response:**
```json
{ "answer": "Based on the anomaly data, ..." }
```

### `GET /status`

Check backend state.

**Response:**
```json
{
  "files_loaded": ["report_data.json"],
  "insights_ready": true,
  "chat_turns": 3
}
```

---

## Future Improvements

- [ ] Add Prophet and ARIMA models as ensemble members for comparison
- [ ] Integrate a vector database (ChromaDB / FAISS) for RAG — so the LLM can quote exact numbers from the JSON instead of relying on compressed summaries
- [ ] Add profit margin and discount data to the forecasting pipeline
- [ ] Build a PostgreSQL backend to persist chat history and analysis runs across sessions
- [ ] Deploy on Railway / Render with a public URL (currently localhost only)
- [ ] Add automated weekly retraining trigger when new CSV is uploaded
- [ ] Extend anomaly detection to subcategory and region level (currently aggregate only)
- [ ] Add a final PDF report export button in the dashboard

---

## Author

**Ved** — B.Tech Computer Science  
Hyderabad, India

Built as a placement project demonstrating end-to-end data science, ML engineering, and full-stack deployment skills.

- GitHub: [github.com/VED-SOURCE](https://github.com/ved-source)
- LinkedIn: [linkedin.com/in/SAI-VED](https://www.linkedin.com/in/sai-ved-713176315)

---

> If you found this useful, please star the repository.
