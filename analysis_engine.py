"""
analysis_engine.py
------------------
Step 2 of 2.

Usage:
    python analysis_engine.py --csv-path your_data.csv --schema-path schema.json --output report_data.json

What it does:
- Reads the CSV + schema.json produced by schema_analyzer.py
- Computes summary statistics, aggregations, percentiles, shape stats, time trends
- Computes group-by breakdowns for all categorical columns found in schema
- Computes top-N entities for any high-cardinality string columns that look like names/products/cities
- Writes a clean report_data.json with ALL NaN/Inf values sanitized (safe for JSON + browser)

No hardcoded column names. Works for any sales/transactional CSV.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ─── helpers ──────────────────────────────────────────────────────────────────

def std_col(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def safe_float(val):
    """Convert value to JSON-safe float. NaN/Inf become null."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 6)
    except Exception:
        return None


def safe_val(val):
    """Make any scalar JSON-safe."""
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return safe_float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, str):
        return val
    try:
        f = float(val)
        return safe_float(f)
    except Exception:
        return str(val)


def sanitize_records(records: list) -> list:
    """Recursively sanitize a list of dicts so all values are JSON-safe."""
    out = []
    for row in records:
        clean = {}
        for k, v in row.items():
            clean[k] = safe_val(v)
        out.append(clean)
    return out


def detect_name_columns(schema: dict) -> dict:
    """
    Detect semantic roles for high-cardinality string columns by keyword matching.
    Returns a dict: role_name -> col_name  (e.g. "customer_name" -> "cust_name")
    """
    semantic = {}
    keyword_map = {
        "customer_name": ["customer_name", "customer", "client_name", "client", "buyer", "user_name", "user"],
        "product_name":  ["product_name", "product", "item_name", "item", "sku_name", "description"],
        "city":          ["city", "town", "municipality", "locality"],
        "order_id":      ["order_id", "order_no", "order_number", "transaction_id", "txn_id"],
    }
    hcs_cols = [c["name"] for c in schema["columns"] if c["role"] in ("high_cardinality_string", "text")]
    cat_cols  = [c["name"] for c in schema["columns"] if c["role"] == "categorical_feature"]

    for role_key, keywords in keyword_map.items():
        for col in (hcs_cols + cat_cols):
            col_lower = col.lower()
            if any(kw in col_lower for kw in keywords):
                if role_key not in semantic:
                    semantic[role_key] = col
                    break

    return semantic


# ─── load ─────────────────────────────────────────────────────────────────────

def load_data(csv_path: str, schema_path: str):
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [std_col(c) for c in df.columns]

    target_cols   = schema["candidate_targets"]
    time_cols     = schema["candidate_time_columns"]

    # parse time columns
    for tc in time_cols:
        if tc in df.columns:
            df[tc] = pd.to_datetime(df[tc], dayfirst=True, errors="coerce")

    cat_cols = [
        c["name"] for c in schema["columns"]
        if c["role"] == "categorical_feature" and 1 < c["n_unique"] <= 100
        and c["name"] in df.columns
    ]

    return df, schema, cat_cols, time_cols, target_cols


# ─── summary ──────────────────────────────────────────────────────────────────

def compute_summary(df: pd.DataFrame, schema: dict, cat_cols: list, time_cols: list, target_cols: list) -> dict:
    target = target_cols[0]
    primary_time = time_cols[0] if time_cols else None
    series = df[target].dropna()

    # KPIs
    kpis = {
        "total_target":      safe_float(series.sum()),
        "mean":              safe_float(series.mean()),
        "median":            safe_float(series.median()),
        "std":               safe_float(series.std()),
        "min":               safe_float(series.min()),
        "max":               safe_float(series.max()),
        "total_rows":        int(len(df)),
    }

    # dynamically find order/customer/product id columns for count KPIs
    order_id_candidates  = [c["name"] for c in schema["columns"] if "order" in c["name"] and "id" in c["name"] and c["name"] in df.columns]
    cust_id_candidates   = [c["name"] for c in schema["columns"] if "customer" in c["name"] and "id" in c["name"] and c["name"] in df.columns]
    prod_id_candidates   = [c["name"] for c in schema["columns"] if "product" in c["name"] and "id" in c["name"] and c["name"] in df.columns]

    kpis["unique_orders"]    = int(df[order_id_candidates[0]].nunique())  if order_id_candidates  else None
    kpis["unique_customers"] = int(df[cust_id_candidates[0]].nunique())   if cust_id_candidates   else None
    kpis["unique_products"]  = int(df[prod_id_candidates[0]].nunique())   if prod_id_candidates   else None

    if primary_time is not None:
        kpis["date_from"] = str(df[primary_time].min().date())
        kpis["date_to"]   = str(df[primary_time].max().date())
    else:
        kpis["date_from"] = None
        kpis["date_to"]   = None

    # Percentiles
    pct_vals   = np.percentile(series.dropna(), [10, 25, 50, 75, 90, 95, 99])
    pct_labels = ["p10", "p25", "p50", "p75", "p90", "p95", "p99"]
    percentiles = [{"percentile": lbl, "value": safe_float(v)} for lbl, v in zip(pct_labels, pct_vals)]

    # Shape stats
    shape_stats = {
        "skewness":   safe_float(sp_stats.skew(series.dropna())),
        "kurtosis":   safe_float(sp_stats.kurtosis(series.dropna())),
        "variance":   safe_float(series.var()),
        "cv_percent": safe_float(series.std() / series.mean() * 100) if series.mean() != 0 else None,
    }

    # Monthly trend
    monthly_records = []
    if primary_time is not None:
        df["_ym"] = df[primary_time].dt.to_period("M")
        monthly = (
            df.groupby("_ym")[target]
            .agg(total_target="sum", num_lines="count", avg_value="mean")
            .reset_index()
        )
        monthly["period"] = monthly["_ym"].astype(str)
        for _, row in monthly.iterrows():
            monthly_records.append({
                "period":       row["period"],
                "total_target": safe_float(row["total_target"]),
                "num_lines":    int(row["num_lines"]),
                "avg_value":    safe_float(row["avg_value"]),
            })

    # Yearly trend
    yearly_records = []
    if primary_time is not None:
        df["_year"] = df[primary_time].dt.year
        yearly = (
            df.groupby("_year")[target]
            .agg(total_target="sum", num_lines="count", avg_value="mean")
            .reset_index()
        )
        for _, row in yearly.iterrows():
            yearly_records.append({
                "year":         int(row["_year"]),
                "total_target": safe_float(row["total_target"]),
                "num_lines":    int(row["num_lines"]),
                "avg_value":    safe_float(row["avg_value"]),
            })

    # Lead time (difference between first and second time column if both exist)
    lead_corr = None
    if len(time_cols) >= 2:
        t0, t1 = time_cols[0], time_cols[1]
        if t0 in df.columns and t1 in df.columns:
            df["_lead_days"] = (df[t1] - df[t0]).dt.days
            c = df["_lead_days"].corr(df[target])
            lead_corr = safe_float(c)

    # Missing values
    missing = []
    for c in schema["columns"]:
        if c["n_missing"] > 0:
            missing.append({
                "name":            c["name"],
                "missing":         int(c["n_missing"]),
                "missing_percent": safe_float(c["missing_ratio"] * 100),
            })

    # Histogram (20 bins)
    hist_counts, hist_edges = np.histogram(series.dropna(), bins=20)
    hist_labels = [
        f"{hist_edges[i]:.0f}-{hist_edges[i+1]:.0f}"
        for i in range(len(hist_counts))
    ]

    return {
        "target":       target,
        "target_label": target.replace("_", " ").title(),
        "primary_time": primary_time,
        "kpis":         kpis,
        "percentiles":  percentiles,
        "shape_stats":  shape_stats,
        "monthly":      monthly_records,
        "yearly":       yearly_records,
        "lead_corr":    lead_corr,
        "missing":      missing,
        "hist_labels":  hist_labels,
        "hist_counts":  [int(v) for v in hist_counts],
    }


# ─── breakdowns ───────────────────────────────────────────────────────────────

def breakdown_by(df: pd.DataFrame, groupby_col: str, target_col: str) -> list:
    grp = df.groupby(groupby_col)[target_col].agg(
        total_target="sum",
        num_lines="count",
        avg_value="mean",
        median_value="median",
        max_value="max",
        std_value="std",
    ).reset_index()
    total = grp["total_target"].sum()
    grp["pct_of_total"] = grp["total_target"] / total * 100 if total > 0 else 0
    grp = grp.sort_values("total_target", ascending=False).reset_index(drop=True)
    return sanitize_records(grp.to_dict(orient="records"))


def compute_breakdowns(df: pd.DataFrame, schema: dict, cat_cols: list, time_cols: list, target_col: str) -> dict:
    # Group-by breakdowns for all categorical columns
    breakdowns = {}
    for cat in cat_cols:
        if cat in df.columns:
            breakdowns[cat] = breakdown_by(df, cat, target_col)

    # Detect semantic columns generically
    sem = detect_name_columns(schema)

    # Top-10 lists for name-like high cardinality columns
    top_entities = {}
    for role_key, col_name in sem.items():
        if col_name in df.columns and role_key != "order_id":
            tmp = df.groupby(col_name)[target_col].sum().nlargest(10).reset_index()
            tmp.columns = [col_name, "total_target"]
            top_entities[role_key] = {
                "col": col_name,
                "rows": sanitize_records(tmp.to_dict(orient="records")),
            }

    # Cross-tab: first two categorical columns with <= 10 unique values each
    low_card = [c for c in cat_cols if df[c].nunique() <= 10]
    crosstab = None
    stacked_datasets = []
    if len(low_card) >= 2:
        col_a, col_b = low_card[0], low_card[1]
        try:
            mat = df.groupby([col_a, col_b])[target_col].sum().unstack(fill_value=0).astype(float)
            categories = mat.index.tolist()
            cols_b     = mat.columns.tolist()
            crosstab = {
                "col_a": col_a,
                "col_b": col_b,
                "categories": [str(c) for c in categories],
                "cols_b":     [str(c) for c in cols_b],
                "matrix":     [[safe_float(v) for v in row] for row in mat.values.tolist()],
            }
            for j, cb in enumerate(cols_b):
                vals = [safe_float(mat.iloc[i, j]) for i in range(len(categories))]
                stacked_datasets.append({"label": str(cb), "data": vals})
        except Exception:
            pass

    # Lead time by a "ship mode" like column (first cat col with "ship" or "mode" or "delivery" in name)
    lead_by_group = []
    if "_lead_days" in df.columns:
        ship_candidates = [c for c in cat_cols if any(kw in c for kw in ["ship", "mode", "delivery", "method", "carrier"])]
        if ship_candidates:
            sc = ship_candidates[0]
            tmp = (
                df.groupby(sc)["_lead_days"]
                .agg(avg_days="mean", median_days="median", min_days="min", max_days="max")
                .reset_index()
            )
            lead_by_group = sanitize_records(tmp.to_dict(orient="records"))

    # Monthly order volume (unique order ids per month)
    order_volume = []
    order_id_candidates = [c["name"] for c in schema["columns"] if "order" in c["name"] and "id" in c["name"] and c["name"] in df.columns]
    if "_ym" in df.columns and order_id_candidates:
        oid = order_id_candidates[0]
        tmp = df.groupby("_ym")[oid].nunique().reset_index()
        tmp.columns = ["period", "num_orders"]
        tmp["period"] = tmp["period"].astype(str)
        order_volume = tmp.to_dict(orient="records")

    return {
        "target":           target_col,
        "target_label":     target_col.replace("_", " ").title(),
        "breakdowns":       breakdowns,
        "breakdown_cols":   list(breakdowns.keys()),
        "top_entities":     top_entities,
        "crosstab":         crosstab,
        "stacked_datasets": stacked_datasets,
        "lead_by_group":    lead_by_group,
        "order_volume":     order_volume,
    }


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 2: Compute summary + breakdown report data")
    parser.add_argument("--csv-path",    required=True, help="Path to input CSV file")
    parser.add_argument("--schema-path", required=True, help="Path to schema.json from schema_analyzer.py")
    parser.add_argument("--output",      default="report_data.json", help="Output JSON path (default: report_data.json)")
    args = parser.parse_args()

    for p in [args.csv_path, args.schema_path]:
        if not Path(p).is_file():
            print(f"ERROR: File not found: {p}", file=sys.stderr)
            sys.exit(1)

    df, schema, cat_cols, time_cols, target_cols = load_data(args.csv_path, args.schema_path)

    if not target_cols:
        print("ERROR: No candidate target columns found. Check schema.json.", file=sys.stderr)
        sys.exit(1)

    print(f"Target column : {target_cols[0]}")
    print(f"Time columns  : {time_cols}")
    print(f"Cat columns   : {cat_cols}")

    summary   = compute_summary(df, schema, cat_cols, time_cols, target_cols)
    breakdown = compute_breakdowns(df, schema, cat_cols, time_cols, target_cols[0])

    report = {
        "schema":    schema,
        "summary":   summary,
        "breakdown": breakdown,
    }

    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Report written -> {args.output}")


if __name__ == "__main__":
    main()
