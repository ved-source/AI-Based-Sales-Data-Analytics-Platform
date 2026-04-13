"""
schema_analyzer.py
------------------
Step 1 of 2.

Usage:
    python schema_analyzer.py --csv-path your_data.csv --output schema.json

What it does:
- Reads any CSV file
- Automatically detects column roles (identifier, time, categorical, numeric, text, high-cardinality)
- Detects candidate target columns (numeric columns that look like amounts/sales/revenue/price)
- Detects candidate time columns
- Writes a schema.json that analysis_engine.py uses

No hardcoded column names. Works for any sales/transactional CSV.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np


# ─── helpers ──────────────────────────────────────────────────────────────────

def std_col(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def is_likely_time(col_name: str, series: pd.Series) -> bool:
    time_keywords = ["date", "time", "datetime", "timestamp", "period", "month", "year", "day"]
    name_match = any(kw in col_name.lower() for kw in time_keywords)
    if not name_match:
        return False
    sample = series.dropna().head(100).astype(str)
    parsed = pd.to_datetime(sample, dayfirst=True, errors="coerce")
    return parsed.notna().mean() >= 0.7


def is_likely_target(col_name: str, series: pd.Series) -> bool:
    target_keywords = [
        "sales", "sale", "revenue", "amount", "price", "cost", "profit",
        "income", "value", "total", "gmv", "spend", "earning", "turnover",
        "billing", "payment", "fee", "charge", "gross", "net"
    ]
    name_match = any(kw in col_name.lower() for kw in target_keywords)
    if not name_match:
        return False
    if not pd.api.types.is_numeric_dtype(series):
        return False
    if series.nunique() < 5:
        return False
    return True


def is_identifier(col_name: str, series: pd.Series, n_rows: int) -> bool:
    id_keywords = ["id", "_id", "row", "index", "key", "uuid", "no", "num", "number"]
    name_match = any(
        col_name.lower() == kw or col_name.lower().endswith(kw) or col_name.lower().startswith(kw)
        for kw in id_keywords
    )
    unique_ratio = series.nunique() / max(n_rows, 1)
    return name_match and unique_ratio > 0.5


def classify_column(col_name: str, series: pd.Series, n_rows: int) -> str:
    if is_identifier(col_name, series, n_rows):
        return "identifier"
    if is_likely_time(col_name, series):
        return "time"
    if pd.api.types.is_numeric_dtype(series):
        unique_ratio = series.nunique() / max(n_rows, 1)
        if unique_ratio < 0.05 and series.nunique() <= 30:
            return "numeric_category"
        return "numeric_feature"
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        n_unique = series.nunique()
        unique_ratio = n_unique / max(n_rows, 1)
        if unique_ratio > 0.5 and n_unique > 50:
            return "high_cardinality_string"
        if n_unique <= 50:
            return "categorical_feature"
        avg_len = series.dropna().astype(str).str.len().mean()
        if avg_len > 30:
            return "text"
        return "high_cardinality_string"
    return "other"


def safe_sample(series: pd.Series, n: int = 5) -> list:
    vals = series.dropna().unique().tolist()[:n]
    cleaned = []
    for v in vals:
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            cleaned.append(None)
        elif isinstance(v, (np.integer,)):
            cleaned.append(int(v))
        elif isinstance(v, (np.floating,)):
            cleaned.append(float(v))
        else:
            cleaned.append(v)
    return cleaned


# ─── main ─────────────────────────────────────────────────────────────────────

def analyze(csv_path: str, output_path: str):
    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    # standardize column names
    original_names = list(df.columns)
    df.columns = [std_col(c) for c in df.columns]
    col_map = dict(zip(df.columns, original_names))  # new_name -> original

    n_rows, n_cols = df.shape
    print(f"Shape: {n_rows} rows x {n_cols} columns")

    columns_meta = []
    candidate_targets = []
    candidate_time_columns = []

    for col in df.columns:
        series = df[col]
        role = classify_column(col, series, n_rows)

        if role == "time":
            candidate_time_columns.append(col)

        if is_likely_target(col, series, ):
            candidate_targets.append(col)

        n_missing = int(series.isna().sum())
        missing_ratio = n_missing / max(n_rows, 1)
        n_unique = int(series.nunique(dropna=True))
        unique_ratio = n_unique / max(n_rows, 1)

        avg_length = None
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            avg_length = float(series.dropna().astype(str).str.len().mean()) if n_missing < n_rows else None

        dtype_clean = str(series.dtype)

        columns_meta.append({
            "name": col,
            "original_name": col_map[col],
            "role": role,
            "dtype_clean": dtype_clean,
            "dtype_raw": dtype_clean,
            "n_unique": n_unique,
            "unique_ratio": round(unique_ratio, 6),
            "n_missing": n_missing,
            "missing_ratio": round(missing_ratio, 6),
            "avg_length": round(avg_length, 4) if avg_length is not None else None,
            "sample_values": safe_sample(series),
            "candidate_target": is_likely_target(col, series),
        })

    # if no target found by keyword, pick the numeric column with highest variance
    if not candidate_targets:
        numeric_cols = [c["name"] for c in columns_meta if c["role"] == "numeric_feature"]
        if numeric_cols:
            best = max(numeric_cols, key=lambda c: df[c].var() if df[c].notna().sum() > 1 else 0)
            candidate_targets = [best]
            for c in columns_meta:
                if c["name"] == best:
                    c["candidate_target"] = True
            print(f"No target keyword found. Auto-selected: {best}")

    schema = {
        "n_rows": n_rows,
        "n_columns": n_cols,
        "columns": columns_meta,
        "candidate_targets": candidate_targets,
        "candidate_time_columns": candidate_time_columns,
    }

    Path(output_path).write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Schema written -> {output_path}")
    print(f"Candidate targets    : {candidate_targets}")
    print(f"Candidate time cols  : {candidate_time_columns}")
    print(f"Categorical columns  : {[c['name'] for c in columns_meta if c['role'] == 'categorical_feature']}")


def main():
    parser = argparse.ArgumentParser(description="Step 1: Analyze CSV schema")
    parser.add_argument("--csv-path", required=True, help="Path to input CSV file")
    parser.add_argument("--output", default="schema.json", help="Output schema JSON path (default: schema.json)")
    args = parser.parse_args()

    if not Path(args.csv_path).is_file():
        print(f"ERROR: File not found: {args.csv_path}", file=sys.stderr)
        sys.exit(1)

    analyze(args.csv_path, args.output)


if __name__ == "__main__":
    main()
