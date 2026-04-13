# ml_anomaly.py
# Run AFTER ml_forecast.py
# Requires: train.csv, forecast_data.json
# Produces: anomaly_data.json

import pandas as pd
import numpy as np
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

CSV = "train.csv"
FORECAST_JSON = "forecast_data.json"
OUT = "anomaly_data.json"
CONTAMINATION = 0.07   # expected anomaly rate ~7%
RANDOM_STATE = 42

# ── 1. Load & aggregate to monthly ──────────────────────────────────────────
df = pd.read_csv(CSV)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
df["order_date"] = pd.to_datetime(df["order_date"], dayfirst=True)
df["period"] = df["order_date"].dt.to_period("M").astype(str)

monthly = df.groupby("period")["sales"].sum().reset_index()
monthly = monthly.sort_values("period").reset_index(drop=True)
monthly["sales"] = monthly["sales"].round(2)

# ── 2. Engineer features ─────────────────────────────────────────────────────
def engineer(m):
    m = m.copy()
    m["lag1"]     = m["sales"].shift(1)
    m["lag2"]     = m["sales"].shift(2)
    m["lag12"]    = m["sales"].shift(12)
    m["roll3"]    = m["sales"].shift(1).rolling(3).mean()
    m["roll6"]    = m["sales"].shift(1).rolling(6).mean()
    m["mom_pct"]  = m["sales"].pct_change() * 100
    m["yoy_pct"]  = m["sales"].pct_change(12) * 100
    m["dev_roll3"]= m["sales"] - m["roll3"]
    m = m.dropna().reset_index(drop=True)
    return m

feat_cols = ["sales","lag1","lag2","lag12","roll3","roll6","mom_pct","dev_roll3"]
monthly_fe = engineer(monthly)

# ── 3. Statistical flags (IQR + Z-score) ─────────────────────────────────────
def stat_flags(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    z = (series - series.mean()) / series.std()
    iqr_flag = (series < lo) | (series > hi)
    z_flag   = z.abs() > 2.5
    return iqr_flag | z_flag, z.round(3), lo, hi

stat_flag, z_scores, iqr_lo, iqr_hi = stat_flags(monthly_fe["sales"])

# ── 4. Isolation Forest ───────────────────────────────────────────────────────
X = monthly_fe[feat_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso = IsolationForest(n_estimators=200, contamination=CONTAMINATION,
                      random_state=RANDOM_STATE, n_jobs=-1)
iso.fit(X_scaled)
iso_pred   = iso.predict(X_scaled)          # -1 = anomaly
iso_score  = iso.decision_function(X_scaled)  # lower = more anomalous
iso_flag   = iso_pred == -1

# ── 5. Forecast-deviation flag ────────────────────────────────────────────────
# Load backtest from forecast_data.json
with open(FORECAST_JSON) as f:
    fdata = json.load(f)

bt = pd.DataFrame(fdata.get("backtest", []))
dev_flag_map = {}
resid_map    = {}
if not bt.empty:
    bt["residual"] = bt["actual"] - bt["predicted"]
    res_std = bt["residual"].std()
    bt["dev_flag"] = bt["residual"].abs() > 1.8 * res_std
    for _, r in bt.iterrows():
        dev_flag_map[r["period"]] = bool(r["dev_flag"])
        resid_map[r["period"]]    = round(float(r["residual"]), 2)

# ── 6. Combine votes ──────────────────────────────────────────────────────────
records = []
for i, row in monthly_fe.iterrows():
    p = row["period"]
    sf = bool(stat_flag.iloc[i])
    iso_f = bool(iso_flag[i])
    dev_f = dev_flag_map.get(p, False)
    votes = int(sf) + int(iso_f) + int(dev_f)
    severity = "critical" if votes >= 3 else "high" if votes == 2 else "low" if votes == 1 else "normal"
    direction = "spike" if row["sales"] > iqr_hi else "drop" if row["sales"] < iqr_lo else "pattern"
    records.append({
        "period":        p,
        "sales":         round(float(row["sales"]), 2),
        "z_score":       round(float(z_scores.iloc[i]), 3),
        "iso_score":     round(float(iso_score[i]), 4),
        "stat_flag":     sf,
        "iso_flag":      iso_f,
        "dev_flag":      dev_f,
        "votes":         votes,
        "severity":      severity,
        "direction":     direction,
        "residual":      resid_map.get(p, None),
        "iqr_lo":        round(float(iqr_lo), 2),
        "iqr_hi":        round(float(iqr_hi), 2),
        "lag1":          round(float(row["lag1"]), 2),
        "mom_pct":       round(float(row["mom_pct"]), 2),
        "yoy_pct":       round(float(row["yoy_pct"]), 2),
    })

results_df = pd.DataFrame(records)

# ── 7. Predict anomalies on FUTURE forecast values ───────────────────────────
future = fdata.get("future_forecast", [])
future_anomalies = []
if future:
    hist_vals = monthly_fe["sales"].values
    future_records = []
    for fc in future:
        fv = fc["forecast"]
        all_vals = np.append(hist_vals, [fv])
        lag1  = float(all_vals[-2])
        lag2  = float(all_vals[-3])
        lag12 = float(all_vals[-13]) if len(all_vals) >= 13 else lag1
        roll3 = float(np.mean(all_vals[-4:-1]))
        roll6 = float(np.mean(all_vals[-7:-1]))
        mom   = float((fv - lag1) / (lag1 + 1e-9) * 100)
        dev3  = float(fv - roll3)
        row_f = np.array([[fv, lag1, lag2, lag12, roll3, roll6, mom, dev3]])
        row_scaled = scaler.transform(row_f)
        pred   = iso.predict(row_scaled)[0]
        score  = iso.decision_function(row_scaled)[0]
        # CI-based flag: outside 90% confidence interval
        ci_flag = fv < fc.get("lower_90", -np.inf) or fv > fc.get("upper_90", np.inf)
        is_anom = (pred == -1) or ci_flag
        future_anomalies.append({
            "period":    fc["period"],
            "forecast":  round(float(fv), 2),
            "lower_90":  round(float(fc.get("lower_90", 0)), 2),
            "upper_90":  round(float(fc.get("upper_90", 0)), 2),
            "iso_score": round(float(score), 4),
            "iso_flag":  pred == -1,
            "ci_flag":   ci_flag,
            "is_anomaly":is_anom,
            "mom_pct":   round(mom, 2),
            "severity":  "high" if (pred == -1 and ci_flag) else "medium" if is_anom else "normal",
        })
        hist_vals = np.append(hist_vals, fv)

# ── 8. Summary stats ──────────────────────────────────────────────────────────
confirmed_mask = results_df["votes"] >= 2
confirmed = results_df[confirmed_mask]

summary = {
    "total_months":     len(results_df),
    "total_anomalies":  int(confirmed_mask.sum()),
    "critical":         int((results_df["severity"] == "critical").sum()),
    "high":             int((results_df["severity"] == "high").sum()),
    "spikes":           int(((results_df["direction"] == "spike") & confirmed_mask).sum()),
    "drops":            int(((results_df["direction"] == "drop") & confirmed_mask).sum()),
    "anomaly_rate_pct": round(float(confirmed_mask.mean() * 100), 1),
    "max_z_score":      round(float(results_df["z_score"].abs().max()), 3),
    "future_anomalies": int(sum(1 for x in future_anomalies if x["is_anomaly"])),
    "contamination":    CONTAMINATION,
    "iqr_lo":           round(float(iqr_lo), 2),
    "iqr_hi":           round(float(iqr_hi), 2),
    "mean_sales":       round(float(monthly_fe["sales"].mean()), 2),
    "std_sales":        round(float(monthly_fe["sales"].std()), 2),
}


# ── 9. Group-level anomaly counts ─────────────────────────────────────────────
group_anomalies = {}
for grp_col in ["category", "segment", "region"]:
    if grp_col not in df.columns:
        continue
    grp_data = df.groupby([grp_col, "period"])["sales"].sum().reset_index()
    grp_out = {}
    for grp_val, gdf in grp_data.groupby(grp_col):
        gdf = gdf.sort_values("period").reset_index(drop=True)
        if len(gdf) < 6:
            continue
        sf_g, _, _, _ = stat_flags(gdf["sales"])
        anom_periods = gdf.loc[sf_g, "period"].tolist()
        grp_out[grp_val] = {
            "n_anomalies": int(sf_g.sum()),
            "anomaly_periods": anom_periods[:10],
            "monthly": gdf[["period","sales"]].rename(columns={"sales":"value"}).to_dict("records")
        }
    group_anomalies[grp_col] = grp_out

# ── 10. Month-over-month series for charts ────────────────────────────────────
full_series = []
for _, row in results_df.iterrows():
    full_series.append({
        "period":   row["period"],
        "sales":    row["sales"],
        "votes":    row["votes"],
        "severity": row["severity"],
        "direction":row["direction"],
        "z_score":  row["z_score"],
        "iso_score":row["iso_score"],
        "stat_flag":row["stat_flag"],
        "iso_flag": row["iso_flag"],
        "dev_flag": row["dev_flag"],
        "iqr_lo":   row["iqr_lo"],
        "iqr_hi":   row["iqr_hi"],
        "mom_pct":  row["mom_pct"],
        "residual": row["residual"],
    })

# ── 11. Export ────────────────────────────────────────────────────────────────
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        if isinstance(obj, (np.integer)):
            return int(obj)
        if isinstance(obj, (np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

output = {
    "summary":          summary,
    "series":           full_series,
    "future_anomalies": future_anomalies,
    "group_anomalies":  group_anomalies,
    "confirmed_list":   confirmed[["period","sales","votes","severity","direction","z_score","iso_score","mom_pct","yoy_pct","residual"]].to_dict("records"),
}

with open(OUT, "w") as f:
    json.dump(output, f, indent=2, cls=NpEncoder)

print(f"Done. {summary['total_anomalies']} confirmed anomalies out of {summary['total_months']} months.")
print(f"Future: {summary['future_anomalies']} anomalous forecast months flagged.")
print(f"Output saved to {OUT}")