import pandas as pd
import numpy as np
import json
import shap
import warnings
import joblib
warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.seasonal import STL

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — must match ml_forecast.py exactly
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH        = "train.csv"
DATE_COL        = "Order Date"
SALES_COL       = "Sales"
ORDER_ID_COL    = "Order ID"
PRODUCT_ID_COL  = "Product ID"
CUSTOMER_ID_COL = "Customer ID"
N_LAGS          = 6
HOLDOUT         = 12
DATE_FORMAT     = "dayfirst"
JSON_OUT        = "shap_data.json"

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & BUILD FEATURE MATRIX  (identical to ml_forecast.py)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
df.rename(columns={"sub-category":"sub_category"}, inplace=True)
df[DATE_COL.lower().replace(" ","_")] = pd.to_datetime(
    df[DATE_COL.lower().replace(" ","_")], dayfirst=(DATE_FORMAT=="dayfirst")
)
date_col = DATE_COL.lower().replace(" ","_")
sales_col= SALES_COL.lower().replace(" ","_")

monthly = df.groupby(df[date_col].dt.to_period("M"))[sales_col].sum().reset_index()
monthly.columns = ["period","sales"]
monthly = monthly.sort_values("period").reset_index(drop=True)
monthly["date"]       = monthly["period"].dt.to_timestamp()
monthly["period_str"] = monthly["period"].astype(str)

txn = df.groupby(df[date_col].dt.to_period("M")).agg(
    n_orders    =(ORDER_ID_COL.lower().replace(" ","_"),    "nunique"),
    n_products  =(PRODUCT_ID_COL.lower().replace(" ","_"),  "nunique"),
    n_customers =(CUSTOMER_ID_COL.lower().replace(" ","_"), "nunique"),
    avg_txn_sale=(sales_col, "mean"),
).reset_index()
txn.columns = ["_period","n_orders","n_products","n_customers","avg_txn_sale"]
txn["period_str"] = txn["_period"].astype(str)
txn.drop("_period", axis=1, inplace=True)
monthly = monthly.merge(txn, on="period_str", how="left")

stl_res = STL(monthly["sales"], period=12, robust=True).fit()
monthly["stl_trend"]    = stl_res.trend
monthly["stl_seasonal"] = stl_res.seasonal

def build_features(df_m, n_lags=6):
    d = df_m.copy().reset_index(drop=True)
    d["sales_log"] = np.log1p(d["sales"])
    d["month"]       = d["date"].dt.month
    d["year"]        = d["date"].dt.year
    d["quarter"]     = d["date"].dt.quarter
    d["month_sin"]   = np.sin(2*np.pi*d["month"]/12)
    d["month_cos"]   = np.cos(2*np.pi*d["month"]/12)
    d["quarter_sin"] = np.sin(2*np.pi*d["quarter"]/4)
    d["quarter_cos"] = np.cos(2*np.pi*d["quarter"]/4)
    d["trend_idx"]   = np.arange(len(d))
    for lag in range(1, n_lags+1):
        d["lag_"+str(lag)] = d["sales"].shift(lag)
    s1 = d["sales"].shift(1)
    d["roll3_mean"]  = s1.rolling(3, min_periods=2).mean()
    d["roll6_mean"]  = s1.rolling(6, min_periods=3).mean()
    d["roll12_mean"] = s1.rolling(12,min_periods=4).mean()
    d["roll3_std"]   = s1.rolling(3, min_periods=2).std().fillna(0)
    d["roll6_std"]   = s1.rolling(6, min_periods=3).std().fillna(0)
    d["roll3_max"]   = s1.rolling(3, min_periods=2).max()
    d["roll6_max"]   = s1.rolling(6, min_periods=3).max()
    d["roll3_min"]   = s1.rolling(3, min_periods=2).min()
    d["ema3"]        = s1.ewm(span=3,  adjust=False).mean()
    d["ema6"]        = s1.ewm(span=6,  adjust=False).mean()
    d["ema12"]       = s1.ewm(span=12, adjust=False).mean()
    d["yoy_lag12"]   = d["sales"].shift(12)
    d["yoy_growth"]  = (s1/(d["sales"].shift(13)+1e-9)-1).fillna(0).clip(-3,5)
    d["mom1"]        = s1 - d["sales"].shift(2)
    d["mom3"]        = s1 - d["sales"].shift(4)
    if "stl_trend" in df_m.columns:
        d["stl_trend_feat"]    = d["stl_trend"]
        d["stl_seasonal_feat"] = d["stl_seasonal"]
        d["stl_trend_lag1"]    = d["stl_trend"].shift(1)
    for col in ["n_orders","n_products","n_customers","avg_txn_sale"]:
        if col in df_m.columns:
            d["txn_"+col] = d[col].shift(1)
    return d

mf = build_features(monthly, n_lags=N_LAGS)
mf = mf.dropna().reset_index(drop=True)

EXCLUDE = {"period","period_str","date","sales","sales_log",
           "n_orders","n_products","n_customers","avg_txn_sale","stl_trend","stl_seasonal"}
feat_cols = [c for c in mf.columns if c not in EXCLUDE and
             not c.startswith("period") and mf[c].dtype in [np.float64,np.int64,float,int]]

X_all  = mf[feat_cols].values.astype(float)
y_all  = mf["sales"].values
yl_all = mf["sales_log"].values
dates  = mf["period_str"].values

X_tr = X_all[:-HOLDOUT]; y_tr = yl_all[:-HOLDOUT]; y_tr_raw = y_all[:-HOLDOUT]
X_te = X_all[-HOLDOUT:]; y_te = y_all[-HOLDOUT:];  dates_te  = dates[-HOLDOUT:]

# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("Training RF...")
rf = RandomForestRegressor(n_estimators=400, max_depth=7, min_samples_leaf=2,
                            max_features=0.7, n_jobs=-1, random_state=42)
rf.fit(X_tr, y_tr)

print("Training HGB...")
hgb = HistGradientBoostingRegressor(
    loss="squared_error", learning_rate=0.04, max_iter=800,
    max_depth=4, max_leaf_nodes=15, min_samples_leaf=2,
    l2_regularization=1.0, early_stopping=True,
    validation_fraction=0.15, n_iter_no_change=25, random_state=42
)
hgb.fit(X_tr, y_tr)
print("HGB iters:", hgb.n_iter_)

# ─────────────────────────────────────────────────────────────────────────────
# 3. SHAP — RF: TreeExplainer (exact, fast)
# ─────────────────────────────────────────────────────────────────────────────
print("Computing SHAP (RF TreeExplainer)...")
rf_explainer  = shap.TreeExplainer(rf, data=X_tr, feature_names=feat_cols)
shap_tr       = rf_explainer.shap_values(X_tr)
shap_te       = rf_explainer.shap_values(X_te)
shap_all_vals = rf_explainer.shap_values(X_all)
rf_base       = float(rf_explainer.expected_value)
print("RF base (log):", round(rf_base,4), "=> raw $", round(float(np.expm1(rf_base)),0))

# ─────────────────────────────────────────────────────────────────────────────
# 4. SHAP — HGB: KernelExplainer
# ─────────────────────────────────────────────────────────────────────────────
print("Computing SHAP (HGB KernelExplainer)...")
bg             = shap.kmeans(X_tr, min(10, len(X_tr)))
hgb_kern       = shap.KernelExplainer(hgb.predict, bg, feature_names=feat_cols)
shap_hgb_tr    = hgb_kern.shap_values(X_tr, nsamples=200)
shap_hgb_te    = hgb_kern.shap_values(X_te, nsamples=200)
hgb_base       = float(hgb_kern.expected_value)
print("HGB base (log):", round(hgb_base,4))

# ─────────────────────────────────────────────────────────────────────────────
# 5. BUILD shap_data.json
# ─────────────────────────────────────────────────────────────────────────────
def sr(v, d=4):
    return round(float(v), d)

def group_label(f):
    if f.startswith("lag_"):                return "Lag Features"
    if f.startswith("roll") or f.startswith("ema"): return "Rolling / EMA"
    if "yoy" in f or "mom" in f:            return "YoY / Momentum"
    if f.startswith("stl"):                 return "STL Decomposition"
    if f.startswith("txn_"):                return "Transaction Stats"
    if f in ["month","year","quarter","month_sin","month_cos",
             "quarter_sin","quarter_cos","trend_idx"]:
        return "Calendar / Trend"
    return "Other"

rf_mean_abs  = np.mean(np.abs(shap_all_vals), axis=0)
hgb_mean_abs = np.mean(np.abs(shap_hgb_tr),   axis=0)
order        = np.argsort(rf_mean_abs)[::-1]
total_rf     = float(rf_mean_abs.sum())
total_hgb    = float(hgb_mean_abs.sum())

# global importance
global_imp = []
for i in order:
    global_imp.append({
        "feature":         feat_cols[i],
        "group":           group_label(feat_cols[i]),
        "rf_mean_abs":     sr(rf_mean_abs[i],  6),
        "hgb_mean_abs":    sr(hgb_mean_abs[i], 6),
        "rf_pct":          sr(rf_mean_abs[i]/total_rf*100,   2),
        "hgb_pct":         sr(hgb_mean_abs[i]/total_hgb*100, 2),
        "rf_gini_imp":     sr(rf.feature_importances_[i],    6),
    })

# group importance
group_sums = {}
for j in range(len(feat_cols)):
    g = group_label(feat_cols[j])
    group_sums.setdefault(g, {"rf":0.0,"hgb":0.0,"count":0})
    group_sums[g]["rf"]    += float(rf_mean_abs[j])
    group_sums[g]["hgb"]   += float(hgb_mean_abs[j])
    group_sums[g]["count"] += 1
group_imp = sorted([
    {"group":g,"rf_shap":sr(v["rf"],4),"hgb_shap":sr(v["hgb"],4),
     "rf_pct":sr(v["rf"]/total_rf*100,2),"hgb_pct":sr(v["hgb"]/total_hgb*100,2),
     "n_features":v["count"]}
    for g,v in group_sums.items()
], key=lambda x: x["rf_shap"], reverse=True)

# test-set waterfall rows (per prediction explanation)
test_rows = []
for i in range(len(X_te)):
    sv  = shap_te[i]
    idx_s = np.argsort(np.abs(sv))[::-1]
    test_rows.append({
        "period":      str(dates_te[i]),
        "actual":      sr(y_te[i], 2),
        "predicted":   sr(float(np.expm1(rf.predict(X_te[i:i+1])[0])), 2),
        "base_raw":    sr(float(np.expm1(rf_base)), 2),
        "shap_pos":    sr(float(np.sum(sv[sv>0])), 4),
        "shap_neg":    sr(float(np.sum(sv[sv<0])), 4),
        "top_contributors": [
            {"feature": feat_cols[j],
             "shap":    sr(float(sv[j]),4),
             "fval":    sr(float(X_te[i,j]),2)}
            for j in idx_s[:12]
        ],
    })

# dependence data — top 6 features
dep = {}
for i in order[:6]:
    fn = feat_cols[i]
    dep[fn] = {
        "feature_values": [sr(float(X_all[r,i]),2) for r in range(len(X_all))],
        "shap_values":    [sr(float(shap_all_vals[r,i]),4) for r in range(len(X_all))],
        "periods":        list(dates),
        "split":          ["train" if r<len(X_tr) else "test" for r in range(len(X_all))],
    }

# beeswarm matrix (all samples × top 15 features)
top15_idx = [int(i) for i in order[:15]]
beeswarm  = {
    "features":     [feat_cols[i] for i in top15_idx],
    "shap_matrix":  [[sr(float(shap_all_vals[r,i]),4) for i in top15_idx] for r in range(len(shap_all_vals))],
    "value_matrix": [[sr(float(X_all[r,i]),2)         for i in top15_idx] for r in range(len(X_all))],
    "periods":      list(dates),
    "split":        ["train" if r<len(X_tr) else "test" for r in range(len(X_all))],
}

# SHAP vs gini comparison for all features
shap_vs_gini = [
    {"feature":feat_cols[i], "shap":sr(rf_mean_abs[i],6), "gini":sr(float(rf.feature_importances_[i]),6)}
    for i in range(len(feat_cols))
]

shap_data = {
    "meta": {
        "model":          "RandomForest (TreeExplainer — exact SHAP)",
        "hgb_model":      "HistGBR (KernelExplainer)",
        "base_value_raw": sr(float(np.expm1(rf_base)), 2),
        "base_value_log": sr(rf_base, 4),
        "hgb_base_log":   sr(hgb_base, 4),
        "n_features":     len(feat_cols),
        "n_train":        int(len(X_tr)),
        "n_test":         int(len(X_te)),
    },
    "global_importance": global_imp,
    "group_importance":  group_imp,
    "test_predictions":  test_rows,
    "dependence":        dep,
    "beeswarm":          beeswarm,
    "shap_vs_gini":      shap_vs_gini,
}

with open(JSON_OUT,"w") as f:
    json.dump(shap_data, f, indent=2)

print("\n=== SHAP RESULTS ===")
print("Base value (avg prediction): $" + str(round(float(np.expm1(rf_base)),0)))
print("Top 5 features by mean |SHAP|:")
for r in global_imp[:5]:
    print("  %-24s RF: %.4f (%s%%)  HGB: %.4f (%s%%)" % (
        r["feature"], r["rf_mean_abs"], r["rf_pct"],
        r["hgb_mean_abs"], r["hgb_pct"]))
print("Feature group breakdown:")
for g in group_imp:
    print("  %-22s %s%% of total importance" % (g["group"], g["rf_pct"]))
print("\nshap_data.json saved.")