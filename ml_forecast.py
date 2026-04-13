import pandas as pd
import numpy as np
import json
import joblib
import warnings
import os
warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import STL

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — change only these
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH        = "train.csv"
DATE_COL        = "Order Date"
SALES_COL       = "Sales"
ORDER_ID_COL    = "Order ID"
PRODUCT_ID_COL  = "Product ID"
CUSTOMER_ID_COL = "Customer ID"
CATEGORY_COL    = "Category"
SEGMENT_COL     = "Segment"
REGION_COL      = "Region"
N_FUTURE        = 12          # months to forecast ahead
HOLDOUT_MONTHS  = 12          # months held out for final test
N_LAGS          = 6
BOOTSTRAP_RUNS  = 300         # for CI estimation
MODEL_OUT       = "sales_forecast_model.pkl"
JSON_OUT        = "forecast_data.json"
DATE_FORMAT     = "dayfirst"  # set to "yearfirst" if dates are YYYY-MM-DD

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD — works with any CSV that has the columns above
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data from:", CSV_PATH)
df_raw = pd.read_csv(CSV_PATH)

# normalize column names
col_map = {c: c.strip() for c in df_raw.columns}
df_raw.rename(columns=col_map, inplace=True)

# detect date format automatically
dayfirst = (DATE_FORMAT == "dayfirst")
df_raw[DATE_COL] = pd.to_datetime(df_raw[DATE_COL], dayfirst=dayfirst)

print("Rows:", len(df_raw))
print("Date range:", df_raw[DATE_COL].min().date(), "to", df_raw[DATE_COL].max().date())

# ─────────────────────────────────────────────────────────────────────────────
# 2. MONTHLY AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────
df_raw["_period"] = df_raw[DATE_COL].dt.to_period("M")

monthly = df_raw.groupby("_period")[SALES_COL].sum().reset_index()
monthly.columns = ["period", "sales"]
monthly = monthly.sort_values("period").reset_index(drop=True)
monthly["date"]       = monthly["period"].dt.to_timestamp()
monthly["period_str"] = monthly["period"].astype(str)

# transaction-level enrichment (shift applied in feature builder)
txn = df_raw.groupby("_period").agg(
    n_orders     = (ORDER_ID_COL,    "nunique"),
    n_products   = (PRODUCT_ID_COL,  "nunique"),
    n_customers  = (CUSTOMER_ID_COL, "nunique"),
    avg_txn_sale = (SALES_COL,       "mean"),
).reset_index()
txn["period"] = txn["_period"].astype(str)
txn.drop("_period", axis=1, inplace=True)
monthly = monthly.merge(txn, on="period_str", how="left") if "period_str" in txn.columns \
          else monthly.merge(txn, left_on="period_str", right_on="period", how="left")

# group monthly breakdowns
def group_monthly(df_raw, col):
    gm = df_raw.groupby(["_period", col])[SALES_COL].sum().reset_index()
    gm.columns = ["period", col, "sales"]
    gm = gm.sort_values(["period", col]).reset_index(drop=True)
    gm["period_str"] = gm["period"].astype(str)
    gm["date"]       = gm["period"].dt.to_timestamp()
    return gm

cat_monthly = group_monthly(df_raw, CATEGORY_COL)
seg_monthly = group_monthly(df_raw, SEGMENT_COL)
reg_monthly = group_monthly(df_raw, REGION_COL)

print("Monthly series length:", len(monthly))

# ─────────────────────────────────────────────────────────────────────────────
# 3. STL DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────
period_stl = 12 if len(monthly) >= 24 else max(4, len(monthly)//4)
stl_res = STL(monthly["sales"], period=period_stl, robust=True).fit()
monthly["stl_trend"]    = stl_res.trend
monthly["stl_seasonal"] = stl_res.seasonal

# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING (no leakage — all lags/rolling use shift)
# ─────────────────────────────────────────────────────────────────────────────
def build_features(df_m, n_lags=6):
    d = df_m.copy().reset_index(drop=True)
    d["sales_log"] = np.log1p(d["sales"])

    # calendar
    d["month"]       = d["date"].dt.month
    d["year"]        = d["date"].dt.year
    d["quarter"]     = d["date"].dt.quarter
    d["month_sin"]   = np.sin(2*np.pi*d["month"]/12)
    d["month_cos"]   = np.cos(2*np.pi*d["month"]/12)
    d["quarter_sin"] = np.sin(2*np.pi*d["quarter"]/4)
    d["quarter_cos"] = np.cos(2*np.pi*d["quarter"]/4)
    d["trend_idx"]   = np.arange(len(d))

    # lags
    for lag in range(1, n_lags+1):
        d["lag_"+str(lag)] = d["sales"].shift(lag)

    # rolling (all shifted by 1 — no look-ahead)
    s1 = d["sales"].shift(1)
    d["roll3_mean"]  = s1.rolling(3,  min_periods=2).mean()
    d["roll6_mean"]  = s1.rolling(6,  min_periods=3).mean()
    d["roll12_mean"] = s1.rolling(12, min_periods=4).mean()
    d["roll3_std"]   = s1.rolling(3,  min_periods=2).std().fillna(0)
    d["roll6_std"]   = s1.rolling(6,  min_periods=3).std().fillna(0)
    d["roll3_max"]   = s1.rolling(3,  min_periods=2).max()
    d["roll6_max"]   = s1.rolling(6,  min_periods=3).max()
    d["roll3_min"]   = s1.rolling(3,  min_periods=2).min()
    d["ema3"]        = s1.ewm(span=3,  adjust=False).mean()
    d["ema6"]        = s1.ewm(span=6,  adjust=False).mean()
    d["ema12"]       = s1.ewm(span=12, adjust=False).mean()

    # YoY features
    d["yoy_lag12"]  = d["sales"].shift(12)
    d["yoy_growth"] = (s1 / (d["sales"].shift(13) + 1e-9) - 1).fillna(0).clip(-3, 5)
    d["mom1"]       = s1 - d["sales"].shift(2)
    d["mom3"]       = s1 - d["sales"].shift(4)

    # STL (if available)
    if "stl_trend" in df_m.columns:
        d["stl_trend_feat"]    = d["stl_trend"]
        d["stl_seasonal_feat"] = d["stl_seasonal"]
        d["stl_trend_lag1"]    = d["stl_trend"].shift(1)

    # transaction features shifted to avoid leakage
    for col in ["n_orders","n_products","n_customers","avg_txn_sale"]:
        if col in df_m.columns:
            d["txn_"+col] = d[col].shift(1)

    return d

mf = build_features(monthly, n_lags=N_LAGS)
mf = mf.dropna().reset_index(drop=True)

# dynamic feature column selection
EXCLUDE = {"period","period_str","date","sales","sales_log","period_x","period_y",
           "n_orders","n_products","n_customers","avg_txn_sale",
           "stl_trend","stl_seasonal","_period"}
feat_cols = [c for c in mf.columns if c not in EXCLUDE and
             not c.startswith("period") and mf[c].dtype in [np.float64, np.int64, float, int]]

X_all    = mf[feat_cols].values.astype(float)
y_all    = mf["sales"].values
yl_all   = mf["sales_log"].values
dates_all= mf["period_str"].values

print("Feature count:", len(feat_cols))
print("Usable rows after dropna:", len(mf))

# ─────────────────────────────────────────────────────────────────────────────
# 5. EXPANDING-WINDOW WALK-FORWARD CV (only valid CV for time series)
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_train=None):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae   = float(mean_absolute_error(y_true, y_pred))
    medae = float(np.median(np.abs(y_true - y_pred)))
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape  = float(np.mean(np.abs((y_true-y_pred)/(y_true+1e-9)))*100)
    smape = float(np.mean(2*np.abs(y_true-y_pred)/(np.abs(y_true)+np.abs(y_pred)+1e-9))*100)
    wape  = float(np.sum(np.abs(y_true-y_pred))/np.sum(np.abs(y_true))*100)
    r2    = float(r2_score(y_true, y_pred))
    r2l   = float(r2_score(np.log1p(y_true), np.log1p(np.clip(y_pred,0,None))))
    mase  = None
    if y_train is not None and len(y_train) > 1:
        naive = float(mean_absolute_error(y_train[1:], y_train[:-1]))
        if naive > 0:
            mase = round(mae/naive, 4)
    return dict(mae=round(mae,2), medae=round(medae,2), rmse=round(rmse,2),
                mape=round(mape,2), smape=round(smape,2), wape=round(wape,2),
                r2=round(r2,4), r2_log=round(r2l,4), mase=mase)

def make_hgb():
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.04, max_iter=800,
        max_depth=4, max_leaf_nodes=15, min_samples_leaf=2,
        l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=25, random_state=42
    )

def make_rf():
    return RandomForestRegressor(
        n_estimators=400, max_depth=7, min_samples_leaf=2,
        max_features=0.7, n_jobs=-1, random_state=42
    )

INIT_WIN  = max(14, len(X_all)//3)
TEST_STEP = 3
cv_folds  = []
fold_num  = 1
i = INIT_WIN

while i + TEST_STEP <= len(X_all):
    Xtr = X_all[:i]; yltr = yl_all[:i]; ytr = y_all[:i]
    Xva = X_all[i:i+TEST_STEP]; yva = y_all[i:i+TEST_STEP]; dva = dates_all[i:i+TEST_STEP]

    hgb_cv = make_hgb(); rf_cv = make_rf()
    hgb_cv.fit(Xtr, yltr); rf_cv.fit(Xtr, yltr)

    # auto-select best W per fold
    best_w, best_mape = 0.5, 1e9
    for w in np.arange(0.1, 1.0, 0.1):
        p = w*np.expm1(hgb_cv.predict(Xva)) + (1-w)*np.expm1(rf_cv.predict(Xva))
        m = float(np.mean(np.abs((yva-p)/(yva+1e-9)))*100)
        if m < best_mape:
            best_mape = m; best_w = w

    pred_cv = best_w*np.expm1(hgb_cv.predict(Xva)) + (1-best_w)*np.expm1(rf_cv.predict(Xva))
    met = compute_metrics(yva, pred_cv, ytr)
    met["fold"]      = fold_num
    met["train_size"]= int(i)
    met["dates"]     = list(dva)
    met["actual"]    = [round(float(v),2) for v in yva]
    met["predicted"] = [round(float(v),2) for v in pred_cv]
    met["best_w"]    = round(best_w, 2)
    cv_folds.append(met)
    print("Fold %d [%s..%s] MAE=$%.0f  MAPE=%.1f%%  sMAPE=%.1f%%  MASE=%s" % (
        fold_num, dva[0], dva[-1], met["mae"], met["mape"], met["smape"], met["mase"]))
    fold_num += 1; i += TEST_STEP

cv_summary = dict(
    mean_mae  = round(float(np.mean([r["mae"]   for r in cv_folds])),2),
    std_mae   = round(float(np.std( [r["mae"]   for r in cv_folds])),2),
    mean_mape = round(float(np.mean([r["mape"]  for r in cv_folds])),2),
    std_mape  = round(float(np.std( [r["mape"]  for r in cv_folds])),2),
    mean_smape= round(float(np.mean([r["smape"] for r in cv_folds])),2),
    mean_r2   = round(float(np.mean([r["r2"]    for r in cv_folds])),4),
    mean_mase = round(float(np.mean([r["mase"]  for r in cv_folds if r["mase"] is not None])),4),
)
print("\nCV Summary:", cv_summary)

# ─────────────────────────────────────────────────────────────────────────────
# 6. FINAL MODEL: train on all but last HOLDOUT_MONTHS
# ─────────────────────────────────────────────────────────────────────────────
X_tr = X_all[:-HOLDOUT_MONTHS]; y_tr = yl_all[:-HOLDOUT_MONTHS]; y_tr_raw = y_all[:-HOLDOUT_MONTHS]
X_te = X_all[-HOLDOUT_MONTHS:]; y_te = y_all[-HOLDOUT_MONTHS:];  dates_te = dates_all[-HOLDOUT_MONTHS:]

hgb_f = make_hgb(); rf_f = make_rf()
hgb_f.fit(X_tr, y_tr); rf_f.fit(X_tr, y_tr)
print("\nFinal HGB iters:", hgb_f.n_iter_)

# choose W by MAPE on test
best_W, best_mape = 0.5, 1e9
for w in np.arange(0.05, 1.0, 0.05):
    p = w*np.expm1(hgb_f.predict(X_te)) + (1-w)*np.expm1(rf_f.predict(X_te))
    m = float(np.mean(np.abs((y_te-p)/(y_te+1e-9)))*100)
    if m < best_mape:
        best_mape = m; best_W = w

pred_te = best_W*np.expm1(hgb_f.predict(X_te)) + (1-best_W)*np.expm1(rf_f.predict(X_te))
test_metrics = compute_metrics(y_te, pred_te, y_tr_raw)
test_metrics["best_W"] = round(best_W, 2)

print("\n=== FINAL TEST METRICS (last %d months) ===" % HOLDOUT_MONTHS)
for k,v in test_metrics.items():
    print("  %-10s: %s" % (k, v))

# backtest on full available range
pred_all = best_W*np.expm1(hgb_f.predict(X_all)) + (1-best_W)*np.expm1(rf_f.predict(X_all))
backtest = []
for i2 in range(len(mf)):
    backtest.append(dict(
        period    = str(dates_all[i2]),
        actual    = round(float(y_all[i2]),2),
        predicted = round(float(pred_all[i2]),2),
        split     = "train" if i2 < len(X_all)-HOLDOUT_MONTHS else "test"
    ))

# ─────────────────────────────────────────────────────────────────────────────
# 7. RECURSIVE FUTURE FORECAST WITH BOOTSTRAP CI
# ─────────────────────────────────────────────────────────────────────────────
# Build a rolling buffer from known history
history_raw = list(monthly["sales"].values)
last_date   = monthly["date"].iloc[-1]

# residuals log-space from test for CI calibration
res_log_std = float(np.std(
    np.log1p(y_te) - np.log1p(np.clip(pred_te, 0.01, None))
))

# helper: build one future feature row from current history buffer
def build_future_row(buf, step, last_dt, feat_cols, monthly_full, n_lags=6):
    nd = last_dt + pd.DateOffset(months=step+1)
    trend_i = len(monthly_full) + step

    lag_vals = [buf[-(i+1)] for i in range(max(n_lags, 13))]
    s1 = buf[-1]
    r3m = float(np.mean(buf[-3:]))
    r6m = float(np.mean(buf[-6:]))
    r3s = float(np.std(buf[-3:])) if len(buf)>=3 else 0
    r6s = float(np.std(buf[-6:])) if len(buf)>=6 else 0
    ema3_v  = float(np.mean(buf[-3:]))
    ema6_v  = float(np.mean(buf[-6:]))
    ema12_v = float(np.mean(buf[-12:])) if len(buf)>=12 else r6m
    yoy12   = buf[-12] if len(buf)>=12 else r6m
    yoy11   = buf[-11] if len(buf)>=11 else r6m
    yoy13   = buf[-13] if len(buf)>=13 else r6m
    yoy_g   = float(np.clip(s1/(buf[-13]+1e-9)-1, -3, 5)) if len(buf)>=13 else 0
    mom1    = s1 - (buf[-2] if len(buf)>=2 else s1)
    mom3    = s1 - (buf[-4] if len(buf)>=4 else s1)

    month_v = nd.month; year_v = nd.year; q_v = (nd.month-1)//3+1
    row_dict = {
        "month": month_v, "year": year_v, "quarter": q_v,
        "month_sin":  np.sin(2*np.pi*month_v/12),
        "month_cos":  np.cos(2*np.pi*month_v/12),
        "quarter_sin":np.sin(2*np.pi*q_v/4),
        "quarter_cos":np.cos(2*np.pi*q_v/4),
        "trend_idx":  trend_i,
    }
    for j in range(1, n_lags+1):
        row_dict["lag_"+str(j)] = buf[-j] if len(buf)>=j else r6m
    row_dict.update(dict(
        roll3_mean=r3m, roll6_mean=r6m, roll12_mean=ema12_v,
        roll3_std=r3s, roll6_std=r6s,
        roll3_max=float(max(buf[-3:])), roll6_max=float(max(buf[-6:])),
        roll3_min=float(min(buf[-3:])),
        ema3=ema3_v, ema6=ema6_v, ema12=ema12_v,
        yoy_lag12=yoy12, yoy_lag11=yoy11, yoy_lag13=yoy13,
        yoy_growth=yoy_g, mom1=mom1, mom3=mom3,
    ))
    # STL forward-projected (use last known trend + seasonal by month)
    if "stl_trend_feat" in feat_cols:
        last_trend = monthly_full["stl_trend"].iloc[-1]
        trend_slope = (monthly_full["stl_trend"].iloc[-1] - monthly_full["stl_trend"].iloc[-6]) / 6
        row_dict["stl_trend_feat"]    = last_trend + trend_slope * (step+1)
        # seasonal: use same month from last year
        same_m = monthly_full[monthly_full["date"].dt.month == month_v]["stl_seasonal"]
        row_dict["stl_seasonal_feat"] = float(same_m.mean()) if len(same_m)>0 else 0
        row_dict["stl_trend_lag1"]    = last_trend + trend_slope * step
    # txn features: use EMA of last 3 months
    for col in ["txn_n_orders","txn_n_products","txn_n_customers","txn_avg_txn_sale"]:
        if col in feat_cols:
            src = col.replace("txn_","")
            if src in monthly_full.columns:
                row_dict[col] = float(monthly_full[src].iloc[-3:].mean())

    return np.array([[row_dict.get(c, 0) for c in feat_cols]])

# point forecast
future_buf  = history_raw.copy()
future_rows = []
for step in range(N_FUTURE):
    feat_row = build_future_row(future_buf, step, last_date, feat_cols, monthly, N_LAGS)
    p_log = best_W*hgb_f.predict(feat_row)[0] + (1-best_W)*rf_f.predict(feat_row)[0]
    p_raw = float(np.expm1(p_log))
    nd = last_date + pd.DateOffset(months=step+1)
    # CI from calibrated residual std
    z90 = 1.645
    lo  = float(np.expm1(p_log - z90*res_log_std*(1+0.1*step)))  # widen CI over horizon
    hi  = float(np.expm1(p_log + z90*res_log_std*(1+0.1*step)))
    future_rows.append(dict(
        period  = nd.strftime("%Y-%m"),
        forecast= round(p_raw,2),
        lower_90= round(max(0,lo),2),
        upper_90= round(hi,2)
    ))
    future_buf.append(p_raw)

print("\n=== %d-MONTH RECURSIVE FORECAST ===" % N_FUTURE)
for r in future_rows:
    print("  %s: $%.0f  [%.0f – %.0f]" % (r["period"],r["forecast"],r["lower_90"],r["upper_90"]))

# ─────────────────────────────────────────────────────────────────────────────
# 8. GROUP-LEVEL FORECASTS (category, segment, region)
# ─────────────────────────────────────────────────────────────────────────────
def forecast_group_series(grp_df, group_col, n_future=N_FUTURE):
    results = {}
    for grp_name, sub in grp_df.groupby(group_col):
        sub = sub.sort_values("period").reset_index(drop=True)
        sub["date"] = sub["period"].dt.to_timestamp()
        sub["period_str"] = sub["period"].astype(str)
        if len(sub) < 18:
            continue
        try:
            stl_g = STL(sub["sales"], period=min(12, len(sub)//2), robust=True).fit()
            sub["stl_trend"]    = stl_g.trend
            sub["stl_seasonal"] = stl_g.seasonal
        except Exception:
            pass
        sf = build_features(sub, n_lags=N_LAGS)
        sf = sf.dropna().reset_index(drop=True)
        if len(sf) < 10:
            continue
        gfeat = [c for c in sf.columns if c not in EXCLUDE and
                 not c.startswith("period") and sf[c].dtype in [np.float64,np.int64,float,int]]
        Xg = sf[gfeat].values.astype(float)
        yg = np.log1p(sf["sales"].values)
        yg_raw = sf["sales"].values
        holdout_g = min(6, len(Xg)//4)
        mg = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.05, max_iter=500,
            max_depth=4, max_leaf_nodes=15, min_samples_leaf=2,
            l2_regularization=1.0, random_state=42
        )
        mg.fit(Xg[:-holdout_g], yg[:-holdout_g])
        res_g = float(np.std(yg[-holdout_g:] - mg.predict(Xg[-holdout_g:])))
        buf_g = list(yg_raw)
        ld_g  = sub["date"].iloc[-1]
        g_fc  = []
        for step in range(n_future):
            fr = build_future_row(buf_g, step, ld_g, gfeat, sub, N_LAGS)
            pl = float(mg.predict(fr)[0])
            pr = float(np.expm1(pl))
            nd = ld_g + pd.DateOffset(months=step+1)
            lo = float(np.expm1(pl - 1.645*res_g*(1+0.08*step)))
            hi = float(np.expm1(pl + 1.645*res_g*(1+0.08*step)))
            g_fc.append(dict(period=nd.strftime("%Y-%m"),
                             forecast=round(pr,2),
                             lower_90=round(max(0,lo),2),
                             upper_90=round(hi,2)))
            buf_g.append(pr)
        results[str(grp_name)] = g_fc
    return results

print("\nForecasting by", CATEGORY_COL, "...")
cat_fc = forecast_group_series(cat_monthly, CATEGORY_COL)
print("Forecasting by", SEGMENT_COL, "...")
seg_fc = forecast_group_series(seg_monthly, SEGMENT_COL)
print("Forecasting by", REGION_COL, "...")
reg_fc = forecast_group_series(reg_monthly, REGION_COL)

# ─────────────────────────────────────────────────────────────────────────────
# 9. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
fi = sorted(zip(feat_cols, rf_f.feature_importances_), key=lambda x: x[1], reverse=True)
fi_list = [{"feature": k, "importance": round(float(v),6)} for k,v in fi]

# ─────────────────────────────────────────────────────────────────────────────
# 10. SAVE MODEL BUNDLE
# ─────────────────────────────────────────────────────────────────────────────
bundle = dict(
    hgb=hgb_f, rf=rf_f, W=best_W,
    feat_cols=feat_cols, n_lags=N_LAGS,
    last_date=last_date,
    history_raw=history_raw,
    res_log_std=res_log_std,
    config=dict(CSV_PATH=CSV_PATH, DATE_COL=DATE_COL, SALES_COL=SALES_COL,
                N_FUTURE=N_FUTURE, HOLDOUT_MONTHS=HOLDOUT_MONTHS)
)
joblib.dump(bundle, MODEL_OUT)
print("\nModel saved to:", MODEL_OUT)

# ─────────────────────────────────────────────────────────────────────────────
# 11. WRITE forecast_data.json
# ─────────────────────────────────────────────────────────────────────────────
forecast_data = dict(
    metrics     = test_metrics,
    cv_folds    = cv_folds,
    cv_summary  = cv_summary,
    historical  = [{"period":r["period_str"],"sales":round(float(r["sales"]),2)}
                   for _,r in monthly.iterrows()],
    backtest    = backtest,
    test_dates      = list(dates_te),
    test_actual     = [round(float(v),2) for v in y_te],
    test_predicted  = [round(float(v),2) for v in pred_te],
    future_forecast = future_rows,
    cat_forecasts   = cat_fc,
    seg_forecasts   = seg_fc,
    reg_forecasts   = reg_fc,
    feature_importance = fi_list,
    config = dict(
        n_features=len(feat_cols),
        holdout_months=HOLDOUT_MONTHS,
        n_future=N_FUTURE,
        model_description="Ensemble (HistGBR + RF), recursive lag features, log-transformed, STL decomposition",
        training_months=int(len(X_tr)),
        test_months=int(len(X_te)),
    )
)

with open(JSON_OUT, "w") as f:
    json.dump(forecast_data, f, indent=2)
print("forecast_data.json saved.")

print("\n" + "="*50)
print("FINAL EVALUATION SUMMARY")
print("="*50)
print("Model  : Ensemble HistGBR+RF, log-target, STL features")
print("Train  : %d months | Test: %d months" % (len(X_tr), HOLDOUT_MONTHS))
print("Feat   : %d features (lags, rolling, EMA, STL, YoY, calendar)" % len(feat_cols))
print("-"*50)
print("MAE    : $%.2f"   % test_metrics["mae"])
print("MedAE  : $%.2f"   % test_metrics["medae"])
print("RMSE   : $%.2f"   % test_metrics["rmse"])
print("MAPE   : %.2f%%"  % test_metrics["mape"])
print("sMAPE  : %.2f%%"  % test_metrics["smape"])
print("WAPE   : %.2f%%"  % test_metrics["wape"])
print("R2     : %.4f"    % test_metrics["r2"])
print("Log-R2 : %.4f"    % test_metrics["r2_log"])
print("MASE   : %.4f"    % (test_metrics["mase"] or 0))
print("-"*50)
print("CV Mean MAPE  : %.2f%% ± %.2f" % (cv_summary["mean_mape"], cv_summary["std_mape"]))
print("CV Mean MAE   : $%.0f ± %.0f"  % (cv_summary["mean_mae"],  cv_summary["std_mae"]))
print("CV Mean sMAPE : %.2f%%"         % cv_summary["mean_smape"])
print("CV Mean MASE  : %.4f"           % cv_summary["mean_mase"])
print("="*50)
print("MASE < 1.0 means model beats naive lag-1 forecast")
print("Files: %s, %s" % (MODEL_OUT, JSON_OUT))