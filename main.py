import warnings
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict
import asyncio
import json
import time

import numpy as np
import pandas as pd
import xlsxwriter
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Forecasting imports (for Predict app)
from catboost import CatBoostRegressor
from darts import TimeSeries
from darts.models import NBEATSModel
from lightgbm import LGBMRegressor
from prophet import Prophet
from pytorch_lightning.callbacks import EarlyStopping
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Automation imports (for Pour app)
import pyautogui
import pyperclip
import pygetwindow as gw

warnings.filterwarnings("ignore")

# Optional models
try:
    from neuralprophet import NeuralProphet
    _HAS_NEURALPROPHET = True
except ImportError:
    _HAS_NEURALPROPHET = False

try:
    from xgboost import XGBRegressor
    import sklearn
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

# Torch 2.6 safe-load fix
_safe_globals_ctx = None
_NP_SAFE_GLOBALS = []
if _HAS_NEURALPROPHET:
    try:
        import torch.serialization as _ts
        from torch.serialization import safe_globals as _safe_globals_ctx
        from neuralprophet.configure import (
            ConfigAR, ConfigCountryHolidays, ConfigCustomSeasonality,
            ConfigEvents, ConfigLaggedRegressor, ConfigSeasonality, ConfigTrend,
        )
        _NP_SAFE_GLOBALS = [
            ConfigSeasonality, ConfigEvents, ConfigTrend, ConfigCountryHolidays,
            ConfigAR, ConfigLaggedRegressor, ConfigCustomSeasonality,
        ]
        _ts.add_safe_globals(_NP_SAFE_GLOBALS)
    except Exception:
        _safe_globals_ctx = None
        _NP_SAFE_GLOBALS = []

# Constants
WEEK_FREQ = "W-SAT"
PI_Z_90 = 1.645

# Create FastAPI app
app = FastAPI(title="Predict & Pour System", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# Pydantic Models
# =====================================================================

class ForecastConfig(BaseModel):
    horizon_weeks: int = 12
    min_points: int = 50
    show_debug: bool = False
    run_prophet: bool = True
    run_neural: bool = False
    run_sarimax: bool = True
    run_xgb: bool = False
    run_catboost: bool = False
    run_holt: bool = False
    run_lgbm: bool = False
    run_nbeats: bool = False

class CoordinateCalibration(BaseModel):
    pdcn_x: Optional[int] = None
    pdcn_y: Optional[int] = None
    week1_x: Optional[int] = None
    week1_y: Optional[int] = None

class ExecuteImportRequest(BaseModel):
    pdcn: str
    week1_x: int
    week1_y: int

# =====================================================================
# Global State (for Pour app)
# =====================================================================
calibration_data = {
    "pdcn_coords": None,
    "week1_coords": None
}
uploaded_pour_data = None

# =====================================================================
# Helper Functions (ALL FROM PREDICT APP - UNCHANGED)
# =====================================================================

def get_custom_holidays() -> pd.DataFrame:
    holiday_week_dates = [
        ("fiesta", pd.date_range("2023-04-20", "2023-04-30")),
        ("fiesta", pd.date_range("2024-04-18", "2024-04-28")),
        ("fiesta", pd.date_range("2025-04-21", "2025-05-04")),
        ("rodeo", pd.date_range("2023-02-09", "2023-02-26")),
        ("rodeo", pd.date_range("2024-02-08", "2024-02-25")),
        ("rodeo", pd.date_range("2025-02-12", "2025-03-01")),
        ("christmas", ["2023-12-25", "2024-12-25", "2025-12-25"]),
        ("christmas_eve", ["2023-12-24", "2024-12-24", "2025-12-24"]),
        ("memorial_day", ["2023-05-29", "2024-05-27", "2025-05-26"]),
        ("cinco_de_mayo", ["2023-05-05", "2024-05-05", "2025-05-05"]),
        ("fourth_of_july", ["2023-07-04", "2024-07-04", "2025-07-04"]),
        ("labor_day", ["2023-09-04", "2024-09-02", "2025-09-01"]),
        ("thanksgiving", ["2023-11-23", "2024-11-28", "2025-11-27"]),
    ]
    records = []
    for name, dates in holiday_week_dates:
        for date in pd.to_datetime(dates):
            records.append({
                "holiday": str(name),
                "ds": pd.Timestamp(date).normalize(),
                "lower_window": -3,
                "upper_window": 3,
            })
    return pd.DataFrame(records)

def _add_event_flags(df_dates: pd.DataFrame, holidays_df: pd.DataFrame, event_names=None) -> pd.DataFrame:
    if event_names is None:
        event_names = holidays_df["holiday"].astype(str).unique().tolist()
    df_dates = df_dates.copy()
    for h in event_names:
        mask = df_dates["ds"].isin(holidays_df.loc[holidays_df["holiday"] == h, "ds"])
        df_dates[h] = mask.astype(int)
    return df_dates

def safe_accuracy(y, yhat) -> float:
    if pd.isna(y) or y == 0 or pd.isna(yhat):
        return np.nan
    return round(1 - abs((y - yhat) / y), 3)

def next_saturday_from(date: pd.Timestamp) -> pd.Timestamp:
    days_ahead = (5 - date.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return date + pd.Timedelta(days=days_ahead)

def build_future_saturdays(last_date: pd.Timestamp, horizon_weeks: int) -> pd.DatetimeIndex:
    first_sat = next_saturday_from(last_date)
    return pd.date_range(start=first_sat, periods=horizon_weeks, freq=WEEK_FREQ)

# =====================================================================
# Model Functions (ALL UNCHANGED from Streamlit version)
# =====================================================================

def run_prophet(df_sku: pd.DataFrame, holidays_df: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        holidays=holidays_df,
        interval_width=0.90,
    )
    m.fit(df_sku[["ds", "y"]])
    last_date = df_sku["ds"].max()
    future_sats = build_future_saturdays(last_date, horizon_weeks)
    future = pd.concat(
        [df_sku[["ds"]], pd.DataFrame({"ds": future_sats})], ignore_index=True
    ).drop_duplicates("ds")
    fcst = m.predict(future)
    cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
    extra_cols = [c for c in ["trend", "weekly", "yearly", "holidays"] if c in fcst.columns]
    cols.extend(extra_cols)
    out = fcst[cols].copy()
    out.rename(
        columns={
            "yhat": "prophet_yhat",
            "yhat_lower": "prophet_yhat_lower",
            "yhat_upper": "prophet_yhat_upper",
        },
        inplace=True,
    )
    return out

def run_neuralprophet(df_sku: pd.DataFrame, holidays_df: pd.DataFrame, horizon_weeks: int, debug: bool = False) -> pd.DataFrame:
    if not _HAS_NEURALPROPHET:
        raise ImportError("neuralprophet is not installed")
    event_names = holidays_df["holiday"].astype(str).unique().tolist()
    m = NeuralProphet(weekly_seasonality=True, yearly_seasonality=True, quantiles=[0.05, 0.95])
    for ev in event_names:
        m.add_events(ev, lower_window=-3, upper_window=3)
    train_df = df_sku[["ds", "y"]].copy()
    train_df = _add_event_flags(train_df, holidays_df, event_names)
    missing_train = [ev for ev in event_names if ev not in train_df.columns]
    if missing_train:
        raise ValueError(f"Missing event cols in train_df: {missing_train}")
    def _fit_model():
        try:
            return m.fit(train_df, freq=WEEK_FREQ, trainer_config={"enable_checkpointing": False, "logger": False})
        except TypeError:
            return m.fit(train_df, freq=WEEK_FREQ)
    if "_safe_globals_ctx" in globals() and _safe_globals_ctx:
        with _safe_globals_ctx(_NP_SAFE_GLOBALS):
            _fit_model()
    else:
        _fit_model()
    last_date = df_sku["ds"].max()
    future_sats = build_future_saturdays(last_date, horizon_weeks)
    future = pd.DataFrame({"ds": future_sats})
    future = _add_event_flags(future, holidays_df, event_names)
    full_future = (
        pd.concat([train_df[["ds"] + event_names], future[["ds"] + event_names]], ignore_index=True)
        .drop_duplicates("ds").sort_values("ds")
    )
    if "y" not in full_future.columns:
        full_future["y"] = np.nan
    missing_future = [ev for ev in event_names if ev not in full_future.columns]
    if missing_future:
        raise ValueError(f"Missing event cols in full_future: {missing_future}")
    def _predict_model():
        return m.predict(full_future)
    if "_safe_globals_ctx" in globals() and _safe_globals_ctx:
        with _safe_globals_ctx(_NP_SAFE_GLOBALS):
            forecast = _predict_model()
    else:
        forecast = _predict_model()
    out = forecast[["ds", "yhat1"]].rename(columns={"yhat1": "neural_yhat"})
    out["neural_yhat_lower"] = forecast.get("yhat1 5.0%", np.nan)
    out["neural_yhat_upper"] = forecast.get("yhat1 95.0%", np.nan)
    return out

def run_sarimax(df_sku: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    ts = df_sku.set_index("ds")["y"].asfreq(WEEK_FREQ)
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 52)
    model = SARIMAX(ts, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    fcst = res.get_forecast(steps=horizon_weeks)
    pred_mean = fcst.predicted_mean
    conf = fcst.conf_int(alpha=0.10)
    conf_cols = [c.lower() for c in conf.columns]
    lower_col = conf.columns[0] if "lower" in conf_cols[0] else conf.columns[1]
    upper_col = conf.columns[1] if "upper" in conf_cols[1] else conf.columns[0]
    conf = conf.rename(columns={lower_col: "sarimax_yhat_lower", upper_col: "sarimax_yhat_upper"})
    future_index = pd.date_range(start=ts.index.max() + pd.Timedelta(weeks=1), periods=horizon_weeks, freq=WEEK_FREQ)
    hist_df = pd.DataFrame({"ds": ts.index, "sarimax_yhat": res.fittedvalues})
    fut_df = pd.DataFrame({"ds": future_index, "sarimax_yhat": pred_mean.values})
    fut_df = pd.concat([fut_df.reset_index(drop=True), conf.reset_index(drop=True)], axis=1)
    out = pd.concat([hist_df, fut_df], ignore_index=True)
    out["sarimax_yhat_lower"] = out.get("sarimax_yhat_lower", np.nan)
    out["sarimax_yhat_upper"] = out.get("sarimax_yhat_upper", np.nan)
    return out

def make_lag_features(y: pd.Series, lags=(1, 2, 3, 4, 5, 6, 7, 12, 26, 52), roll_windows=(4, 12)) -> pd.DataFrame:
    df_feat = pd.DataFrame({"y": y})
    for l in lags:
        df_feat[f"lag_{l}"] = y.shift(l)
    for w in roll_windows:
        df_feat[f"roll_mean_{w}"] = y.shift(1).rolling(window=w).mean()
    return df_feat

def add_date_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({
        "weekofyear": idx.isocalendar().week.astype(int),
        "month": idx.month,
        "quarter": idx.quarter,
        "year": idx.year,
    }, index=idx)

def run_xgboost(df_sku: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    if not _HAS_XGBOOST:
        raise ImportError("xgboost / scikit-learn not installed")
    ts = df_sku.set_index("ds")["y"].asfreq(WEEK_FREQ)
    features = make_lag_features(ts)
    date_feats = add_date_features(features.index)
    X = pd.concat([features.drop(columns=["y"]), date_feats], axis=1)
    y = features["y"]
    mask = X.notna().all(axis=1)
    X_train, y_train = X[mask], y[mask]
    model = XGBRegressor(
        n_estimators=600, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42,
    )
    model.fit(X_train, y_train)
    preds_in = model.predict(X_train)
    resid_std = np.std(y_train - preds_in)
    future_index = pd.date_range(start=ts.index.max() + pd.Timedelta(weeks=1), periods=horizon_weeks, freq=WEEK_FREQ)
    full_y = ts.copy()
    rows = []
    for ds in future_index:
        feats = make_lag_features(full_y).iloc[-1:].drop(columns=["y"])
        feats = feats.assign(**add_date_features(pd.DatetimeIndex([ds])).iloc[0].to_dict())
        pred = model.predict(feats)[0]
        rows.append({
            "ds": ds, "xgb_yhat": pred,
            "xgb_yhat_lower": pred - PI_Z_90 * resid_std,
            "xgb_yhat_upper": pred + PI_Z_90 * resid_std,
        })
        full_y.loc[ds] = pred
    hist_pred = model.predict(X)
    hist_df = pd.DataFrame({"ds": X.index, "xgb_yhat": hist_pred})
    hist_df["xgb_yhat_lower"] = np.nan
    hist_df["xgb_yhat_upper"] = np.nan
    fut_df = pd.DataFrame(rows)
    return pd.concat([hist_df, fut_df], ignore_index=True)

def run_catboost(df_sku: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    ts = df_sku.set_index("ds")["y"].asfreq(WEEK_FREQ)
    features = make_lag_features(ts)
    date_feats = add_date_features(features.index)
    X = pd.concat([features.drop(columns=["y"]), date_feats], axis=1)
    y = features["y"]
    mask = X.notna().all(axis=1)
    X_train, y_train = X[mask], y[mask]
    model = CatBoostRegressor(verbose=0)
    model.fit(X_train, y_train)
    preds_in = model.predict(X_train)
    resid_std = np.std(y_train - preds_in)
    future_index = pd.date_range(start=ts.index.max() + pd.Timedelta(weeks=1), periods=horizon_weeks, freq=WEEK_FREQ)
    full_y = ts.copy()
    rows = []
    for ds in future_index:
        feats = make_lag_features(full_y).iloc[-1:].drop(columns=["y"])
        feats = feats.assign(**add_date_features(pd.DatetimeIndex([ds])).iloc[0].to_dict())
        pred = model.predict(feats)[0]
        rows.append({
            "ds": ds, "catboost_yhat": pred,
            "catboost_yhat_lower": pred - PI_Z_90 * resid_std,
            "catboost_yhat_upper": pred + PI_Z_90 * resid_std,
        })
        full_y.loc[ds] = pred
    hist_pred = model.predict(X)
    hist_df = pd.DataFrame({"ds": X.index, "catboost_yhat": hist_pred})
    hist_df["catboost_yhat_lower"] = np.nan
    hist_df["catboost_yhat_upper"] = np.nan
    fut_df = pd.DataFrame(rows)
    return pd.concat([hist_df, fut_df], ignore_index=True)

def run_holtwinters(df_sku: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    ts = df_sku.set_index("ds")["y"].asfreq(WEEK_FREQ)
    model = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=52).fit(optimized=True)
    hist_df = pd.DataFrame({"ds": ts.index, "holt_yhat": model.fittedvalues})
    hist_df["holt_yhat_lower"] = np.nan
    hist_df["holt_yhat_upper"] = np.nan
    future_index = pd.date_range(start=ts.index.max() + pd.Timedelta(weeks=1), periods=horizon_weeks, freq=WEEK_FREQ)
    forecast = model.forecast(horizon_weeks)
    resid_std = np.std(model.resid)
    fut_df = pd.DataFrame({
        "ds": future_index, "holt_yhat": forecast.values,
        "holt_yhat_lower": forecast.values - PI_Z_90 * resid_std,
        "holt_yhat_upper": forecast.values + PI_Z_90 * resid_std,
    })
    return pd.concat([hist_df, fut_df], ignore_index=True)

def run_lightgbm(df_sku: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    ts = df_sku.set_index("ds")["y"].asfreq(WEEK_FREQ)
    features = make_lag_features(ts)
    date_feats = add_date_features(features.index)
    X = pd.concat([features.drop(columns=["y"]), date_feats], axis=1)
    y = features["y"]
    mask = X.notna().all(axis=1)
    X_train, y_train = X[mask], y[mask]
    model = LGBMRegressor(
        n_estimators=600, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    preds_in = model.predict(X_train)
    resid_std = np.std(y_train - preds_in)
    future_index = pd.date_range(start=ts.index.max() + pd.Timedelta(weeks=1), periods=horizon_weeks, freq=WEEK_FREQ)
    full_y = ts.copy()
    rows = []
    for ds in future_index:
        feats = make_lag_features(full_y).iloc[-1:].drop(columns=["y"])
        feats = feats.assign(**add_date_features(pd.DatetimeIndex([ds])).iloc[0].to_dict())
        pred = model.predict(feats)[0]
        rows.append({
            "ds": ds, "lgbm_yhat": pred,
            "lgbm_yhat_lower": pred - PI_Z_90 * resid_std,
            "lgbm_yhat_upper": pred + PI_Z_90 * resid_std,
        })
        full_y.loc[ds] = pred
    hist_pred = model.predict(X)
    hist_df = pd.DataFrame({"ds": X.index, "lgbm_yhat": hist_pred})
    hist_df["lgbm_yhat_lower"] = np.nan
    hist_df["lgbm_yhat_upper"] = np.nan
    fut_df = pd.DataFrame(rows)
    return pd.concat([hist_df, fut_df], ignore_index=True)

def run_nbeats(df_sku: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    series = TimeSeries.from_dataframe(df_sku[["ds", "y"]], "ds", "y", freq=WEEK_FREQ)
    n = len(series)
    desired_in = 26
    out_len = max(1, int(horizon_weeks))
    min_in = 8
    in_len = min(desired_in, max(min_in, n - out_len - 1))
    if n < in_len + out_len + 1:
        return pd.DataFrame(columns=["ds", "nbeats_yhat", "nbeats_yhat_lower", "nbeats_yhat_upper"])
    can_val = n >= (in_len + out_len + 20)
    val_series = None
    early_cb = None
    pl_kwargs = {"enable_checkpointing": False, "logger": False}
    if can_val:
        _, val_series = series.split_before(0.70)
        early_cb = EarlyStopping(monitor="val_loss", patience=5, mode="min")
        pl_kwargs["callbacks"] = [early_cb]
    else:
        early_cb = EarlyStopping(monitor="train_loss", patience=5, mode="min")
        pl_kwargs["callbacks"] = [early_cb]
    bs = max(4, min(64, n // 4))
    model = NBEATSModel(
        input_chunk_length=in_len, output_chunk_length=out_len,
        n_epochs=50, batch_size=bs, random_state=42, pl_trainer_kwargs=pl_kwargs,
    )
    try:
        import torch
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(1)
    except Exception:
        prev_threads = None
    model.fit(series, verbose=False, val_series=val_series)
    if prev_threads is not None:
        try:
            import torch
            torch.set_num_threads(prev_threads)
        except Exception:
            pass
    forecast = model.predict(out_len)
    df_pred = forecast.pd_dataframe().reset_index().rename(columns={"index": "ds", "y": "nbeats_yhat"})
    df_pred["nbeats_yhat_lower"] = np.nan
    df_pred["nbeats_yhat_upper"] = np.nan
    hist_frames = []
    try:
        if n >= in_len + out_len + 5:
            hist = (
                model.historical_forecasts(series, start=0.8, forecast_horizon=1, verbose=False)
                .pd_dataframe().reset_index()
            )
            hist.rename(columns={"index": "ds", "y": "nbeats_yhat"}, inplace=True)
            hist["nbeats_yhat_lower"] = np.nan
            hist["nbeats_yhat_upper"] = np.nan
            hist_frames.append(hist)
    except Exception:
        pass
    if hist_frames:
        return pd.concat([hist_frames[0], df_pred], ignore_index=True)
    return df_pred

# =====================================================================
# ROUTES - Homepage & Navigation
# =====================================================================

@app.get("/")
async def root():
    """Homepage with navigation to both apps"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Predict & Pour</title>
        <style>
            body { 
                margin: 0; 
                font-family: Arial, sans-serif; 
                background: #000;
                color: #fff;
            }
            
            /* VIDEO HEADER - MATCHES YOUR APPS */
            .header { 
                position: relative;
                overflow: hidden;
                color: #EBBB40; 
                padding: 20px; 
                text-align: center; 
                border-bottom: 3px solid #EBBB40;
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 150px;
            }
            .header-video {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                object-fit: cover;
                z-index: 0;
            }
            .header-content {
                position: relative;
                z-index: 2;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-direction: column;
            }
            .header h1 {
                -webkit-text-stroke: 4px #000;
                text-stroke: 4px #000;
                paint-order: stroke fill;
                margin: 10px 0;
                font-size: 48px;
            }
            .header p {
                -webkit-text-stroke: 3px #000;
                text-stroke: 3px #000;
                paint-order: stroke fill;
                margin: 0;
                font-size: 20px;
            }
            
            .container {
                max-width: 1000px;
                margin: 50px auto;
                text-align: center;
                padding: 20px;
            }
            
            .app-cards {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin: 40px 0;
            }
            
            .app-card {
                background: url('https://static.vecteezy.com/system/resources/previews/027/815/346/large_2x/dark-wood-background-texture-rustic-wooden-floor-textured-backdrop-free-photo.jpg');
                background-size: cover;
                border: 3px solid #EBBB40;
                border-radius: 12px;
                padding: 40px 30px;
                transition: transform 0.3s, box-shadow 0.3s;
            }
            
            .app-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(235, 187, 64, 0.3);
            }
            
            .app-card h2 {
                color: #EBBB40;
                margin: 20px 0 10px 0;
                font-size: 32px;
            }
            
            .app-card p {
                color: #fff;
                font-size: 16px;
                margin: 15px 0;
                line-height: 1.6;
            }
            
            .app-card a {
                display: inline-block;
                background: #EBBB40;
                color: #000;
                padding: 15px 40px;
                text-decoration: none;
                border-radius: 6px;
                font-weight: bold;
                margin-top: 20px;
                font-size: 16px;
                transition: background 0.3s;
            }
            
            .app-card a:hover {
                background: #d4a634;
            }
            
            .emoji {
                font-size: 64px;
                margin: 10px 0;
            }
            
            .subtitle {
                color: #EBBB40;
                font-size: 24px;
                margin: 30px 0 10px 0;
            }
            
            .description {
                color: #ddd;
                max-width: 800px;
                margin: 0 auto 40px auto;
                line-height: 1.8;
                font-size: 18px;
            }
        </style>
    </head>
    <body>
        <!-- VIDEO HEADER -->
        <div class="header">
            <video autoplay muted loop playsinline class="header-video">
                <source src="https://i.imgur.com/ity2XJw.mp4" type="video/mp4">
            </video>
            <div class="header-content">
                <img src="https://i.imgur.com/Bf1hNE0.png" alt="Predict & Pour Logo" width="150" style="margin-bottom: 10px;">
                <h1>Predict & Pour System</h1>
                <p>Professional Forecasting & Execution Platform</p>
            </div>
        </div>
        
        <div class="container">
            <h2 class="subtitle">Choose Your Tool</h2>
            <p class="description">
                A complete forecasting solution: Generate multi-model AI predictions with Predict, 
                then automate execution with Pour. The perfect one-two punch for data-driven decisions.
            </p>
            
            <div class="app-cards">
                <div class="app-card">
                    <img src="https://i.imgur.com/uFQzEoN.png" alt="Predict Logo" style="width: 150px; height: 150px; margin: 10px 0;">
                    <h2>Predict</h2>
                    <p>Multi-model forecasting powered by advanced algorithims and AI models. Choose from 8 different forecasting models to predict future sales.</p>
                    <a href="/predict">Launch Predict →</a>
                </div>
                
                <div class="app-card">
                    <img src="https://i.imgur.com/cNL5gwd.png" alt="Pour Logo" style="width: 150px; height: 150px; margin: 10px 0;">
                    <h2>Pour</h2>
                    <p>Automated forecast execution into OnePortal. Smart calibration and one-click import automation for your data.</p>
                    <a href="/pour">Launch Pour →</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """)

@app.get("/predict")
async def serve_predict():
    """Serve the Predict app"""
    return FileResponse("static/predict.html")

@app.get("/pour")
async def serve_pour():
    """Serve the Pour app"""
    return FileResponse("static/pour.html")

# =====================================================================
# PREDICT APP ROUTES (Your existing forecast endpoint)
# =====================================================================

@app.post("/api/forecast")
async def create_forecast(file: UploadFile = File(...), config: str = None):
    """Forecast endpoint - SAME AS YOUR EXISTING CODE"""
    # [PASTE YOUR ENTIRE EXISTING FORECAST LOGIC HERE]
    # This is the same code from your original main.py /forecast endpoint
    pass  # Replace with your full forecast code

# =====================================================================
# POUR APP ROUTES (New automation endpoints)
# =====================================================================

@app.post("/api/pour/upload")
async def upload_pour_file(file: UploadFile = File(...)):
    """Upload Excel file for Pour app"""
    global uploaded_pour_data
    
    contents = await file.read()
    df = pd.read_excel(BytesIO(contents))
    
    # Clean data (from original Streamlit code)
    df.columns = df.columns.str.strip().str.lower()
    df = df[df['week number'] != "Past Date"]
    df = df[df['supplier'].str.strip().str.lower() == "anheuser busch"]
    df = df[['pdcn', 'description', 'final forecast', 'week number']]
    
    uploaded_pour_data = df
    
    # Create display options
    df['pdcn'] = df['pdcn'].astype(str)
    df['description'] = df['description'].astype(str)
    df['display_label'] = df['pdcn'] + " (" + df['description'] + ")"
    unique_options = df[['pdcn', 'display_label']].drop_duplicates().sort_values('display_label')
    
    return {
        "status": "success",
        "items": unique_options.to_dict('records')
    }

@app.post("/api/pour/calibrate")
async def calibrate_coordinates():
    """Capture mouse coordinates after 3 second delay"""
    time.sleep(3)
    pos = pyautogui.position()
    return {"x": pos.x, "y": pos.y}

@app.post("/api/pour/find-pdcn")
async def find_pdcn(pdcn_x: int, pdcn_y: int):
    """Find PDCN by reading from screen"""
    pyautogui.moveTo(pdcn_x, pdcn_y)
    pyautogui.doubleClick()
    time.sleep(0.2)
    pyautogui.hotkey('ctrl', 'c')
    time.sleep(0.2)
    copied_pdcn = pyperclip.paste().strip()
    
    return {"pdcn": copied_pdcn}

@app.post("/api/pour/execute")
async def execute_import(request: ExecuteImportRequest):
    """Execute the automation to type forecast values"""
    global uploaded_pour_data
    
    if uploaded_pour_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    
    # Filter data for selected PDCN
    filtered_df = uploaded_pour_data[uploaded_pour_data['pdcn'] == request.pdcn].sort_values(by='week number')
    
    if filtered_df.empty:
        raise HTTPException(status_code=404, detail="PDCN not found")
    
    # Focus browser
    browser_window = None
    for w in gw.getWindowsWithTitle(''):
        if any(browser in w.title for browser in ['Chrome', 'Edge', 'Safari', 'Firefox']):
            browser_window = w
            break
    
    if browser_window:
        browser_window.activate()
        time.sleep(0.5)
    
    # Click Week 1 field
    pyautogui.moveTo(request.week1_x, request.week1_y)
    pyautogui.click()
    time.sleep(0.5)
    
    # Type values
    values_typed = []
    for value in filtered_df['final forecast']:
        whole_number = str(int(round(value)))
        pyautogui.write(whole_number)
        pyautogui.press('tab')
        time.sleep(0.05)
        values_typed.append(whole_number)
    
    return {
        "status": "success",
        "pdcn": request.pdcn,
        "values_typed": values_typed,
        "count": len(values_typed)
    }

@app.get("/api/models")
async def get_available_models():
    """Return available forecast models"""
    return {
        "prophet": True,
        "sarimax": True,
        "holt_winters": True,
        "catboost": True,
        "lightgbm": True,
        "nbeats": True,
        "neuralprophet": _HAS_NEURALPROPHET,
        "xgboost": _HAS_XGBOOST,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)