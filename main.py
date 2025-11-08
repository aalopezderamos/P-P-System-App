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
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
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
# NEW: Dynamic Formula Builder Helper Functions
# =====================================================================

def _col_number_to_letter(n: int) -> str:
    """Convert 0-indexed column number to Excel column letter (A, B, ..., Z, AA, AB, ...)"""
    result = ""
    while n >= 0:
        result = chr(n % 26 + 65) + result
        n = n // 26 - 1
    return result


def get_model_column_letter(df_out: pd.DataFrame, model_col_name: str) -> Optional[str]:
    """
    Get Excel column letter for a given DataFrame column name.
    Returns None if column doesn't exist.
    """
    if model_col_name not in df_out.columns:
        return None
    
    col_idx = df_out.columns.get_loc(model_col_name)
    return _col_number_to_letter(col_idx)


def build_dynamic_final_forecast_formula(row_num: int, model_col_refs: List[str], ab_current_col_ref: str = None) -> str:
    """
    Build a dynamic Final Forecast formula that:
    1. Includes all model columns + AB Current Forecast
    2. Filters out blanks and zeros
    3. Drops bottom 25% (rounded) of values based on distance from mean
    4. Returns average of remaining values
    
    Args:
        row_num: Excel row number (1-indexed)
        model_col_refs: List of Excel column references (e.g., ['C2', 'J2', 'M2'])
        ab_current_col_ref: Column reference for AB Current Forecast (e.g., 'AC2')
    
    Returns:
        Excel formula string
    """
    # Combine all value sources
    all_refs = model_col_refs.copy()
    if ab_current_col_ref:
        all_refs.append(ab_current_col_ref)
    
    if not all_refs:
        return ""
    
    # Build index array for CHOOSE function
    num_refs = len(all_refs)
    indices = ",".join([str(i+1) for i in range(num_refs)])
    
    # Build cell references for CHOOSE function
    refs_string = ",".join(all_refs)
    
    # Build the formula - avoid f-string issues by building piece by piece
    formula = (
        "LET("
        "all_vals, TOCOL(CHOOSE({" + indices + "}," + refs_string + ")),"
        "valid_vals, FILTER(all_vals, (all_vals<>\"\")*(all_vals<>0)),"
        "n_vals, COUNTA(valid_vals),"
        "mean_val, AVERAGE(valid_vals),"
        "n_drop, ROUND(n_vals*0.25, 0),"
        "n_keep, n_vals - n_drop,"
        "sorted_vals, SORTBY(valid_vals, ABS(valid_vals - mean_val), 1),"
        "kept_vals, IF(n_keep > 0, TAKE(sorted_vals, n_keep), valid_vals),"
        "AVERAGE(kept_vals)"
        ")"
    )
    
    return f"=IFERROR({formula}, \"\")"
# =====================================================================
# Model Functions
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
            "yhat": "house_lager_yhat",
            "yhat_lower": "house_lager_yhat_lower",
            "yhat_upper": "house_lager_yhat_upper",
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
    out = forecast[["ds", "yhat1"]].rename(columns={"yhat1": "mind_melt_double_ipa_yhat"})
    out["mind_melt_double_ipa_yhat_lower"] = forecast.get("yhat1 5.0%", np.nan)
    out["mind_melt_double_ipa_yhat_upper"] = forecast.get("yhat1 95.0%", np.nan)
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
    conf = conf.rename(columns={lower_col: "heritage_blend_yhat_lower", upper_col: "heritage_blend_yhat_upper"})

    future_index = pd.date_range(start=ts.index.max() + pd.Timedelta(weeks=1), periods=horizon_weeks, freq=WEEK_FREQ)
    hist_df = pd.DataFrame({"ds": ts.index, "heritage_blend_yhat": res.fittedvalues})
    fut_df = pd.DataFrame({"ds": future_index, "heritage_blend_yhat": pred_mean.values})
    fut_df = pd.concat([fut_df.reset_index(drop=True), conf.reset_index(drop=True)], axis=1)
    out = pd.concat([hist_df, fut_df], ignore_index=True)
    out["heritage_blend_yhat_lower"] = out.get("heritage_blend_yhat_lower", np.nan)
    out["heritage_blend_yhat_upper"] = out.get("heritage_blend_yhat_upper", np.nan)
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
            "ds": ds, "west_coast_ipa_yhat": pred,
            "west_coast_ipa_yhat_lower": pred - PI_Z_90 * resid_std,
            "west_coast_ipa_yhat_upper": pred + PI_Z_90 * resid_std,
        })
        full_y.loc[ds] = pred
    hist_pred = model.predict(X)
    hist_df = pd.DataFrame({"ds": X.index, "west_coast_ipa_yhat": hist_pred})
    hist_df["west_coast_ipa_yhat_lower"] = np.nan
    hist_df["west_coast_ipa_yhat_upper"] = np.nan
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
            "ds": ds, "creamy_nitro_yhat": pred,
            "creamy_nitro_yhat_lower": pred - PI_Z_90 * resid_std,
            "creamy_nitro_yhat_upper": pred + PI_Z_90 * resid_std,
        })
        full_y.loc[ds] = pred
    hist_pred = model.predict(X)
    hist_df = pd.DataFrame({"ds": X.index, "creamy_nitro_yhat": hist_pred})
    hist_df["creamy_nitro_yhat_lower"] = np.nan
    hist_df["creamy_nitro_yhat_upper"] = np.nan
    fut_df = pd.DataFrame(rows)
    return pd.concat([hist_df, fut_df], ignore_index=True)

def run_holtwinters(df_sku: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    ts = df_sku.set_index("ds")["y"].asfreq(WEEK_FREQ)
    model = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=52).fit(optimized=True)
    hist_df = pd.DataFrame({"ds": ts.index, "small_batch_classic_yhat": model.fittedvalues})
    hist_df["small_batch_classic_yhat_lower"] = np.nan
    hist_df["small_batch_classic_yhat_upper"] = np.nan
    future_index = pd.date_range(start=ts.index.max() + pd.Timedelta(weeks=1), periods=horizon_weeks, freq=WEEK_FREQ)
    forecast = model.forecast(horizon_weeks)
    resid_std = np.std(model.resid)
    fut_df = pd.DataFrame({
        "ds": future_index, "small_batch_classic_yhat": forecast.values,
        "small_batch_classic_yhat_lower": forecast.values - PI_Z_90 * resid_std,
        "small_batch_classic_yhat_upper": forecast.values + PI_Z_90 * resid_std,
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
            "ds": ds, "light_hazy_yhat": pred,
            "light_hazy_yhat_lower": pred - PI_Z_90 * resid_std,
            "light_hazy_yhat_upper": pred + PI_Z_90 * resid_std,
        })
        full_y.loc[ds] = pred
    hist_pred = model.predict(X)
    hist_df = pd.DataFrame({"ds": X.index, "light_hazy_yhat": hist_pred})
    hist_df["light_hazy_yhat_lower"] = np.nan
    hist_df["light_hazy_yhat_upper"] = np.nan
    fut_df = pd.DataFrame(rows)
    return pd.concat([hist_df, fut_df], ignore_index=True)

def run_nbeats(df_sku: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    # Drop any NaN values in y column first
    df_clean = df_sku[["ds", "y"]].dropna(subset=["y"]).copy()
    
    if len(df_clean) < 50:  # Need minimum data
        return pd.DataFrame(columns=["ds", "legacy_grand_reserve_yhat", "legacy_grand_reserve_yhat_lower", "legacy_grand_reserve_yhat_upper"])
    
    try:
        series = TimeSeries.from_dataframe(df_clean, "ds", "y", freq=WEEK_FREQ, fill_missing_dates=True)
    except Exception as e:
        print(f"N-BEATS TimeSeries creation failed: {e}")
        return pd.DataFrame(columns=["ds", "legacy_grand_reserve_yhat", "legacy_grand_reserve_yhat_lower", "legacy_grand_reserve_yhat_upper"])

    n = len(series)
    
    # More conservative parameters for shorter series
    if n < 60:
        # Too short for N-BEATS to work reliably
        return pd.DataFrame(columns=["ds", "legacy_grand_reserve_yhat", "legacy_grand_reserve_yhat_lower", "legacy_grand_reserve_yhat_upper"])
    
    desired_in = min(26, n // 3)  # Adaptive input length
    out_len = max(1, int(horizon_weeks))
    min_in = 8
    in_len = min(desired_in, max(min_in, n - out_len - 10))  # Leave more buffer

    if n < in_len + out_len + 10:  # Need more buffer
        return pd.DataFrame(columns=["ds", "legacy_grand_reserve_yhat", "legacy_grand_reserve_yhat_lower", "legacy_grand_reserve_yhat_upper"])

    # Decide whether we can afford a validation split
    can_val = n >= (in_len + out_len + 25)  # More conservative
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

    bs = max(4, min(32, n // 6))  # Smaller batch size

    model = NBEATSModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        n_epochs=30,  # Reduced epochs for speed
        batch_size=bs,
        random_state=42,
        pl_trainer_kwargs=pl_kwargs,
    )

    try:
        import torch
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(1)
    except Exception:
        prev_threads = None

    try:
        model.fit(series, verbose=False, val_series=val_series)
    except Exception as e:
        print(f"N-BEATS fit failed: {e}")
        if prev_threads is not None:
            try:
                import torch
                torch.set_num_threads(prev_threads)
            except Exception:
                pass
        return pd.DataFrame(columns=["ds", "legacy_grand_reserve_yhat", "legacy_grand_reserve_yhat_lower", "legacy_grand_reserve_yhat_upper"])

    if prev_threads is not None:
        try:
            import torch
            torch.set_num_threads(prev_threads)
        except Exception:
            pass

    # Predict future
    try:
        forecast = model.predict(out_len)
        df_pred = forecast.pd_dataframe().reset_index().rename(columns={"index": "ds", "y": "legacy_grand_reserve_yhat"})
        df_pred["legacy_grand_reserve_yhat_lower"] = np.nan
        df_pred["legacy_grand_reserve_yhat_upper"] = np.nan
    except Exception as e:
        print(f"N-BEATS predict failed: {e}")
        return pd.DataFrame(columns=["ds", "legacy_grand_reserve_yhat", "legacy_grand_reserve_yhat_lower", "legacy_grand_reserve_yhat_upper"])

    # Skip historical forecasts for simplicity/reliability
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
                position: relative;
            }
            body::before {
                content: '';
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: url('https://i.imgur.com/cdzKb9r.png') repeat;
                opacity: 0.18;
                z-index: -1;
                pointer-events: none;
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
                font-size: 26px;
            }
            
            .container {
                max-width: 1000px;
                margin: 0px auto;
                text-align: center;
                padding: 15px 20px;
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
                transform: translateY(-15px) scale(1.05);
                box-shadow: 0 25px 80px rgba(235, 187, 64, 0.7);
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
                margin: 10px 0 10px 0;
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
                <h1>Predict & Pour Forecasting System</h1>
                <p>Professional Forecasting & Execution Platform</p>
            </div>
        </div>
        
        <div class="container">
            <h2 class="subtitle">Choose Your Tool</h2>
            <p class="description">
                A complete forecasting solution: Generate multi-model AI predictions with Predict, 
                then automate execution with Pour. The perfect one-two punch for data-driven forecasting.
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
async def create_forecast(file: UploadFile = File(...), config: str = Form(None)):
    """
    Multi-model forecast endpoint
    Processes CSV upload and returns Excel with forecasts
    """
    try:
        # Parse configuration
        if config:
            try:
                config_dict = json.loads(config)
                print(f"DEBUG: Received config: {config_dict}")  # Debug log
            except json.JSONDecodeError as e:
                print(f"ERROR: Failed to parse config JSON: {e}")
                config_dict = {}
        else:
            print("WARNING: No config received, using defaults")
            config_dict = {}
        
        horizon_weeks = config_dict.get("horizon_weeks", 12)
        min_points = config_dict.get("min_points", 50)
        show_debug = config_dict.get("show_debug", False)
        
        # Get model flags - NO DEFAULTS (False if not specified)
        run_prophet_flag = config_dict.get("run_prophet", False)
        run_neural_flag = config_dict.get("run_neural", False)
        run_sarimax_flag = config_dict.get("run_sarimax", False)
        run_xgb_flag = config_dict.get("run_xgb", False)
        run_catboost_flag = config_dict.get("run_catboost", False)
        run_holt_flag = config_dict.get("run_holt", False)
        run_lgbm_flag = config_dict.get("run_lgbm", False)
        run_nbeats_flag = config_dict.get("run_nbeats", False)
        
        # DEBUG: Print what models were requested
        print("\n===== MODEL SELECTION DEBUG =====")
        print(f"Prophet (House Lager): {run_prophet_flag}")
        print(f"NeuralProphet (Mind Melt): {run_neural_flag}")
        print(f"SARIMAX (Heritage Blend): {run_sarimax_flag}")
        print(f"XGBoost (West Coast IPA): {run_xgb_flag}")
        print(f"CatBoost (Creamy Nitro): {run_catboost_flag}")
        print(f"Holt-Winters (Small Batch): {run_holt_flag}")
        print(f"LightGBM (Light Hazy): {run_lgbm_flag}")
        print(f"N-BEATS (Legacy Grand Reserve): {run_nbeats_flag}")
        print("="*40 + "\n")
        
        print(f"DEBUG: Model flags - Prophet:{run_prophet_flag}, Sarimax:{run_sarimax_flag}, Holt:{run_holt_flag}, XGB:{run_xgb_flag}")

        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
        
        # Validate required columns
        required_cols = {"sku_id", "ds", "y"}
        if not required_cols.issubset(df.columns):
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {required_cols - set(df.columns)}"
            )

        # Process data
        df["ds"] = pd.to_datetime(df["ds"])
        holidays_df = get_custom_holidays()
        holidays_df["ds"] = pd.to_datetime(holidays_df["ds"])

        sku_list = df["sku_id"].unique()
        total = len(sku_list)

        detailed_rows = []

        for idx, (sku, df_sku) in enumerate(df.groupby("sku_id"), start=1):
            print(f"Processing {idx}/{total} → {sku}")  # Console logging

            if len(df_sku) < min_points:
                continue

            try:
                base = pd.DataFrame({"ds": df_sku["ds"].unique()}).sort_values("ds")
                base = base.merge(df_sku[["ds", "y"]], on="ds", how="left")

                model_futures = {}
                with ThreadPoolExecutor(max_workers=6) as executor:
                    if run_prophet_flag:
                        model_futures["house_lager"] = executor.submit(
                            run_prophet, df_sku, holidays_df, horizon_weeks
                        )
                    if run_neural_flag:
                        model_futures["mind_melt_double_ipa"] = executor.submit(
                            run_neuralprophet, df_sku, holidays_df, horizon_weeks
                        )
                    if run_sarimax_flag:
                        model_futures["heritage_blend"] = executor.submit(
                            run_sarimax, df_sku, horizon_weeks
                        )
                    if run_xgb_flag:
                        model_futures["west_coast_ipa"] = executor.submit(
                            run_xgboost, df_sku, horizon_weeks
                        )
                    if run_catboost_flag:
                        model_futures["creamy_nitro"] = executor.submit(
                            run_catboost, df_sku, horizon_weeks
                        )
                    if run_holt_flag:
                        model_futures["small_batch_classic"] = executor.submit(
                            run_holtwinters, df_sku, horizon_weeks
                        )
                    if run_lgbm_flag:
                        model_futures["light_hazy"] = executor.submit(
                            run_lightgbm, df_sku, horizon_weeks
                        )
                    if run_nbeats_flag:
                        model_futures["legacy_grand_reserve"] = executor.submit(
                            run_nbeats, df_sku, horizon_weeks
                        )

                # Merge model outputs
                for model_name, future in model_futures.items():
                    try:
                        result_df = future.result()
                        base = base.merge(result_df, on="ds", how="outer", validate="one_to_one")
                    except Exception as e:
                        print(f"Warning: {model_name} failed for {sku}: {e}")

                # Compute model-specific accuracy
                for model_prefix in [
                    "house_lager", "mind_melt_double_ipa", "heritage_blend", 
                    "west_coast_ipa", "creamy_nitro", "small_batch_classic", 
                    "light_hazy", "legacy_grand_reserve"
                ]:
                    pred_col = f"{model_prefix}_yhat"
                    if pred_col in base.columns:
                        base[f"{model_prefix}_acc"] = base.apply(
                            lambda r: safe_accuracy(r.get("y"), r.get(pred_col)), axis=1
                        )

                base["sku_id"] = sku
                detailed_rows.append(base)

            except Exception as e:
                print(f"Warning: Error processing {sku}: {e}")
                continue

        if not detailed_rows:
            raise HTTPException(
                status_code=400,
                detail="No forecasts were generated. Check your file or adjust filters."
            )

        df_out = pd.concat(detailed_rows, ignore_index=True).sort_values(["sku_id", "ds"])

        # ======================== Excel Export ========================
        output = BytesIO()
        with pd.ExcelWriter(
            output, 
            engine="xlsxwriter", 
            date_format="mm/dd/yyyy", 
            datetime_format="mm/dd/yyyy",
            engine_kwargs={'options': {'strings_to_formulas': True, 'strings_to_urls': False}}
        ) as writer:
            df_out.to_excel(writer, index=False, sheet_name="forecasts")
            workbook = writer.book
            workbook.use_future_functions = True
            worksheet = writer.sheets["forecasts"]
            n_rows, n_cols = df_out.shape

            # Formats
            date_fmt = workbook.add_format({"num_format": "mm/dd/yyyy"})
            int_fmt = workbook.add_format({"num_format": "0"})
            pct_fmt = workbook.add_format({"num_format": "0.00%"})
            wrap_center = {"bold": True, "text_wrap": True, "align": "center", "valign": "vcenter"}
            hdr1 = workbook.add_format({**wrap_center, "bg_color": "#E6B8B7"})
            hdr2 = workbook.add_format({**wrap_center, "bg_color": "#CCC0DA"})
            hdr3 = workbook.add_format({**wrap_center, "bg_color": "#8DB4E2"})
            hdr4 = workbook.add_format({**wrap_center, "bg_color": "#FCD5B4"})
            hdr5 = workbook.add_format({**wrap_center, "bg_color": "#C6EFCE"})
            hdr6 = workbook.add_format({**wrap_center, "bg_color": "#FFFF00"})
            hdr7 = workbook.add_format({**wrap_center, "bg_color": "#B4C6E7"})
            hdr8 = workbook.add_format({**wrap_center, "bg_color": "#D9EAD3"})
            hdr9 = workbook.add_format({**wrap_center, "bg_color": "#F4CCCC"})
            hdr10 = workbook.add_format({**wrap_center, "bg_color": "#FFF2CC"})
            hdr_extra = workbook.add_format({**wrap_center, "bg_color": "#C4BD97"})

            # Extra columns with formulas
            extra_headers = [
                ("Increase", "=X{row}*0.93"),
                (
                    "AB Past Forecast",
                    "=XLOOKUP(AF{row}, '[Master Incoming Report NEW.xlsm]AB Forecast Accy-Bias'!$B:$B, "
                    "'[Master Incoming Report NEW.xlsm]AB Forecast Accy-Bias'!$M:$M, \"\")",
                ),
                (
                    "Past Forecast",
                    "=XLOOKUP(AF{row}, '[Master Incoming Report NEW.xlsm]AB Forecast Accy-Bias'!$B:$B, "
                    "'[Master Incoming Report NEW.xlsm]AB Forecast Accy-Bias'!$AF:$AF, \"\")",
                ),
                (
                    "AB Current Forecast",
                    "=XLOOKUP(AF{row}, '[Master Incoming Report NEW.xlsm]AB Forecast Report'!$A:$A, "
                    "'[Master Incoming Report NEW.xlsm]AB Forecast Report'!$P:$P, \"\")",
                ),
                (
                    "Current Forecast",
                    "=XLOOKUP(AF{row}, '[Master Incoming Report NEW.xlsm]AB Forecast Report'!$A:$A, "
                    "'[Master Incoming Report NEW.xlsm]AB Forecast Report'!$T:$T, \"\")",
                ),
                (
                    "LY Sales",
                    "=XLOOKUP(AF{row}, '[Master Incoming Report NEW.xlsm]AB Forecast Report'!$A:$A, "
                    "'[Master Incoming Report NEW.xlsm]AB Forecast Report'!$S:$S, \"\")",
                ),
                ("Helper", "=W{row}&A{row}"),
                (
                    "Week Number",
                    '=IF(A{row} < (TODAY() - WEEKDAY(TODAY(), 2) + 1), "Past Date", '
                    'IF(A{row} <= (TODAY() - WEEKDAY(TODAY(), 2) + 6), 0, '
                    'IF(INT((A{row} - (TODAY() - WEEKDAY(TODAY(), 2) + 1)) / 7) <= 12, '
                    'INT((A{row} - (TODAY() - WEEKDAY(TODAY(), 2) + 1)) / 7), "Past Date")))'
                ),
                (
                    "PDCN",
                    "=XLOOKUP(W{row}, '[Master Incoming Report NEW.xlsm]Overview'!$D:$D, "
                    "'[Master Incoming Report NEW.xlsm]Overview'!$E:$E, \"\")",
                ),
                (
                    "Supplier",
                    "=XLOOKUP(W{row}, '[Master Incoming Report NEW.xlsm]Overview'!$D:$D, "
                    "'[Master Incoming Report NEW.xlsm]Overview'!$B:$B, \"\")",
                ),
                (
                    "Description",
                    "=XLOOKUP(W{row}, '[Master Incoming Report NEW.xlsm]Overview'!$D:$D, "
                    "'[Master Incoming Report NEW.xlsm]Overview'!$C:$C, \"\")",
                ),
            ]
            numeric_headers = {
                "Increase",
                "AB Past Forecast",
                "Past Forecast",
                "AB Current Forecast",
                "Current Forecast",
                "LY Sales",
            }

            # Freeze panes and set column widths
            worksheet.freeze_panes(1, 0)
            num_total_cols = n_cols + 2 + len(extra_headers)
            for col in range(num_total_cols):
                px = 75
                width = (px - 5) / 7
                worksheet.set_column(col, col, width)

            # ========== FRIENDLY COLUMN NAMES MAPPING ==========
            friendly_names = {
                # Base columns
                "ds": "Date",
                "y": "Actual Sales",
                "sku_id": "SKU ID",
                
                # House Lager (Prophet)
                "house_lager_yhat": "House Lager Forecast",
                "house_lager_yhat_lower": "House Lager Lower",
                "house_lager_yhat_upper": "House Lager Upper",
                "house_lager_acc": "House Lager Accy",
                "trend": "Trend",
                "weekly": "Weekly",
                "yearly": "Yearly",
                "holidays": "Holidays",
                
                # Mind Melt Double IPA (NeuralProphet)
                "mind_melt_double_ipa_yhat": "Mind Melt IPA Forecast",
                "mind_melt_double_ipa_yhat_lower": "Mind Melt IPA Lower",
                "mind_melt_double_ipa_yhat_upper": "Mind Melt IPA Upper",
                "mind_melt_double_ipa_acc": "Mind Melt IPA Accy",
                
                # Heritage Blend (SARIMAX)
                "heritage_blend_yhat": "Heritage Blend Forecast",
                "heritage_blend_yhat_lower": "Heritage Blend Lower",
                "heritage_blend_yhat_upper": "Heritage Blend Upper",
                "heritage_blend_acc": "Heritage Blend Accy",
                
                # West Coast IPA (XGBoost)
                "west_coast_ipa_yhat": "West Coast IPA Forecast",
                "west_coast_ipa_yhat_lower": "West Coast IPA Lower",
                "west_coast_ipa_yhat_upper": "West Coast IPA Upper",
                "west_coast_ipa_acc": "West Coast IPA Accy",
                
                # Creamy Nitro (CatBoost)
                "creamy_nitro_yhat": "Creamy Nitro Forecast",
                "creamy_nitro_yhat_lower": "Creamy Nitro Lower",
                "creamy_nitro_yhat_upper": "Creamy Nitro Upper",
                "creamy_nitro_acc": "Creamy Nitro Accy",
                
                # Small Batch Classic (Holt-Winters)
                "small_batch_classic_yhat": "Small Batch Classic Forecast",
                "small_batch_classic_yhat_lower": "Small Batch Classic Lower",
                "small_batch_classic_yhat_upper": "Small Batch Classic Upper",
                "small_batch_classic_acc": "Small Batch Classic Accy",
                
                # Light Hazy (LightGBM)
                "light_hazy_yhat": "Light Hazy Forecast",
                "light_hazy_yhat_lower": "Light Hazy Lower",
                "light_hazy_yhat_upper": "Light Hazy Upper",
                "light_hazy_acc": "Light Hazy Accy",
                
                # Legacy Grand Reserve (N-BEATS)
                "legacy_grand_reserve_yhat": "Legacy Grand Reserve Forecast",
                "legacy_grand_reserve_yhat_lower": "Legacy Grand Reserve Lower",
                "legacy_grand_reserve_yhat_upper": "Legacy Grand Reserve Upper",
                "legacy_grand_reserve_acc": "Legacy Grand Reserve Accy",
            }

            # Format existing columns with beer names
            col_idx = {name: i for i, name in enumerate(df_out.columns)}
            group_map = {
                hdr1: ["ds", "y", "sku_id"],
                hdr2: [
                    "house_lager_yhat",
                    "house_lager_yhat_lower",
                    "house_lager_yhat_upper",
                    "trend",
                    "weekly",
                    "yearly",
                    "holidays",
                    "house_lager_acc",
                ],
                hdr3: [
                    "mind_melt_double_ipa_yhat",
                    "mind_melt_double_ipa_yhat_lower",
                    "mind_melt_double_ipa_yhat_upper",
                    "mind_melt_double_ipa_acc"
                ],
                hdr4: [
                    "heritage_blend_yhat",
                    "heritage_blend_yhat_lower",
                    "heritage_blend_yhat_upper",
                    "heritage_blend_acc"
                ],
                hdr5: [
                    "west_coast_ipa_yhat",
                    "west_coast_ipa_yhat_lower",
                    "west_coast_ipa_yhat_upper",
                    "west_coast_ipa_acc"
                ],
                hdr7: [
                    "creamy_nitro_yhat",
                    "creamy_nitro_yhat_lower",
                    "creamy_nitro_yhat_upper",
                    "creamy_nitro_acc"
                ],
                hdr8: [
                    "small_batch_classic_yhat",
                    "small_batch_classic_yhat_lower",
                    "small_batch_classic_yhat_upper",
                    "small_batch_classic_acc"
                ],
                hdr9: [
                    "light_hazy_yhat",
                    "light_hazy_yhat_lower",
                    "light_hazy_yhat_upper",
                    "light_hazy_acc"
                ],
                hdr10: [
                    "legacy_grand_reserve_yhat",
                    "legacy_grand_reserve_yhat_lower",
                    "legacy_grand_reserve_yhat_upper",
                    "legacy_grand_reserve_acc"
                ],
            }
            
            for fmt, names in group_map.items():
                for name in names:
                    if name not in col_idx:
                        continue
                    c = col_idx[name]
                    
                    # Use friendly name if available, otherwise use original name
                    display_name = friendly_names.get(name, name)
                    
                    worksheet.write(0, c, display_name, fmt)
                    if name == "ds":
                        worksheet.set_column(c, c, 11, date_fmt)  # Wider for "Date"
                    elif name.endswith("_acc"):
                        worksheet.set_column(c, c, 11, pct_fmt)  # Wider for "Accuracy"
                    else:
                        worksheet.set_column(c, c, 11, int_fmt)  # Wider for longer names

            # ========== DYNAMIC COLUMN GROUPING (collapsible sections) ==========
            
            # Define which columns to hide for each model (detail columns like lower/upper bounds, components)
            model_detail_columns = {
                "house_lager": [
                    "house_lager_yhat_lower",
                    "house_lager_yhat_upper", 
                    "trend",
                    "weekly",
                    "yearly",
                    "holidays"
                ],
                "mind_melt_double_ipa": [
                    "mind_melt_double_ipa_yhat_lower",
                    "mind_melt_double_ipa_yhat_upper"
                ],
                "heritage_blend": [
                    "heritage_blend_yhat_lower",
                    "heritage_blend_yhat_upper"
                ],
                "west_coast_ipa": [
                    "west_coast_ipa_yhat_lower",
                    "west_coast_ipa_yhat_upper"
                ],
                "creamy_nitro": [
                    "creamy_nitro_yhat_lower",
                    "creamy_nitro_yhat_upper"
                ],
                "small_batch_classic": [
                    "small_batch_classic_yhat_lower",
                    "small_batch_classic_yhat_upper"
                ],
                "light_hazy": [
                    "light_hazy_yhat_lower",
                    "light_hazy_yhat_upper"
                ],
                "legacy_grand_reserve": [
                    "legacy_grand_reserve_yhat_lower",
                    "legacy_grand_reserve_yhat_upper"
                ]
            }
            
            # Group detail columns for each model that's present
            for model_name, detail_cols in model_detail_columns.items():
                # Check if this model was run (check for main yhat column)
                main_col = f"{model_name}_yhat"
                if main_col not in df_out.columns:
                    continue  # Skip this model, it wasn't run
                
                # Find the column indices for detail columns that exist
                detail_indices = []
                for detail_col in detail_cols:
                    if detail_col in df_out.columns:
                        detail_indices.append(df_out.columns.get_loc(detail_col))
                
                if not detail_indices:
                    continue  # No detail columns to hide
                
                # Group consecutive columns together
                detail_indices.sort()
                start_col = min(detail_indices)
                end_col = max(detail_indices)
                
                # Convert to Excel column letters
                start_letter = _col_number_to_letter(start_col)
                end_letter = _col_number_to_letter(end_col)
                
                # Create collapsible group
                worksheet.set_column(f'{start_letter}:{end_letter}', None, None, {'level': 1, 'hidden': True})
            
            # Group all accuracy columns together (if they exist)
            accuracy_cols = [col for col in df_out.columns if col.endswith('_acc')]
            if accuracy_cols:
                acc_indices = [df_out.columns.get_loc(col) for col in accuracy_cols]
                acc_indices.sort()
                start_letter = _col_number_to_letter(min(acc_indices))
                end_letter = _col_number_to_letter(max(acc_indices))
                worksheet.set_column(f'{start_letter}:{end_letter}', None, None, {'level': 1, 'hidden': True})
            
            # Group the extra columns (MIR block: Increase through Description)
            # These start at n_cols + 2 and go to n_cols + 2 + len(extra_headers) - 1
            # But we want to KEEP visible: Final Forecast, Final Accy, and the first few extra columns
            # Let's hide from "Helper" column onward (which is typically column 7+ in extra_headers)
            
            # Find which extra columns to hide (everything after "LY Sales")
            extra_col_start = n_cols + 2  # First extra column
            columns_to_keep_visible = ["Increase", "AB Past Forecast", "Past Forecast", 
                                       "AB Current Forecast", "Current Forecast", "LY Sales"]
            
            first_hidden_extra = None
            for idx, (header, formula) in enumerate(extra_headers):
                if header not in columns_to_keep_visible:
                    first_hidden_extra = extra_col_start + idx
                    break
            
            if first_hidden_extra is not None:
                last_extra = extra_col_start + len(extra_headers) - 1
                start_letter = _col_number_to_letter(first_hidden_extra)
                end_letter = _col_number_to_letter(last_extra)
                worksheet.set_column(f'{start_letter}:{end_letter}', None, None, {'level': 1, 'hidden': True})

            # Show outline symbols
            worksheet.outline_settings(True, True, True, False)

            # ========== DYNAMIC FINAL FORECAST FORMULA SECTION ==========
            
            # Step 1: Detect which model columns are actually present
            model_yhat_columns = [
                "house_lager_yhat",
                "mind_melt_double_ipa_yhat",
                "heritage_blend_yhat",
                "west_coast_ipa_yhat",
                "creamy_nitro_yhat",
                "small_batch_classic_yhat",
                "light_hazy_yhat",
                "legacy_grand_reserve_yhat"
            ]
            
            present_model_cols = [col for col in model_yhat_columns if col in df_out.columns]
            
            # Step 2: Define column positions
            col_final = n_cols
            col_accy = n_cols + 1
            
            # Step 3: Write extra column headers first (so we can reference AB Current Forecast)
            for idx, (header, formula) in enumerate(extra_headers, start=n_cols + 2):
                worksheet.write(0, idx, header, hdr_extra)
                if header in numeric_headers:
                    worksheet.set_column(idx, idx, None, int_fmt)
            
            # Step 4: Build dynamic formulas for each row
            for row in range(1, n_rows + 1):
                excel_row = row + 1  # Excel is 1-indexed, add 1 for header
                
                # Build list of cell references for present models
                model_cell_refs = []
                for model_col in present_model_cols:
                    col_letter = _col_number_to_letter(df_out.columns.get_loc(model_col))
                    model_cell_refs.append(f"{col_letter}{excel_row}")
                
                # Find AB Current Forecast column (4th extra column: n_cols + 2 + 3)
                ab_current_col_num = n_cols + 2 + 3  # Index of "AB Current Forecast"
                ab_current_letter = _col_number_to_letter(ab_current_col_num)
                ab_current_ref = f"{ab_current_letter}{excel_row}"
                
                # Generate dynamic Final Forecast formula
                final_forecast_formula = build_dynamic_final_forecast_formula(
                    excel_row, 
                    model_cell_refs, 
                    ab_current_ref
                )
                
                # DEBUG: Print the formula for the first row
                if row == 1:
                    print(f"\n===== DEBUG: Final Forecast Formula for Row 2 =====")
                    print(f"Model cell refs: {model_cell_refs}")
                    print(f"AB Current ref: {ab_current_ref}")
                    print(f"Generated formula: {final_forecast_formula}")
                    print(f"Formula length: {len(final_forecast_formula)}")
                    print("="*60 + "\n")
                
                worksheet.write_formula(row, col_final, final_forecast_formula, int_fmt, "")
                
                # Final Accuracy formula (references Final Forecast column dynamically)
                final_forecast_letter = _col_number_to_letter(col_final)
                base_f2 = (
                    f"IF(ABS({final_forecast_letter}{excel_row}-$B{excel_row})/$B{excel_row}>1,"
                    f"ABS({final_forecast_letter}{excel_row}-$B{excel_row})/$B{excel_row}-1,"
                    f"1-ABS({final_forecast_letter}{excel_row}-$B{excel_row})/$B{excel_row})"
                )
                worksheet.write_formula(row, col_accy, f"=IFERROR({base_f2}, \"\")", pct_fmt, "")
                
                # Write extra column formulas
                for idx, (header, formula) in enumerate(extra_headers, start=n_cols + 2):
                    fmt = int_fmt if header in numeric_headers else None
                    worksheet.write_formula(row, idx, formula.format(row=excel_row), fmt)
            
            # Step 5: Write Final Forecast and Accuracy headers
            worksheet.write(0, col_final, "Final Forecast", hdr6)
            worksheet.write(0, col_accy, "Final Accy %", hdr6)
        
        # CRITICAL: Close the writer first, then modify the raw XML
        # This forces Excel to recalculate on open
        
        # Return Excel file
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=NEST_Forecasts_MultiModel.xlsx"}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast processing failed: {str(e)}")

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