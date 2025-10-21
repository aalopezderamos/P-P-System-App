import warnings
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import streamlit as st
import xlsxwriter  # required by pandas ExcelWriter engine

from catboost import CatBoostRegressor
from darts import TimeSeries
from darts.models import NBEATSModel
from lightgbm import LGBMRegressor
from prophet import Prophet
from pytorch_lightning.callbacks import EarlyStopping
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# Optional models (import if available)
try:
    from neuralprophet import NeuralProphet

    _HAS_NEURALPROPHET = True
except ImportError:
    _HAS_NEURALPROPHET = False

try:
    # xgboost requires sklearn runtime available
    from xgboost import XGBRegressor
    import sklearn  # noqa: F401

    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False


# ---------------------------------------------------------------------
# Torch 2.6 safe-load fix for NeuralProphet checkpoints
# PyTorch 2.6 defaults to `weights_only=True` which can break NP/Lightning
# when loading checkpoints. We allowlist NP config classes.
# ---------------------------------------------------------------------
_safe_globals_ctx = None
_NP_SAFE_GLOBALS = []
if _HAS_NEURALPROPHET:
    try:
        import torch.serialization as _ts
        from torch.serialization import safe_globals as _safe_globals_ctx
        from neuralprophet.configure import (
            ConfigAR,
            ConfigCountryHolidays,
            ConfigCustomSeasonality,
            ConfigEvents,
            ConfigLaggedRegressor,
            ConfigSeasonality,
            ConfigTrend,
        )

        _NP_SAFE_GLOBALS = [
            ConfigSeasonality,
            ConfigEvents,
            ConfigTrend,
            ConfigCountryHolidays,
            ConfigAR,
            ConfigLaggedRegressor,
            ConfigCustomSeasonality,
        ]
        _ts.add_safe_globals(_NP_SAFE_GLOBALS)
    except Exception:
        _safe_globals_ctx = None
        _NP_SAFE_GLOBALS = []


# =====================================================================
# Streamlit page config
# =====================================================================
st.set_page_config(page_title="Predict & Pour Multi‑Model Forecast App", layout="wide")
st.markdown(
    """
    <div class="app-header">
        <img src="https://i.imgur.com/Bf1hNE0.png" alt="Predict & Pour Logo" width="100" style="margin-right: 15px;">
        <h1>Predict & Pour Forecast App – Multi-Model Compare</h1>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
        /* ===== GENERAL APP THEME ===== */
        body, .stApp {
            background-color: #FFFFFF;
            color: #000000;
        }

        /* ===== HEADER BAR ===== */
        .app-header {
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #000000;
            padding: 15px;
            border-bottom: 3px solid #EBBB40;
            color: white;
        }
        .app-header h1 {
            color: #EBBB40 !important;
            margin: 0;
        }

        /* ===== BUTTONS ===== */
        div.stButton > button {
            background-color: #DCDCDC !important;
            color: black !important;
            border-radius: 6px;
            border: none;
            font-weight: bold;
        }
        div.stButton > button:hover {
            background-color: #d4a634 !important;
            color: white !important;
        }

        /* ===== SLIDERS ===== */
        .stSlider [role="slider"] {
            background-color: #EBBB40 !important;
        }

        /* ===== SIDEBAR ===== */
        [data-testid="stSidebar"] {
            background-color: #000000;
            color: #FFFFFF;
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
            color: #EBBB40 !important;
        }

        /* ===== HEADINGS ===== */
        h1, h2, h3 {
            color: #EBBB40;
        }

        /* ===== DATAFRAME ===== */
        .stDataFrame {
            border: 2px solid #EBBB40;
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
# ---------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------
WEEK_FREQ = "W-SAT"
PI_Z_90 = 1.645  # ~90% prediction interval z-score

# ---------------------------------------------------------------------
# Session state keys for run / abort
# ---------------------------------------------------------------------
if "run_forecast" not in st.session_state:
    st.session_state["run_forecast"] = False
if "abort_forecast" not in st.session_state:
    st.session_state["abort_forecast"] = False


# =====================================================================
# Helpers
# =====================================================================
@st.cache_data
def get_custom_holidays() -> pd.DataFrame:
    """
    Build a DataFrame of custom multi-day events/holidays in Prophet format.
    """
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
            records.append(
                {
                    "holiday": str(name),
                    "ds": pd.Timestamp(date).normalize(),
                    "lower_window": -3,
                    "upper_window": 3,
                }
            )
    return pd.DataFrame(records)


def _add_event_flags(
    df_dates: pd.DataFrame, holidays_df: pd.DataFrame, event_names=None
) -> pd.DataFrame:
    """Create 0/1 columns for each event listed in holidays_df."""
    if event_names is None:
        event_names = holidays_df["holiday"].astype(str).unique().tolist()
    df_dates = df_dates.copy()
    for h in event_names:
        mask = df_dates["ds"].isin(holidays_df.loc[holidays_df["holiday"] == h, "ds"])
        df_dates[h] = mask.astype(int)
    return df_dates


def safe_accuracy(y, yhat) -> float:
    """Return simple accuracy proxy (1 - abs(pct error)); NaN when undefined."""
    if pd.isna(y) or y == 0 or pd.isna(yhat):
        return np.nan
    return round(1 - abs((y - yhat) / y), 3)


def next_saturday_from(date: pd.Timestamp) -> pd.Timestamp:
    """Return the next Saturday after a given date."""
    # Monday=0 .. Saturday=5
    days_ahead = (5 - date.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7  # ensure *next* Saturday
    return date + pd.Timedelta(days=days_ahead)


def build_future_saturdays(last_date: pd.Timestamp, horizon_weeks: int) -> pd.DatetimeIndex:
    """Build a weekly DateTimeIndex on Saturdays for a forecast horizon."""
    first_sat = next_saturday_from(last_date)
    return pd.date_range(start=first_sat, periods=horizon_weeks, freq=WEEK_FREQ)


# =====================================================================
# Models
# =====================================================================
# ------------------------------ Prophet --------------------------------
def run_prophet(
    df_sku: pd.DataFrame, holidays_df: pd.DataFrame, horizon_weeks: int
) -> pd.DataFrame:
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


# --------------------------- NeuralProphet ------------------------------
def run_neuralprophet(
    df_sku: pd.DataFrame, holidays_df: pd.DataFrame, horizon_weeks: int, debug: bool = False
) -> pd.DataFrame:
    """
    NeuralProphet wrapper with:
    - event one-hot flags
    - torch 2.6 safe-load guards
    - ensures 'y' exists in full_future (NP may require it)
    """
    if not _HAS_NEURALPROPHET:
        raise ImportError("neuralprophet is not installed. pip install neuralprophet")

    event_names = holidays_df["holiday"].astype(str).unique().tolist()

    m = NeuralProphet(weekly_seasonality=True, yearly_seasonality=True, quantiles=[0.05, 0.95])
    for ev in event_names:
        m.add_events(ev, lower_window=-3, upper_window=3)

    # ---------- Train ----------
    train_df = df_sku[["ds", "y"]].copy()
    train_df = _add_event_flags(train_df, holidays_df, event_names)

    missing_train = [ev for ev in event_names if ev not in train_df.columns]
    if missing_train:
        raise ValueError(f"Missing event cols in train_df: {missing_train}")

    def _fit_model():
        try:
            return m.fit(
                train_df, freq=WEEK_FREQ, trainer_config={"enable_checkpointing": False, "logger": False}
            )
        except TypeError:
            return m.fit(train_df, freq=WEEK_FREQ)

    if "_safe_globals_ctx" in globals() and _safe_globals_ctx:
        with _safe_globals_ctx(_NP_SAFE_GLOBALS):
            _fit_model()
    else:
        _fit_model()

    # ---------- Future ----------
    last_date = df_sku["ds"].max()
    future_sats = build_future_saturdays(last_date, horizon_weeks)
    future = pd.DataFrame({"ds": future_sats})
    future = _add_event_flags(future, holidays_df, event_names)

    full_future = (
        pd.concat([train_df[["ds"] + event_names], future[["ds"] + event_names]], ignore_index=True)
        .drop_duplicates("ds")
        .sort_values("ds")
    )

    # Ensure 'y' column exists for NP
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
    # Fallbacks keep behavior stable (NaN when NP did not produce quantiles)
    out["neural_yhat_lower"] = forecast.get("yhat1 5.0%", np.nan)
    out["neural_yhat_upper"] = forecast.get("yhat1 95.0%", np.nan)
    return out


# ------------------------------- SARIMAX --------------------------------
def run_sarimax(df_sku: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    ts = df_sku.set_index("ds")["y"].asfreq(WEEK_FREQ)
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 52)

    model = SARIMAX(
        ts, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False
    )
    res = model.fit(disp=False)

    fcst = res.get_forecast(steps=horizon_weeks)
    pred_mean = fcst.predicted_mean
    conf = fcst.conf_int(alpha=0.10)

    # Normalize conf int column names
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


# ----------------------- Generic feature builders -----------------------
def make_lag_features(
    y: pd.Series, lags=(1, 2, 3, 4, 5, 6, 7, 12, 26, 52), roll_windows=(4, 12)
) -> pd.DataFrame:
    df_feat = pd.DataFrame({"y": y})
    for l in lags:
        df_feat[f"lag_{l}"] = y.shift(l)
    for w in roll_windows:
        df_feat[f"roll_mean_{w}"] = y.shift(1).rolling(window=w).mean()
    return df_feat


def add_date_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "weekofyear": idx.isocalendar().week.astype(int),
            "month": idx.month,
            "quarter": idx.quarter,
            "year": idx.year,
        },
        index=idx,
    )


# ------------------------------ XGBoost ---------------------------------
def run_xgboost(df_sku: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    if not _HAS_XGBOOST:
        raise ImportError("xgboost / scikit-learn not installed. pip install xgboost scikit-learn")

    ts = df_sku.set_index("ds")["y"].asfreq(WEEK_FREQ)
    features = make_lag_features(ts)
    date_feats = add_date_features(features.index)
    X = pd.concat([features.drop(columns=["y"]), date_feats], axis=1)
    y = features["y"]

    mask = X.notna().all(axis=1)
    X_train, y_train = X[mask], y[mask]

    model = XGBRegressor(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
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
        rows.append(
            {
                "ds": ds,
                "xgb_yhat": pred,
                "xgb_yhat_lower": pred - PI_Z_90 * resid_std,
                "xgb_yhat_upper": pred + PI_Z_90 * resid_std,
            }
        )
        full_y.loc[ds] = pred

    hist_pred = model.predict(X)
    hist_df = pd.DataFrame({"ds": X.index, "xgb_yhat": hist_pred})
    hist_df["xgb_yhat_lower"] = np.nan
    hist_df["xgb_yhat_upper"] = np.nan

    fut_df = pd.DataFrame(rows)
    return pd.concat([hist_df, fut_df], ignore_index=True)


# ------------------------------ CatBoost --------------------------------
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
        rows.append(
            {
                "ds": ds,
                "catboost_yhat": pred,
                "catboost_yhat_lower": pred - PI_Z_90 * resid_std,
                "catboost_yhat_upper": pred + PI_Z_90 * resid_std,
            }
        )
        full_y.loc[ds] = pred

    hist_pred = model.predict(X)
    hist_df = pd.DataFrame({"ds": X.index, "catboost_yhat": hist_pred})
    hist_df["catboost_yhat_lower"] = np.nan
    hist_df["catboost_yhat_upper"] = np.nan

    fut_df = pd.DataFrame(rows)
    return pd.concat([hist_df, fut_df], ignore_index=True)


# --------------------------- Holt-Winters (ES) --------------------------
def run_holtwinters(df_sku: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    """
    Holt-Winters Exponential Smoothing Forecast (weekly seasonality = 52).
    """
    ts = df_sku.set_index("ds")["y"].asfreq(WEEK_FREQ)

    model = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=52).fit(optimized=True)

    # In-sample fitted values
    hist_df = pd.DataFrame({"ds": ts.index, "holt_yhat": model.fittedvalues})
    hist_df["holt_yhat_lower"] = np.nan
    hist_df["holt_yhat_upper"] = np.nan

    # Forecast future
    future_index = pd.date_range(start=ts.index.max() + pd.Timedelta(weeks=1), periods=horizon_weeks, freq=WEEK_FREQ)
    forecast = model.forecast(horizon_weeks)
    resid_std = np.std(model.resid)

    fut_df = pd.DataFrame(
        {
            "ds": future_index,
            "holt_yhat": forecast.values,
            # No built-in intervals in HW — approximate via ±1.645 * resid_std
            "holt_yhat_lower": forecast.values - PI_Z_90 * resid_std,
            "holt_yhat_upper": forecast.values + PI_Z_90 * resid_std,
        }
    )

    return pd.concat([hist_df, fut_df], ignore_index=True)


# ------------------------------ LightGBM --------------------------------
def run_lightgbm(df_sku: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    ts = df_sku.set_index("ds")["y"].asfreq(WEEK_FREQ)
    features = make_lag_features(ts)
    date_feats = add_date_features(features.index)
    X = pd.concat([features.drop(columns=["y"]), date_feats], axis=1)
    y = features["y"]

    mask = X.notna().all(axis=1)
    X_train, y_train = X[mask], y[mask]

    model = LGBMRegressor(
        n_estimators=600, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42
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
        rows.append(
            {
                "ds": ds,
                "lgbm_yhat": pred,
                "lgbm_yhat_lower": pred - PI_Z_90 * resid_std,
                "lgbm_yhat_upper": pred + PI_Z_90 * resid_std,
            }
        )
        full_y.loc[ds] = pred

    hist_pred = model.predict(X)
    hist_df = pd.DataFrame({"ds": X.index, "lgbm_yhat": hist_pred})
    hist_df["lgbm_yhat_lower"] = np.nan
    hist_df["lgbm_yhat_upper"] = np.nan

    fut_df = pd.DataFrame(rows)
    return pd.concat([hist_df, fut_df], ignore_index=True)


# -------------------------------- N-BEATS --------------------------------
def run_nbeats(df_sku: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    series = TimeSeries.from_dataframe(df_sku[["ds", "y"]], "ds", "y", freq=WEEK_FREQ)

    n = len(series)
    desired_in = 26
    out_len = max(1, int(horizon_weeks))
    min_in = 8  # don’t go lower than this to keep some context
    in_len = min(desired_in, max(min_in, n - out_len - 1))

    if n < in_len + out_len + 1:
        # Not enough data to train at all → return empty forecast frame for this SKU
        return pd.DataFrame(columns=["ds", "nbeats_yhat", "nbeats_yhat_lower", "nbeats_yhat_upper"])

    # Decide whether we can afford a validation split (need room for batches)
    can_val = n >= (in_len + out_len + 20)
    val_series = None
    early_cb = None
    pl_kwargs = {"enable_checkpointing": False, "logger": False}

    if can_val:
        # 70/30 split works; ensures val has multiple batches
        _, val_series = series.split_before(0.70)
        early_cb = EarlyStopping(monitor="val_loss", patience=5, mode="min")
        pl_kwargs["callbacks"] = [early_cb]
    else:
        early_cb = EarlyStopping(monitor="train_loss", patience=5, mode="min")
        pl_kwargs["callbacks"] = [early_cb]

    # Batch size scaled to series length (keeps optimizer steps sensible)
    bs = max(4, min(64, n // 4))

    model = NBEATSModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        n_epochs=50,  # much cheaper than 300
        batch_size=bs,
        random_state=42,
        pl_trainer_kwargs=pl_kwargs,
    )

    # Keep PyTorch to a single thread to avoid CPU thrash when parallelizing SKUs
    try:
        import torch

        prev_threads = torch.get_num_threads()
        torch.set_num_threads(1)
    except Exception:
        prev_threads = None

    # Fit (with or without validation, depending on can_val)
    model.fit(series, verbose=False, val_series=val_series)

    # Restore torch threads
    if prev_threads is not None:
        try:
            import torch

            torch.set_num_threads(prev_threads)
        except Exception:
            pass

    # Predict future (point forecast for speed)
    forecast = model.predict(out_len)
    df_pred = forecast.pd_dataframe().reset_index().rename(columns={"index": "ds", "y": "nbeats_yhat"})
    df_pred["nbeats_yhat_lower"] = np.nan
    df_pred["nbeats_yhat_upper"] = np.nan

    # Historical preds only if there’s enough room; otherwise skip
    hist_frames = []
    try:
        if n >= in_len + out_len + 5:
            hist = (
                model.historical_forecasts(series, start=0.8, forecast_horizon=1, verbose=False)
                .pd_dataframe()
                .reset_index()
            )
            hist.rename(columns={"index": "ds", "y": "nbeats_yhat"}, inplace=True)
            hist["nbeats_yhat_lower"] = np.nan
            hist["nbeats_yhat_upper"] = np.nan
            hist_frames.append(hist)
    except Exception:
        # If darts complains on short series, just skip historical forecasts
        pass

    if hist_frames:
        return pd.concat([hist_frames[0], df_pred], ignore_index=True)
    return df_pred


# =====================================================================
# UI
# =====================================================================
def ui_controls():
    """
    Render all Streamlit UI controls and return their values.
    Layout: General controls on top, a bordered Models section (2-wide x 4-long),
    and run/stop controls. Return values/order remain identical to the original.
    """
    # Upload
    uploaded = st.file_uploader("📤 Upload your 'NEST Forecast Template.csv' file", type=["csv"])

    # General controls
    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        horizon_weeks = st.slider("Forecast horizon (weeks)", min_value=4, max_value=52, value=12, step=1)
    with colB:
        min_points = st.slider("Minimum rows per SKU", min_value=20, max_value=300, value=50, step=5)
    with colC:
        show_debug = st.checkbox("Show debug columns", value=False)

    st.markdown("")

    # === Models section (3 columns side-by-side) ===
    st.markdown("---")
    st.markdown("<h2 style='color:#EBBB40;'>🧠 Forecast Models</h2>", unsafe_allow_html=True)

    col_classic, col_ml, col_dl = st.columns(3, gap="large")

    # --- Classical Statistical Models ---
    with col_classic:
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.markdown("<h3>📈 Classic Statistical</h3>", unsafe_allow_html=True)
        run_prophet_flag = st.checkbox(
            "Prophet", value=True,
            help="A decomposable time series model that captures trend, seasonality, and holidays."
        )
        run_sarimax_flag = st.checkbox(
            "SARIMAX", value=True,
            help="Seasonal ARIMA with exogenous regressors for trend & seasonality."
        )
        run_holt_flag = st.checkbox(
            "Holt-Winters", value=False,
            help="Exponential smoothing that handles level, trend, and seasonality."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Machine Learning Models ---
    with col_ml:
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.markdown("<h3>🤖 Machine Learning</h3>", unsafe_allow_html=True)
        run_xgb_flag = st.checkbox(
            "XGBoost", value=False, disabled=not _HAS_XGBOOST,
            help="Gradient-boosted trees combining many weak learners."
        )
        run_catboost_flag = st.checkbox(
            "CatBoost", value=False,
            help="Boosting with strong handling of categorical features."
        )
        run_lgbm_flag = st.checkbox(
            "LightGBM", value=False,
            help="Fast, memory-efficient gradient boosting for large datasets."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Deep Learning Models ---
    with col_dl:
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.markdown("<h3>🧬 Deep Learning</h3>", unsafe_allow_html=True)
        run_neural_flag = st.checkbox(
            "NeuralProphet", value=False, disabled=not _HAS_NEURALPROPHET,
            help="Neural extension of Prophet for nonlinear patterns."
        )
        run_nbeats_flag = st.checkbox(
            "N-BEATS", value=False,
            help="Neural basis expansions for complex time series."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Run/Stop controls
    start_clicked = st.button("🚀 Start forecasting", disabled=not bool(uploaded))
    stop_placeholder = st.empty()

    if start_clicked:
        st.session_state["run_forecast"] = True
        st.session_state["abort_forecast"] = False

    # Show stop button only while running
    if st.session_state.get("run_forecast", False) and not st.session_state.get("abort_forecast", False):
        if stop_placeholder.button("🛑 Stop forecasting", type="secondary"):
            st.session_state["abort_forecast"] = True

    # Return values — keep identical order to original
    return (
        uploaded,
        horizon_weeks,
        min_points,
        run_prophet_flag,
        run_neural_flag,
        run_sarimax_flag,
        run_xgb_flag,
        run_catboost_flag,
        run_holt_flag,
        run_lgbm_flag,
        run_nbeats_flag,
        show_debug,
        start_clicked,
        stop_placeholder,
    )


# =====================================================================
# Main
# =====================================================================
def main():
    # Render UI and retrieve control values
    (
        uploaded,
        horizon_weeks,
        min_points,
        run_prophet_flag,
        run_neural_flag,
        run_sarimax_flag,
        run_xgb_flag,
        run_catboost_flag,
        run_holt_flag,
        run_lgbm_flag,
        run_nbeats_flag,
        show_debug,
        start_clicked,
        stop_placeholder,  # noqa: F841
    ) = ui_controls()

    # ----------------------------------------------
    # Main processing block (only when run flag true)
    # ----------------------------------------------
    if uploaded and st.session_state["run_forecast"] and not st.session_state["abort_forecast"]:
        with st.spinner("Processing your forecast..."):
            df = pd.read_csv(uploaded)
            required_cols = {"sku_id", "ds", "y"}
            if not required_cols.issubset(df.columns):
                st.error(f"❌ Missing columns: {required_cols - set(df.columns)}")
                st.session_state["run_forecast"] = False
                st.stop()

            df["ds"] = pd.to_datetime(df["ds"])
            holidays_df = get_custom_holidays()
            holidays_df["ds"] = pd.to_datetime(holidays_df["ds"])

            sku_list = df["sku_id"].unique()
            total = len(sku_list)

            progress = st.progress(0.0)
            status = st.empty()

            detailed_rows = []
            aborted = False

            for idx, (sku, df_sku) in enumerate(df.groupby("sku_id"), start=1):
                # Check abort flag each loop
                if st.session_state.get("abort_forecast"):
                    aborted = True
                    break

                progress.progress(idx / total)
                status.text(f"Processing {idx}/{total} → {sku}")

                if len(df_sku) < min_points:
                    continue

                try:
                    base = pd.DataFrame({"ds": df_sku["ds"].unique()}).sort_values("ds")
                    base = base.merge(df_sku[["ds", "y"]], on="ds", how="left")

                    model_futures = {}
                    with ThreadPoolExecutor(max_workers=6) as executor:
                        if run_prophet_flag:
                            model_futures["prophet"] = executor.submit(run_prophet, df_sku, holidays_df, horizon_weeks)
                        if run_neural_flag:
                            model_futures["neural"] = executor.submit(
                                run_neuralprophet, df_sku, holidays_df, horizon_weeks
                            )
                        if run_sarimax_flag:
                            model_futures["sarimax"] = executor.submit(run_sarimax, df_sku, horizon_weeks)
                        if run_xgb_flag:
                            model_futures["xgb"] = executor.submit(run_xgboost, df_sku, horizon_weeks)
                        if run_catboost_flag:
                            model_futures["catboost"] = executor.submit(run_catboost, df_sku, horizon_weeks)
                        if run_holt_flag:
                            model_futures["holt"] = executor.submit(run_holtwinters, df_sku, horizon_weeks)
                        if run_lgbm_flag:
                            model_futures["lgbm"] = executor.submit(run_lightgbm, df_sku, horizon_weeks)
                        if run_nbeats_flag:
                            model_futures["nbeats"] = executor.submit(run_nbeats, df_sku, horizon_weeks)

                    # Merge model outputs
                    for model_name, future in model_futures.items():
                        try:
                            result_df = future.result()
                            base = base.merge(result_df, on="ds", how="outer", validate="one_to_one")
                        except Exception as e:
                            st.warning(f"{model_name} failed for {sku}: {e}")

                    # Compute model-specific accuracy
                    for model_prefix in ["prophet", "neural", "sarimax", "xgb", "catboost", "holt", "lgbm", "nbeats"]:
                        pred_col = f"{model_prefix}_yhat"
                        if pred_col in base.columns:
                            base[f"{model_prefix}_acc"] = base.apply(
                                lambda r: safe_accuracy(r.get("y"), r.get(pred_col)), axis=1
                            )

                    base["sku_id"] = sku
                    detailed_rows.append(base)

                except Exception as e:
                    st.warning(f"⚠️ Error processing {sku}: {e}")
                    continue

            # After loop
            if aborted:
                st.warning("🚫 Forecasting aborted by user.")
                st.session_state["run_forecast"] = False
                st.stop()

            if not detailed_rows:
                st.error("❌ No forecasts were generated. Check your file or adjust filters.")
                st.session_state["run_forecast"] = False
                st.stop()

            df_out = pd.concat(detailed_rows, ignore_index=True).sort_values(["sku_id", "ds"])

            st.success("✅ Forecasting complete!")
            st.subheader("📊 Forecast Preview")
            preview_cols = [c for c in df_out.columns if (not c.endswith("_acc") or show_debug)]
            st.dataframe(df_out[preview_cols].head(50))

            # --------------------------- Excel export ---------------------------
            output = BytesIO()
            with pd.ExcelWriter(
                output, engine="xlsxwriter", date_format="mm/dd/yyyy", datetime_format="mm/dd/yyyy"
            ) as writer:
                df_out.to_excel(writer, index=False, sheet_name="forecasts")
                workbook = writer.book
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

                # New columns (kept formulas as-is)
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

                # Freeze panes and set uniform widths (existing + final + extras)
                worksheet.freeze_panes(1, 0)
                num_total_cols = n_cols + 2 + len(extra_headers)
                for col in range(num_total_cols):
                    px = 75
                    width = (px - 5) / 7
                    worksheet.set_column(col, col, width)

                # Existing headers formatting
                col_idx = {name: i for i, name in enumerate(df_out.columns)}
                group_map = {
                    hdr1: ["ds", "y", "sku_id"],
                    hdr2: [
                        "prophet_yhat",
                        "prophet_yhat_lower",
                        "prophet_yhat_upper",
                        "trend",
                        "weekly",
                        "yearly",
                        "holidays",
                        "prophet_acc",
                    ],
                    hdr3: ["neural_yhat", "neural_yhat_lower", "neural_yhat_upper", "neural_acc"],
                    hdr4: ["sarimax_yhat", "sarimax_yhat_lower", "sarimax_yhat_upper", "sarimax_acc"],
                    hdr5: ["xgb_yhat", "xgb_yhat_lower", "xgb_yhat_upper", "xgb_acc"],
                    hdr7: ["catboost_yhat", "catboost_yhat_lower", "catboost_yhat_upper", "catboost_acc"],
                    hdr8: ["holt_yhat", "holt_yhat_lower", "holt_yhat_upper", "holt_acc"],
                    hdr9: ["lgbm_yhat", "lgbm_yhat_lower", "lgbm_yhat_upper", "lgbm_acc"],
                    hdr10: ["nbeats_yhat", "nbeats_yhat_lower", "nbeats_yhat_upper", "nbeats_acc"],
                }
                for fmt, names in group_map.items():
                    for name in names:
                        if name not in col_idx:
                            continue
                        c = col_idx[name]
                        worksheet.write(0, c, name, fmt)
                        if name == "ds":
                            worksheet.set_column(c, c, 11, date_fmt)
                        elif name.endswith("_acc"):
                            worksheet.set_column(c, c, 11, pct_fmt)
                        else:
                            worksheet.set_column(c, c, 11, int_fmt)
                # --------------------------- Column Grouping ---------------------------
                worksheet.set_column('D:I', None, None, {'level': 1, 'hidden': True})   # Prophet block
                worksheet.set_column('K:L', None, None, {'level': 1, 'hidden': True})   # NeuralProphet block
                worksheet.set_column('N:O', None, None, {'level': 1, 'hidden': True})   # SARIMAX block
                worksheet.set_column('Q:R', None, None, {'level': 1, 'hidden': True})   # XGBoost block
                worksheet.set_column('T:U', None, None, {'level': 1, 'hidden': True})   # CatBoost block
                worksheet.set_column('W:X', None, None, {'level': 1, 'hidden': True})   # Holt-Winters block
                worksheet.set_column('Z:AA', None, None, {'level': 1, 'hidden': True})  # LightGBM block
                worksheet.set_column('AC:AD', None, None, {'level': 1, 'hidden': True}) # N-BEATS block
                worksheet.set_column('AE:AL', None, None, {'level': 1, 'hidden': True}) # Accuracy block
                worksheet.set_column('AP:AZ', None, None, {'level': 1, 'hidden': True}) # MIR Block

                # Show outline symbols
                worksheet.outline_settings(True, True, True, False)

                # Final Forecast columns + formulas
                col_final = n_cols
                col_accy = n_cols + 1
                worksheet.write(0, col_final, "Final Forecast", hdr6)
                worksheet.write(0, col_accy, "Final Accy %", hdr6)

                for row in range(1, n_rows + 1):
                    base_f1 = (
                        "LET("
                        "vals, HSTACK(C{r},J{r},M{r},P{r},S{r},V{r},Y{r},AB{r}),"
                        "arr, TOCOL(CHOOSE({{1,2,3,4,5,6,7,8}}, "
                        "C{r},J{r},M{r},P{r},S{r},V{r},Y{r},AB{r})),"
                        "mu, AVERAGE(arr),"
                        "AVERAGE(TAKE(SORTBY(arr, ABS(arr-mu), 1), 5))"
                        ")"
                    ).format(r=row + 1)
                    worksheet.write_formula(row, col_final, f"=IFERROR({base_f1}, \"\")", int_fmt)

                    base_f2 = (
                        "IF(ABS(X{r}-$B{r})/$B{r}>1,"
                        "ABS(X{r}-$B{r})/$B{r}-1,"
                        "1-ABS(X{r}-$B{r})/$B{r})"
                    ).format(r=row + 1)
                    worksheet.write_formula(row, col_accy, f"=IFERROR({base_f2}, \"\")", pct_fmt)

                # Extra columns: headers + formulas
                for idx, (header, formula) in enumerate(extra_headers, start=n_cols + 2):
                    worksheet.write(0, idx, header, hdr_extra)
                    if header in numeric_headers:
                        worksheet.set_column(idx, idx, None, int_fmt)
                    for row in range(1, n_rows + 1):
                        fmt = int_fmt if header in numeric_headers else None
                        worksheet.write_formula(row, idx, formula.format(row=row + 1), fmt)

            st.download_button(
                label="📥 Download Forecast Excel",
                data=output,
                file_name="NEST_Forecasts_MultiModel.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.session_state["run_forecast"] = False

    elif uploaded and not st.session_state["run_forecast"]:
        st.info("✅ File uploaded. Click **Start forecasting** when you're ready.")
    else:
        st.info("👆 Upload your CSV to get started.")


if __name__ == "__main__":
    main()