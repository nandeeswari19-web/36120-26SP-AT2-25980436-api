"""
Weather Intelligence API — AT2 (36120), student 25980436.

Serves two trained models:
  - Climate Comfort Index (CCI) regression, 3-day horizon
  - Weather Hazard Category (WHC) classification, 7-day horizon

Prediction endpoints look up the requested date in a pre-computed daily
feature table (built in the experimentation notebooks) and feed that row's
features to the corresponding trained model. This means predictions are
only available for dates present in the bundled historical dataset -- see
README.md for how to regenerate/extend it.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

APP_VERSION = "1.0.0"
GITHUB_REPO_URL = "https://github.com/nandeeswari19-web/36120-26SP-AT2-25980436-api"

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

WHC_LABELS = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk", 3: "Extreme Risk"}

# ---------------------------------------------------------------------
# Load models and feature lookup tables at startup
# ---------------------------------------------------------------------

app = FastAPI(title="Weather Intelligence API", version=APP_VERSION)

try:
    cci_model = joblib.load(MODELS_DIR / "comfort_climate" / "cci_model.joblib")
    whc_model = joblib.load(MODELS_DIR / "weather_hazard" / "whc_model.joblib")
except FileNotFoundError as e:
    cci_model = None
    whc_model = None
    _model_load_error = str(e)
else:
    _model_load_error = None

try:
    cci_features = pd.read_csv(MODELS_DIR / "comfort_climate" / "cci_feature_lookup.csv", parse_dates=["date"])
    cci_features = cci_features.set_index("date")
except FileNotFoundError:
    cci_features = None

try:
    whc_features = pd.read_csv(MODELS_DIR / "weather_hazard" / "whc_feature_lookup.csv", parse_dates=["date"])
    whc_features = whc_features.set_index("date")
except FileNotFoundError:
    whc_features = None

CCI_FEATURE_COLS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "cloud_cover",
    "precipitation", "temp_lag1", "temp_roll7", "cci_lag1", "month", "day_of_year",
]
WHC_FEATURE_COLS = [
    "precipitation", "wind_gusts_10m", "cloud_cover", "snowfall", "temperature_2m",
    "wind_gusts_lag1", "precip_roll3", "whi_lag1", "month", "day_of_year",
]


def _load_metrics(path: Path):
    """Return performance metrics saved by the experiment notebook, if present."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return "not available (metrics file missing)"


def _feature_importance(model, cols):
    """Feature importances from the trained model, keyed by feature name."""
    if model is None:
        return "not available (model not loaded)"
    return {c: round(float(v), 4) for c, v in zip(cols, model.feature_importances_)}


def _parse_date(date_str: str) -> datetime:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format '{date_str}'. Expected YYYY-MM-DD.",
        )


# ---------------------------------------------------------------------
# Root + health
# ---------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "project": "Weather Intelligence API",
        "description": "Machine learning services for Sydney weather forecasting (AT2, 36120)",
        "version": APP_VERSION,
        "github": GITHUB_REPO_URL,
        "prediction_targets": {
            "comfort_climate": "Climate Comfort Index (0-100), regression, 3 days ahead",
            "weather_hazard": "Weather Hazard Category (Low/Moderate/High/Extreme Risk), classification, 7 days ahead",
        },
        "endpoints": [
            "/",
            "/health",
            "/predict/index/comfort_climate",
            "/predict/category/weather_hazard",
            "/model-metadata",
        ],
    }


@app.get("/health")
def health():
    if cci_model is None or whc_model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": f"Model(s) failed to load: {_model_load_error}"},
        )
    return {"status": "healthy", "message": "Weather Intelligence API is running"}


# ---------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------

@app.get("/model-metadata")
def model_metadata():
    return [
        {
            "target": "Climate Comfort Index",
            "prediction_type": "Regression",
            "algorithm": "XGBoost Regressor",
            "forecast_horizon": "3 Days",
            "features": CCI_FEATURE_COLS,
            "feature_importance": _feature_importance(cci_model, CCI_FEATURE_COLS),
            "performance_metrics": _load_metrics(MODELS_DIR / "comfort_climate" / "cci_metrics.json"),
        },
        {
            "target": "Weather Hazard Category",
            "prediction_type": "Multiclass Classification",
            "algorithm": "XGBoost Classifier",
            "forecast_horizon": "7 Days",
            "features": WHC_FEATURE_COLS,
            "classes": list(WHC_LABELS.values()),
            "feature_importance": _feature_importance(whc_model, WHC_FEATURE_COLS),
            "performance_metrics": _load_metrics(MODELS_DIR / "weather_hazard" / "whc_metrics.json"),
        },
    ]


# ---------------------------------------------------------------------
# CCI prediction
# ---------------------------------------------------------------------

@app.get("/predict/index/comfort_climate")
def predict_comfort_climate(date: str = Query(..., description="Input date, YYYY-MM-DD")):
    if cci_model is None or cci_features is None:
        raise HTTPException(status_code=503, detail="CCI model or feature data not available.")

    input_date = _parse_date(date)

    if pd.Timestamp(input_date) not in cci_features.index:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No historical feature data available for {date}. "
                "This API serves predictions only for dates covered by the "
                "bundled historical dataset -- see /model-metadata and README."
            ),
        )

    row = cci_features.loc[[pd.Timestamp(input_date)], CCI_FEATURE_COLS]
    predicted_cci = float(cci_model.predict(row)[0])
    target_date = (input_date + timedelta(days=3)).strftime("%Y-%m-%d")

    return {
        "input_date": date,
        "predictions": {
            "comfort_climate": {
                target_date: round(predicted_cci, 1),
            }
        },
    }


# ---------------------------------------------------------------------
# WHC prediction
# ---------------------------------------------------------------------

@app.get("/predict/category/weather_hazard")
def predict_weather_hazard(date: str = Query(..., description="Input date, YYYY-MM-DD")):
    if whc_model is None or whc_features is None:
        raise HTTPException(status_code=503, detail="WHC model or feature data not available.")

    input_date = _parse_date(date)

    if pd.Timestamp(input_date) not in whc_features.index:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No historical feature data available for {date}. "
                "This API serves predictions only for dates covered by the "
                "bundled historical dataset -- see /model-metadata and README."
            ),
        )

    row = whc_features.loc[[pd.Timestamp(input_date)], WHC_FEATURE_COLS]
    predicted_class = int(whc_model.predict(row)[0])
    predicted_label = WHC_LABELS.get(predicted_class, "Unknown")
    target_date = (input_date + timedelta(days=7)).strftime("%Y-%m-%d")

    return {
        "input_date": date,
        "predictions": {
            "weather_hazard": {
                target_date: predicted_label,
            }
        },
    }
