# Weather Intelligence API

**Student:** Nandeeswari Kumaravelu
**Student ID:** 25980436
**Course:** 36120 — Advanced Machine Learning Application (AT2)

FastAPI service that serves two trained models for Sydney weather:

- **Climate Comfort Index (CCI)**: regression, 0-100, predicted 3 days ahead
- **Weather Hazard Category (WHC)**: multiclass classification (Low / Moderate / High / Extreme Risk), predicted exactly 7 days ahead

The models were trained in the companion experimentation repository
`36120-26SP-AT2-25980436-experiments`. Data comes from the Open-Meteo Historical
Weather API (Sydney, data before 2026 only).

**Deployed API (Render):** `https://three6120-26sp-at2-25980436-api.onrender.com`
**Interactive docs:** `https://three6120-26sp-at2-25980436-api.onrender.com/docs`

## Project structure

```
app/
  main.py                        FastAPI application (all 5 endpoints)
models/
  comfort_climate/
    cci_model.joblib             trained XGBoost regressor
    cci_feature_lookup.csv       daily features used to serve predictions
    cci_metrics.json             validation/test metrics from the notebook
  weather_hazard/
    whc_model.joblib             trained XGBoost classifier
    whc_feature_lookup.csv
    whc_metrics.json
Dockerfile
pyproject.toml
README.md
```

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Project overview, targets and endpoint list |
| GET | `/health` | Health check |
| GET | `/predict/index/comfort_climate?date=YYYY-MM-DD` | CCI prediction 3 days after `date` |
| GET | `/predict/category/weather_hazard?date=YYYY-MM-DD` | WHC prediction 7 days after `date` |
| GET | `/model-metadata` | Algorithm, features, feature importance and performance metrics |

### Example requests and responses

```
GET /predict/index/comfort_climate?date=2020-06-15
{"input_date":"2020-06-15","predictions":{"comfort_climate":{"2020-06-18":64.9}}}

GET /predict/category/weather_hazard?date=2020-06-15
{"input_date":"2020-06-15","predictions":{"weather_hazard":{"2020-06-22":"Low Risk"}}}
```

### Error handling

| Situation | Response |
|---|---|
| Date not in `YYYY-MM-DD` format | `400` with a clear message |
| Date outside the bundled historical data | `404` with a clear message |
| Missing `date` parameter | `422` (FastAPI validation) |
| Model files missing or failed to load | `/health` returns `503` with `"status": "unhealthy"` |

## Design notes and limitations

- **Single-horizon predictions.** The CCI model predicts one value exactly 3 days ahead
  (its training target), so the response contains that single date rather than separate
  values for days +1, +2 and +3. Producing all three would need a separate model per horizon.
- **Historical dates only.** Predictions are served by looking up the requested date in
  the daily feature table produced during experimentation. Only dates present in
  `models/*/*_feature_lookup.csv` are supported (the first and last valid dates are the
  first and last rows of each file). Serving arbitrary future dates would require fetching
  live weather from Open-Meteo inside the API.
- **Class imbalance.** Extreme Risk days are very rare in Sydney, so WHC performance on that
  class is limited. See `/model-metadata` and the report for per-class results.

## Run locally

Requires Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install fastapi "uvicorn[standard]" "pandas>=3.0,<4" "scikit-learn>=1.9,<2" "xgboost>=3.4,<4" joblib
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive documentation.

## Run with Docker

```bash
docker build -t weather-intelligence-api .
docker run -p 8000:8000 weather-intelligence-api
```

## Deploy to Render

1. Push this repository to GitHub (private, with the UTS staff granted admin access).
2. On [render.com](https://render.com) choose **New > Web Service** and connect this repository.
3. Set **Language / Environment** to **Docker**. Render detects the `Dockerfile` automatically.
4. Render provides the `PORT` environment variable; the Dockerfile already uses it.
5. Deploy, then test `<service-url>/health` first.

Free-tier services sleep when idle, so the first request after a pause can take up to a
minute to respond.

## Regenerating the model files

The files in `models/` are produced by the experimentation repository:

1. Run both notebooks there (each saves its model and metrics JSON in its final cells).
2. Run `python export_for_api.py` there to write the feature lookup tables.
3. Copy the six files listed under "Project structure" into this repository's `models/` folders.

The library versions in `pyproject.toml` and the `Dockerfile` match the environment the
models were trained in, so the model files load without version warnings.