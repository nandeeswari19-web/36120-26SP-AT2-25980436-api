FROM python:3.12-slim

# xgboost needs the OpenMP runtime, which the slim base image does not include.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

RUN pip install --no-cache-dir fastapi "uvicorn[standard]" "pandas>=3.0,<4" "scikit-learn>=1.9,<2" "xgboost>=3.4,<4" joblib

COPY app /code/app
COPY models /code/models

# Render provides $PORT at runtime; default to 8000 for local Docker runs.
ENV PORT=8000
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
