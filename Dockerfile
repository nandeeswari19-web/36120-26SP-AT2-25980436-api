FROM python:3.12-slim

WORKDIR /code

COPY pyproject.toml /code/pyproject.toml
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" "pandas>=3.0,<4" "scikit-learn>=1.9,<2" "xgboost>=3.4,<4" joblib

COPY app /code/app
COPY models /code/models

# Render provides $PORT at runtime; default to 8000 for local testing.
ENV PORT=8000
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
