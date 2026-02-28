![CI](https://github.com/CJ-PRO-bot/ml-mlops-capstone/actions/workflows/ci.yml/badge.svg)

# EcoGuard Lite (ML + MLOps)

## What it does
Air quality dataset → preprocessing → train models → track experiments in MLflow → serve predictions with FastAPI/Flask/Django → UI with Streamlit → (optional) log predictions to Postgres.

## Repository structure
- `src/` — preprocessing, training, clustering, inference utilities
- `deployments/` — FastAPI, Flask, Django, Streamlit apps
- `notebooks/` — EDA/ANN/NLP/transfer learning experiments (Colab-exported)
- `tests/` — pytest tests
- `artifacts/` — trained models + reports + summaries
- `docker-compose.yml` — local Postgres + pgAdmin + MLflow
- `.github/workflows/ci.yml` — CI pipeline (ruff + pytest)

## Note about notebooks (why different dataset)
Some notebooks were trained on Google Colab due to limited local compute / no CUDA.
They are included as supporting evidence of Deep Learning + NLP work.
The core end-to-end pipeline code (in `src/` + `deployments/`) uses the processed AirQuality dataset.

## Quickstart (local)
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-dev.txt

# Preprocess + Train
python src/preprocessing.py
python src/train_classical.py
python src/train_ann.py
python src/clustering.py

# Run MLflow UI (optional if using docker-compose MLflow)
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Run FastAPI
uvicorn deployments.fastapi_app.main:app --reload --port 8000
# Open: http://127.0.0.1:8000/docs

# Run Streamlit
streamlit run deployments/streamlit_app/Home.py

# Run tests
pytest -q

# Docker services (Postgres + pgAdmin + MLflow)
docker compose up --build

# pgAdmin: http://localhost:5050
# Postgres: localhost:5433