![CI](https://github.com/CJ-PRO-bot/ml-mlops-capstone/actions/workflows/ci.yml/badge.svg)

# EcoGuard Lite (ML + MLOps)

## What it does
Air quality dataset → preprocessing → train models → track experiments in MLflow → serve predictions with FastAPI → UI with Streamlit → log predictions to Postgres.

## Repository structure
- `src/` — preprocessing, training, inference utilities
- `deployments/` — FastAPI, Flask, Django, Streamlit apps
- `notebooks/` — EDA, ANN, NLP, transfer learning experiments
- `tests/` — pytest tests
- `docker-compose.yml` — local Postgres + pgAdmin
- `.github/workflows/ci.yml` — CI pipeline (ruff + pytest)

## Notebooks delivered (ML requirements)
- `notebooks/01_EDA.ipynb` — EDA on breast cancer dataset
- `notebooks/02_ANN.ipynb` — ANN experiments (dropout/L2/LR)
- `notebooks/03_NLP.ipynb` — NLP tasks (sentiment + QA) with metrics
- `notebooks/04_transfer_learning.ipynb` — transfer learning (freeze vs fine-tune) with comparison

## Quickstart (local)
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
Preprocess + Train (examples)
python src/preprocessing.py
python src/train_classical.py
python src/train_ann.py
Run MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
Run FastAPI
uvicorn deployments.fastapi_app.main:app --reload --port 8000
# Open: http://127.0.0.1:8000/docs
Run Streamlit
streamlit run deployments/streamlit_app/Home.py
Run tests
pytest -q
Docker (Postgres + pgAdmin)
docker compose up --build

pgAdmin: http://localhost:5050

Postgres exposed on: localhost:5433