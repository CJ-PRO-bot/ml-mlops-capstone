# EcoGuard Lite (ML + MLOps)

## What it does
Air quality dataset → preprocessing → train classifier/regressor → track in MLflow → serve with FastAPI → UI with Streamlit → log predictions to Postgres.

## Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

Preprocess + Train
python src/preprocessing.py
python src/train_classical.py
python src/clustering.py

MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

Run API
uvicorn deployments.fastapi_app.main:app --reload --port 8000
Check: http://127.0.0.1:8000/docs

Run Streamlit
streamlit run deployments/streamlit_app/Home.py

Tests
$env:PYTHONPATH="."
pytest -q


---

# 8) Presentation (quick structure you can speak in 3–5 minutes)
Use this slide flow:

1. **Problem** (air quality prediction / monitoring)
2. **Data + preprocessing** (targets y_cls + y_reg, feature engineering time features)
3. **Training** (4 classifiers + 4 regressors, best selected)
4. **Tracking** (MLflow metrics + artifacts)
5. **Deployment** (FastAPI endpoints + Streamlit UI)
6. **Logging** (Postgres predictions table)
7. **Testing + CI/CD** (pytest + GitHub Actions)
8. **Next improvements** (model versioning, Docker compose deploy, drift monitoring)

For screenshots to include:
- MLflow experiments page (runs list)
- MLflow best run metrics
- FastAPI `/docs`
- Streamlit prediction screen
- Postgres `SELECT * FROM predictions LIMIT 5;`
- GitHub Actions green tick

# ML & MLOps Capstone Project

## Overview
This project demonstrates:
- Exploratory Data Analysis (EDA)
- Artificial Neural Networks (ANN)
- NLP tasks (Sentiment + QA)
- Transfer Learning (Feature Extraction vs Fine-Tuning)

## Structure
notebooks/
    01_EDA.ipynb
    02_ANN.ipynb
    03_NLP.ipynb
    04_transfer_learning.ipynb

artifacts/
    ann/
    nlp/
    transfer_learning/

## ML Components
- ANN with regularization and LR tuning
- NLP with TF-IDF baseline + Transformers
- Transfer Learning using DistilBERT

## How to Run
pip install -r requirements.txt