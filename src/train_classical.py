from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, StackingClassifier
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_processed.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

EXPERIMENT_NAME = "ecoguard-lite-phase1"

MLFLOW_DB = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
mlflow.set_tracking_uri(MLFLOW_DB)
mlflow.set_registry_uri(MLFLOW_DB)
def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def evaluate_classification(y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    # ROC-AUC only if probabilities are provided and multi-class compatible
    if y_proba is not None:
        try:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
        except Exception:
            pass
    return metrics


def evaluate_regression(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    return {
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }

def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing processed dataset: {DATA_PATH}. Run preprocessing first.")

    df = pd.read_csv(DATA_PATH)
    # Features and targets
    y_cls = df["y_cls"].astype(int)
    y_reg = df["y_reg"].astype(float)
    X = df.drop(columns=["y_cls", "y_reg"])

    numeric_features = list(X.columns)

    # Shared preprocessing pipeline (lightweight)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ],
        remainder="drop",
    )

    # Split once for classification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    # Split once for regression (same X but different y)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    # ---------------------------
    # CLASSIFICATION MODELS (4)
    # ---------------------------
    cls_models = {
        "logreg": LogisticRegression(max_iter=2000, n_jobs=None),
        "svm_rbf": SVC(kernel="rbf", probability=True),
        "rf": RandomForestClassifier(n_estimators=250, random_state=42),
    }

    # Stacking ensemble (4th)
    stack = StackingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=2000)),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ],
        final_estimator=LogisticRegression(max_iter=2000),
        passthrough=False,
    )
    cls_models["stacking"] = stack

    best_cls_name = None
    best_f1 = -1.0
    best_cls_pipeline = None

    for name, model in cls_models.items():
        with mlflow.start_run(run_name=f"classification_{name}"):
            pipe = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", model),
                ]
            )

            mlflow.log_param("task", "classification")
            mlflow.log_param("model_name", name)

            start = time.time()
            pipe.fit(X_train, y_train)
            fit_ms = int((time.time() - start) * 1000)

            y_pred = pipe.predict(X_test)
            y_proba = None
            if hasattr(pipe.named_steps["model"], "predict_proba"):
                y_proba = pipe.predict_proba(X_test)

            metrics = evaluate_classification(y_test, y_pred, y_proba=y_proba)
            metrics["fit_ms"] = fit_ms
            mlflow.log_metrics(metrics)

            cm = confusion_matrix(y_test, y_pred)
            cm_path = ARTIFACTS_DIR / f"cm_{name}.txt"
            save_text(cm_path, np.array2string(cm))
            mlflow.log_artifact(str(cm_path))

            rep = classification_report(y_test, y_pred, zero_division=0)
            rep_path = ARTIFACTS_DIR / f"classification_report_{name}.txt"
            save_text(rep_path, rep)
            mlflow.log_artifact(str(rep_path))

            # Track best
            if metrics["f1_macro"] > best_f1:
                best_f1 = metrics["f1_macro"]
                best_cls_name = name
                best_cls_pipeline = pipe

    # Save best classification model
    if best_cls_pipeline is None:
        raise RuntimeError("No classification model trained?")

    best_cls_path = ARTIFACTS_DIR / "best_classifier.joblib"
    joblib.dump(best_cls_pipeline, best_cls_path)

    # Log best model to MLflow (separate run for registry-style clarity)
    with mlflow.start_run(run_name=f"classification_best_{best_cls_name}"):
        mlflow.log_param("task", "classification")
        mlflow.log_param("best_model_name", best_cls_name)
        mlflow.log_metric("best_f1_macro", float(best_f1))
        mlflow.sklearn.log_model(best_cls_pipeline, artifact_path="model")
        mlflow.log_artifact(str(best_cls_path))

    # ---------------------------
    # REGRESSION MODELS (4)
    # ---------------------------
    reg_models = {
        "ridge": Ridge(alpha=1.0, random_state=42),
        "svr_rbf": SVR(kernel="rbf"),
        "rf_reg": RandomForestRegressor(n_estimators=300, random_state=42),
        "gbr": GradientBoostingRegressor(random_state=42),
    }

    best_reg_name = None
    best_rmse = float("inf")
    best_reg_pipeline = None

    for name, model in reg_models.items():
        with mlflow.start_run(run_name=f"regression_{name}"):
            pipe = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", model),
                ]
            )
            mlflow.log_param("task", "regression")
            mlflow.log_param("model_name", name)

            start = time.time()
            pipe.fit(Xr_train, yr_train)
            fit_ms = int((time.time() - start) * 1000)

            y_pred = pipe.predict(Xr_test)
            metrics = evaluate_regression(yr_test, y_pred)
            metrics["fit_ms"] = fit_ms
            mlflow.log_metrics(metrics)

            # Store lightweight sample outputs
            sample_path = ARTIFACTS_DIR / f"regression_sample_{name}.json"
            sample = {
                "y_true_head": yr_test.head(10).tolist(),
                "y_pred_head": list(map(float, y_pred[:10])),
            }
            save_text(sample_path, json.dumps(sample, indent=2))
            mlflow.log_artifact(str(sample_path))

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_reg_name = name
                best_reg_pipeline = pipe

    # Save best regression model
    if best_reg_pipeline is None:
        raise RuntimeError("No regression model trained?")

    best_reg_path = ARTIFACTS_DIR / "best_regressor.joblib"
    joblib.dump(best_reg_pipeline, best_reg_path)

    with mlflow.start_run(run_name=f"regression_best_{best_reg_name}"):
        mlflow.log_param("task", "regression")
        mlflow.log_param("best_model_name", best_reg_name)
        mlflow.log_metric("best_rmse", float(best_rmse))
        mlflow.sklearn.log_model(best_reg_pipeline, artifact_path="model")
        mlflow.log_artifact(str(best_reg_path))

    print("✅ Training done.")
    print(f"Best classifier: {best_cls_name} (F1_macro={best_f1:.4f}) saved to {best_cls_path}")
    print(f"Best regressor : {best_reg_name} (RMSE={best_rmse:.4f}) saved to {best_reg_path}")


if __name__ == "__main__":
    main()