from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_processed.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
MLFLOW_DB = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
mlflow.set_tracking_uri(MLFLOW_DB)
mlflow.set_registry_uri(MLFLOW_DB)
EXPERIMENT_NAME = "ecoguard-lite-phase1"


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing processed dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    # clustering uses features only
    X = df.drop(columns=["y_cls", "y_reg"])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # PCA for 2D visualization (and feature compression)
    pca = PCA(n_components=2, random_state=42)
    _ = pca.fit_transform(Xs)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # KMeans (try a small range for speed)
    for k in [2, 3, 4, 5, 6]:
        with mlflow.start_run(run_name=f"clustering_kmeans_k{k}"):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(Xs)

            sil = float(silhouette_score(Xs, labels))
            mlflow.log_param("algo", "kmeans")
            mlflow.log_param("k", k)
            mlflow.log_metric("silhouette", sil)

            info = {
                "k": k,
                "silhouette": sil,
                "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            }
            out = ARTIFACTS_DIR / f"kmeans_k{k}_summary.json"
            save_json(out, info)
            mlflow.log_artifact(str(out))

    # DBSCAN (parameter choices kept simple/light)
    for eps in [0.6, 0.8, 1.0]:
        with mlflow.start_run(run_name=f"clustering_dbscan_eps{eps}"):
            db = DBSCAN(eps=eps, min_samples=8)
            labels = db.fit_predict(Xs)

            # DBSCAN label -1 means noise. Silhouette needs >=2 clusters and no single cluster.
            unique = set(labels)
            n_clusters = len([u for u in unique if u != -1])

            mlflow.log_param("algo", "dbscan")
            mlflow.log_param("eps", eps)
            mlflow.log_param("min_samples", 8)
            mlflow.log_metric("n_clusters_excluding_noise", float(n_clusters))
            mlflow.log_metric("noise_points", float(np.sum(labels == -1)))

            sil = None
            try:
                if n_clusters >= 2:
                    # silhouette score requires >1 label
                    sil = float(
                        silhouette_score(Xs[labels != -1], labels[labels != -1])
                    )
                    mlflow.log_metric("silhouette_non_noise", sil)
            except Exception:
                pass

            info = {
                "eps": eps,
                "min_samples": 8,
                "n_clusters_excluding_noise": n_clusters,
                "noise_points": int(np.sum(labels == -1)),
                "silhouette_non_noise": sil,
                "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            }
            out = ARTIFACTS_DIR / f"dbscan_eps{eps}_summary.json"
            save_json(out, info)
            mlflow.log_artifact(str(out))

    print("✅ Clustering done. Check MLflow runs + artifacts folder.")


if __name__ == "__main__":
    main()
