from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_processed.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

EXPERIMENT_NAME = "ecoguard-lite-phase1"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(TRACKING_URI)


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing processed dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["y_cls", "y_reg"], errors="ignore")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    _ = pca.fit_transform(Xs)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # --- KMeans: Elbow (inertia) + Silhouette + Davies-Bouldin ---
    elbow_rows = []

    for k in [2, 3, 4, 5, 6]:
        with mlflow.start_run(run_name=f"clustering_kmeans_k{k}"):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(Xs)

            inertia = float(km.inertia_)  # Elbow metric
            sil = float(silhouette_score(Xs, labels))
            dbi = float(davies_bouldin_score(Xs, labels))

            mlflow.log_param("algo", "kmeans")
            mlflow.log_param("k", k)
            mlflow.log_metric("inertia", inertia)
            mlflow.log_metric("silhouette", sil)
            mlflow.log_metric("davies_bouldin", dbi)

            elbow_rows.append({"k": k, "inertia": inertia, "silhouette": sil, "davies_bouldin": dbi})

            out = ARTIFACTS_DIR / f"kmeans_k{k}_summary.json"
            save_json(
                out,
                {
                    "k": k,
                    "inertia": inertia,
                    "silhouette": sil,
                    "davies_bouldin": dbi,
                    "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                },
            )
            mlflow.log_artifact(str(out))

    # Save a single elbow table artifact (nice for report/demo)
    elbow_df = pd.DataFrame(elbow_rows)
    elbow_csv = ARTIFACTS_DIR / "kmeans_elbow_metrics.csv"
    elbow_df.to_csv(elbow_csv, index=False)
    with mlflow.start_run(run_name="clustering_kmeans_elbow_table"):
        mlflow.log_param("algo", "kmeans")
        mlflow.log_artifact(str(elbow_csv))

    # --- DBSCAN: log clusters + optional silhouette/DBI (non-noise only) ---
    for eps in [0.6, 0.8, 1.0]:
        with mlflow.start_run(run_name=f"clustering_dbscan_eps{eps}"):
            db = DBSCAN(eps=eps, min_samples=8)
            labels = db.fit_predict(Xs)

            unique = set(labels)
            n_clusters = len([u for u in unique if u != -1])
            noise_points = int(np.sum(labels == -1))

            mlflow.log_param("algo", "dbscan")
            mlflow.log_param("eps", eps)
            mlflow.log_param("min_samples", 8)
            mlflow.log_metric("n_clusters_excluding_noise", float(n_clusters))
            mlflow.log_metric("noise_points", float(noise_points))

            sil_non_noise = None
            dbi_non_noise = None

            try:
                mask = labels != -1
                if n_clusters >= 2 and np.sum(mask) > 10:
                    sil_non_noise = float(silhouette_score(Xs[mask], labels[mask]))
                    dbi_non_noise = float(davies_bouldin_score(Xs[mask], labels[mask]))
                    mlflow.log_metric("silhouette_non_noise", sil_non_noise)
                    mlflow.log_metric("davies_bouldin_non_noise", dbi_non_noise)
            except Exception:
                pass

            out = ARTIFACTS_DIR / f"dbscan_eps{eps}_summary.json"
            save_json(
                out,
                {
                    "eps": eps,
                    "min_samples": 8,
                    "n_clusters_excluding_noise": n_clusters,
                    "noise_points": noise_points,
                    "silhouette_non_noise": sil_non_noise,
                    "davies_bouldin_non_noise": dbi_non_noise,
                    "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                },
            )
            mlflow.log_artifact(str(out))

    print("✅ Clustering done. Check MLflow runs + artifacts/ outputs.")


if __name__ == "__main__":
    main()