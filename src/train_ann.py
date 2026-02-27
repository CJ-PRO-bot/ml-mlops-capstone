from __future__ import annotations

import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Keras
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_processed.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

MLFLOW_DB = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
mlflow.set_tracking_uri(MLFLOW_DB)
mlflow.set_registry_uri(MLFLOW_DB)

EXPERIMENT_NAME = "ecoguard-lite-phase2"


def build_model(
    n_features: int,
    hidden: list[int],
    activation: str,
    dropout: float,
    l2: float,
) -> tf.keras.Model:
    inp = layers.Input(shape=(n_features,))
    x = inp
    for h in hidden:
        x = layers.Dense(
            h,
            kernel_initializer="he_normal" if activation.lower().startswith("relu") else "glorot_uniform",
            kernel_regularizer=regularizers.l2(l2) if l2 > 0 else None,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)

    out = layers.Dense(3, activation="softmax")(x)  # assuming 3 classes; adjust if needed
    return tf.keras.Model(inp, out)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    y = df["y_cls"].astype(int).values
    X = df.drop(columns=["y_cls", "y_reg"])

    # Ensure classes count (change 3 above if your y_cls has different)
    n_classes = int(np.unique(y).size)
    if n_classes != 3:
        print(f"⚠️ Detected {n_classes} classes. Update final Dense layer to {n_classes}.")
        # quick auto-fix:
        # (for assignment, you can set it manually)
        # return

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    mlflow.set_experiment(EXPERIMENT_NAME)

    configs = [
        {"hidden": [128, 64], "activation": "relu", "dropout": 0.2, "l2": 1e-4, "lr": 1e-3, "opt": "adam"},
        {"hidden": [256, 128, 64], "activation": "relu", "dropout": 0.3, "l2": 1e-4, "lr": 5e-4, "opt": "rmsprop"},
        {"hidden": [128, 64], "activation": "tanh", "dropout": 0.2, "l2": 1e-3, "lr": 1e-3, "opt": "adam"},
    ]

    for i, cfg in enumerate(configs, start=1):
        with mlflow.start_run(run_name=f"ann_cfg{i}"):
            model = build_model(
                n_features=X_train_s.shape[1],
                hidden=cfg["hidden"],
                activation=cfg["activation"],
                dropout=cfg["dropout"],
                l2=cfg["l2"],
            )

            if cfg["opt"] == "adam":
                optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["lr"])
            elif cfg["opt"] == "rmsprop":
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=cfg["lr"])
            else:
                optimizer = tf.keras.optimizers.Adagrad(learning_rate=cfg["lr"])

            model.compile(
                optimizer=optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            cb = [
                callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-5),
            ]

            mlflow.log_params(cfg)

            hist = model.fit(
                X_train_s,
                y_train,
                validation_data=(X_val_s, y_val),
                epochs=40,
                batch_size=64,
                verbose=0,
                callbacks=cb,
            )

            best_val_acc = float(np.max(hist.history["val_accuracy"]))
            best_val_loss = float(np.min(hist.history["val_loss"]))
            mlflow.log_metric("best_val_accuracy", best_val_acc)
            mlflow.log_metric("best_val_loss", best_val_loss)

            # Save learning curves (lightweight CSV)
            curves_path = ARTIFACTS_DIR / f"ann_cfg{i}_curves.csv"
            pd.DataFrame(hist.history).to_csv(curves_path, index=False)
            mlflow.log_artifact(str(curves_path))

    print("✅ ANN Phase2 done. Check MLflow experiment ecoguard-lite-phase2.")


if __name__ == "__main__":
    # silence TF spam
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()