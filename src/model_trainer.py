"""
Model Trainer Module
====================
Trains multiple classification models with varying hyperparameters,
tracks all experiments via MLflow, and registers the best model.

Usage:
    python -m src.model_trainer

Requires an MLflow tracking server running (locally or via Docker Compose).
"""

import os
import tempfile

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_processor import load_and_preprocess_data

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "Wine-Classification"
REGISTRY_MODEL_NAME = "ClassificationModel"

# Each entry defines one training run: (model_class, params_dict, run_name)
EXPERIMENT_RUNS = [
    (
        LogisticRegression,
        {"C": 0.1, "solver": "lbfgs", "max_iter": 1000, "random_state": 42},
        "LogReg_C0.1",
    ),
    (
        LogisticRegression,
        {"C": 1.0, "solver": "lbfgs", "max_iter": 1000, "random_state": 42},
        "LogReg_C1.0",
    ),
    (
        LogisticRegression,
        {"C": 10.0, "solver": "lbfgs", "max_iter": 1000, "random_state": 42},
        "LogReg_C10.0",
    ),
    (
        RandomForestClassifier,
        {"n_estimators": 50, "random_state": 42},
        "RF_n50",
    ),
    (
        RandomForestClassifier,
        {"n_estimators": 100, "random_state": 42},
        "RF_n100",
    ),
]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _save_confusion_matrix(y_true, y_pred, target_names, run_name, tmpdir):
    """Generate and save a confusion matrix heatmap as a PNG file."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title(f"Confusion Matrix — {run_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    filepath = os.path.join(tmpdir, f"confusion_matrix_{run_name}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    return filepath


def _save_scaler(scaler, tmpdir):
    """Serialize the fitted scaler to disk via joblib."""
    filepath = os.path.join(tmpdir, "scaler.joblib")
    joblib.dump(scaler, filepath)
    return filepath


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def train_and_log():
    """
    Execute all experiment runs, log to MLflow, and register the best model.

    Returns:
        str: The run ID of the best performing model.
    """
    # --- Resolve MLflow tracking URI ----------------------------------------
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"[INFO] MLflow tracking URI: {tracking_uri}")

    # --- Load and preprocess data -------------------------------------------
    (
        X_train, X_test, y_train, y_test,
        scaler, feature_names, target_names,
    ) = load_and_preprocess_data()
    print(f"[INFO] Dataset loaded — train: {X_train.shape}, test: {X_test.shape}")

    # --- Create / set experiment --------------------------------------------
    mlflow.set_experiment(EXPERIMENT_NAME)

    # --- Run experiments ----------------------------------------------------
    run_results = []  # list of (run_id, f1)

    for model_class, params, run_name in EXPERIMENT_RUNS:
        print(f"\n{'='*60}")
        print(f"[RUN] {run_name}")
        print(f"{'='*60}")

        with mlflow.start_run(run_name=run_name) as run:
            # -- Train -------------------------------------------------------
            model = model_class(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # -- Compute metrics ---------------------------------------------
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            prec = precision_score(y_test, y_pred, average="weighted")
            rec = recall_score(y_test, y_pred, average="weighted")

            print(f"  Accuracy : {acc:.4f}")
            print(f"  F1-Score : {f1:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall   : {rec:.4f}")

            # -- Log params --------------------------------------------------
            mlflow.log_params({
                "algorithm": model_class.__name__,
                **{k: str(v) for k, v in params.items()},
            })

            # -- Log metrics -------------------------------------------------
            mlflow.log_metrics({
                "accuracy": round(acc, 4),
                "f1_score": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
            })

            # -- Log artifacts -----------------------------------------------
            with tempfile.TemporaryDirectory() as tmpdir:
                # Scaler
                scaler_path = _save_scaler(scaler, tmpdir)
                mlflow.log_artifact(scaler_path, artifact_path="preprocessing")

                # Confusion matrix
                cm_path = _save_confusion_matrix(
                    y_test, y_pred, target_names, run_name, tmpdir,
                )
                mlflow.log_artifact(cm_path, artifact_path="plots")

            # -- Log model ---------------------------------------------------
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=None,  # we register the best one below
            )

            run_results.append((run.info.run_id, f1))

    # --- Identify & register best model -------------------------------------
    best_run_id, best_f1 = max(run_results, key=lambda x: x[1])
    best_model_uri = f"runs:/{best_run_id}/model"

    print(f"\n{'='*60}")
    print(f"[BEST] Run {best_run_id}  |  F1 = {best_f1:.4f}")
    print(f"[REGISTRY] Registering as '{REGISTRY_MODEL_NAME}'")
    print(f"{'='*60}")

    mlflow.register_model(best_model_uri, REGISTRY_MODEL_NAME)
    print("[DONE] Training complete. Best model registered.")

    return best_run_id


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_and_log()
