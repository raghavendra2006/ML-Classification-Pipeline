"""
Inference API Module
====================
FastAPI application that serves predictions from the MLflow Model Registry.

The model and scaler are loaded exactly ONCE at startup to avoid I/O
bottlenecks on every request. The /predict endpoint applies the same
StandardScaler transformation used during training before calling the model.

Environment Variables:
    MLFLOW_TRACKING_URI : URI of the MLflow tracking server
                          (default: http://localhost:5000)
    MODEL_NAME          : Registered model name in the MLflow registry
                          (default: ClassificationModel)
"""

import os
import time
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from typing import List

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """Request body for the /predict endpoint."""
    features: List[float]

    @field_validator("features")
    @classmethod
    def features_must_not_be_empty(cls, v):
        if len(v) == 0:
            raise ValueError("features list must not be empty")
        return v


class PredictionResponse(BaseModel):
    """Response body for the /predict endpoint."""
    prediction: int


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""
    status: str


# ---------------------------------------------------------------------------
# Global model and scaler — loaded once at startup
# ---------------------------------------------------------------------------
MODEL = None
SCALER = None

# Expected number of features for the Wine dataset
EXPECTED_FEATURES = 13


def _load_model_and_scaler():
    """
    Load the registered model and its associated scaler from MLflow.

    Implements retry logic to handle the case where the MLflow server
    is not yet available (e.g., during Docker Compose startup).
    """
    global MODEL, SCALER

    # Lazy imports so tests can mock at module level
    import joblib
    import mlflow
    import mlflow.sklearn

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    model_name = os.environ.get("MODEL_NAME", "ClassificationModel")
    max_retries = int(os.environ.get("MAX_RETRIES", "5"))
    retry_delay = int(os.environ.get("RETRY_DELAY", "5"))

    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    logger.info(f"Loading model: {model_name}")

    # --- Load model with retries -------------------------------------------
    for attempt in range(1, max_retries + 1):
        try:
            model_uri = f"models:/{model_name}/latest"
            MODEL = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Model loaded successfully on attempt {attempt}")
            break
        except Exception as e:
            logger.warning(
                f"Attempt {attempt}/{max_retries} failed: {e}"
            )
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                logger.error("Failed to load model after all retries.")
                raise RuntimeError(
                    f"Could not load model '{model_name}' from MLflow. "
                    f"Ensure the training script has been run and the model "
                    f"is registered in the Model Registry."
                )

    # --- Load scaler artifact from the model's run -------------------------
    try:
        # Get the run ID associated with the latest model version
        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions(model_name)
        if not latest_versions:
            raise RuntimeError(f"No versions found for model '{model_name}'")

        run_id = latest_versions[0].run_id
        logger.info(f"Loading scaler from run: {run_id}")

        # Download the scaler artifact
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="preprocessing/scaler.joblib"
        )
        SCALER = joblib.load(artifact_path)
        logger.info("Scaler loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load scaler: {e}")
        raise RuntimeError(
            f"Could not load scaler artifact. Ensure the training script "
            f"logged the scaler under 'preprocessing/scaler.joblib'. Error: {e}"
        )


# ---------------------------------------------------------------------------
# Lifespan (modern FastAPI pattern, replaces on_event("startup"))
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load model on startup, cleanup on shutdown."""
    _load_model_and_scaler()
    yield
    # Cleanup (if needed) goes here


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ML Classification API",
    description="Serves Wine classification predictions from MLflow Model Registry.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint for container orchestration readiness probes."""
    return HealthResponse(status="healthy")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Generate a classification prediction.

    Accepts raw features, applies the same scaling used during training,
    and returns the model's predicted class.
    """
    if MODEL is None or SCALER is None:
        raise HTTPException(
            status_code=503,
            detail="Model or scaler not loaded. Service is not ready.",
        )

    # Validate feature count
    if len(request.features) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected {EXPECTED_FEATURES} features, "
                f"got {len(request.features)}."
            ),
        )

    try:
        # Reshape to 2D array and apply scaling
        features_array = np.array(request.features).reshape(1, -1)
        scaled_features = SCALER.transform(features_array)

        # Predict
        prediction = MODEL.predict(scaled_features)
        return PredictionResponse(prediction=int(prediction[0]))

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}",
        )
