<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/MLflow-2.17.0-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLflow"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115.0-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.5.2-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-Learn"/>
</p>

# 🍷 ML Classification Pipeline

> A production-ready machine learning workflow for Wine classification demonstrating the three pillars of **MLOps**: experiment tracking, model registry management, and containerized model serving.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Getting Started](#-getting-started)
- [MLflow Experiments](#-mlflow-experiments)
- [API Reference](#-api-reference)
- [Docker Deployment](#-docker-deployment)
- [Testing](#-testing)
- [Environment Variables](#-environment-variables)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Overview

This project implements an end-to-end machine learning pipeline that bridges the gap between data science experimentation and production deployment:

| Pillar | Tool | What It Does |
|--------|------|-------------|
| **Experiment Tracking** | MLflow | Records all model runs, hyperparameters, metrics, and artifacts for reproducibility |
| **Model Registry** | MLflow Registry | Versions models and manages lifecycle transitions (staging → production) |
| **Containerized Serving** | Docker + FastAPI | Wraps the trained model in a REST API, packaged in an isolated Docker container |

---

## 🏗️ Architecture

```
┌─── Development & Training ───────────────┐    ┌─── Production Serving (Docker) ──────┐
│                                           │    │                                      │
│  ┌──────────────────┐                     │    │   End User / Service                 │
│  │  Data Processor   │                     │    │         │                            │
│  │  (Wine Dataset)   │                     │    │         ▼  POST /predict             │
│  └────────┬─────────┘                     │    │   ┌─────────────────┐               │
│           ▼                               │    │   │  Inference API   │               │
│  ┌──────────────────┐                     │    │   │  (FastAPI:8000)  │               │
│  │  Model Trainer    │                     │    │   └────────┬────────┘               │
│  │  (Scikit-Learn)   │                     │    │            │                        │
│  └──┬─────┬─────┬───┘                     │    │   Loads on Startup                  │
│     │     │     │                          │    │            │                        │
│     ▼     ▼     ▼                          │    │            ▼                        │
│  Metrics Artifacts Register                │    │   ┌─────────────────┐               │
│  (Acc,F1) (Scaler,  Best                  │    │   │  MLflow Server   │               │
│           Plots)   Model                   │    │   │  (:5000)         │               │
│     │     │     │                          │    │   └─────────────────┘               │
│     ▼     ▼     ▼                          │    │                                      │
│  ┌──────────────────────────┐              │    │   Shared Docker Volume               │
│  │  MLflow Tracking Server   │◄─────────────────►  (mlflow_data)                      │
│  └──────────────────────────┘              │    │                                      │
└───────────────────────────────────────────┘    └──────────────────────────────────────┘
```

---

## 📁 Project Structure

```
ML-Classification-Pipeline/
├── src/                          # Application source code
│   ├── __init__.py               # Package initializer
│   ├── data_processor.py         # Data loading & preprocessing (Wine dataset)
│   ├── model_trainer.py          # MLflow experiment tracking & model training
│   └── inference_api.py          # FastAPI REST API for serving predictions
├── tests/                        # Automated test suite
│   ├── __init__.py               # Test package initializer
│   └── test_api.py               # pytest unit tests for API endpoints
├── Dockerfile                    # Container blueprint for Inference API
├── docker-compose.yml            # Multi-service orchestration (MLflow + API)
├── requirements.txt              # Pinned Python dependencies
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
└── README.md                     # Project documentation (this file)
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Machine Learning** | Scikit-Learn, NumPy, Pandas |
| **Experiment Tracking** | MLflow (Tracking Server + Model Registry) |
| **API Framework** | FastAPI + Uvicorn |
| **Visualization** | Matplotlib, Seaborn |
| **Containerization** | Docker, Docker Compose |
| **Testing** | pytest, httpx |
| **Data Serialization** | joblib |

---

## ✅ Prerequisites

- **Python 3.9+** installed
- **Docker** and **Docker Compose** v2.0+ installed
- **pip** or **uv** package manager

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/raghavendra2006/ML-Classification-Pipeline.git
cd ML-Classification-Pipeline
```

### 2️⃣ Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using uv
uv pip install -r requirements.txt
```

### 3️⃣ Start the MLflow Tracking Server

**Option A — Run locally (for development):**

```bash
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

**Option B — Run via Docker Compose:**

```bash
docker-compose up mlflow_server -d
```

> 📌 The MLflow UI is accessible at **http://localhost:5000**

### 4️⃣ Train Models & Populate the Registry

```bash
# Set tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000      # Linux/Mac
set MLFLOW_TRACKING_URI=http://localhost:5000          # Windows CMD
$env:MLFLOW_TRACKING_URI="http://localhost:5000"       # PowerShell

# Run the training pipeline
python -m src.model_trainer
```

This will:
- Execute **5 experiment runs** (3 Logistic Regression + 2 Random Forest)
- Log hyperparameters, metrics, and artifacts to MLflow
- Generate confusion matrix plots for each run
- Register the **best-performing model** (by F1-score) as `ClassificationModel`

### 5️⃣ Deploy the Full Stack with Docker Compose

```bash
docker-compose up --build -d
```

| Service | URL | Description |
|---------|-----|-------------|
| **MLflow Server** | http://localhost:5000 | Experiment tracking UI & Model Registry |
| **Inference API** | http://localhost:8000 | REST API for model predictions |
| **API Docs** | http://localhost:8000/docs | Interactive Swagger documentation |

> ⚠️ **Important:** Run the training script (Step 4) before starting the API, as it loads the model from the MLflow Registry on startup.

---

## 🧪 MLflow Experiments

The training pipeline executes **5 experiment runs** with varying hyperparameters:

| Run Name | Algorithm | Key Hyperparameter | Metrics Tracked |
|----------|-----------|-------------------|-----------------|
| `LogReg_C0.1` | Logistic Regression | C=0.1 | Accuracy, F1, Precision, Recall |
| `LogReg_C1.0` | Logistic Regression | C=1.0 | Accuracy, F1, Precision, Recall |
| `LogReg_C10.0` | Logistic Regression | C=10.0 | Accuracy, F1, Precision, Recall |
| `RF_n50` | Random Forest | n_estimators=50 | Accuracy, F1, Precision, Recall |
| `RF_n100` | Random Forest | n_estimators=100 | Accuracy, F1, Precision, Recall |

### Artifacts Logged Per Run

| Artifact | Path | Description |
|----------|------|-------------|
| 🤖 Trained Model | `model/` | Serialized Scikit-Learn model (MLflow format) |
| ⚖️ Fitted Scaler | `preprocessing/scaler.joblib` | StandardScaler for input transformation |
| 📊 Confusion Matrix | `plots/confusion_matrix_*.png` | Heatmap visualization of predictions |

---

## 📡 API Reference

### `GET /health` — Health Check

Readiness probe for container orchestration.

```bash
curl http://localhost:8000/health
```

**Response** `200 OK`:
```json
{
  "status": "healthy"
}
```

---

### `POST /predict` — Classification Prediction

Accepts raw Wine features, applies StandardScaler transformation, returns predicted class.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [14.23, 1.71, 2.43, 15.6, 127.0, 2.80, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0]}'
```

**Response** `200 OK`:
```json
{
  "prediction": 0
}
```

**Error — Wrong Feature Count** `400 Bad Request`:
```json
{
  "detail": "Expected 13 features, got 3."
}
```

**Error — Missing Features Key** `422 Unprocessable Entity`:
```json
{
  "detail": [{"msg": "Field required", "type": "missing"}]
}
```

### Wine Dataset Features (13 total)

| # | Feature | Description |
|---|---------|-------------|
| 1 | Alcohol | Alcohol content |
| 2 | Malic acid | Malic acid concentration |
| 3 | Ash | Ash content |
| 4 | Alcalinity of ash | Alcalinity measurement |
| 5 | Magnesium | Magnesium content |
| 6 | Total phenols | Total phenol count |
| 7 | Flavanoids | Flavanoid content |
| 8 | Nonflavanoid phenols | Non-flavanoid phenol content |
| 9 | Proanthocyanins | Proanthocyanin content |
| 10 | Color intensity | Color intensity measurement |
| 11 | Hue | Hue value |
| 12 | OD280/OD315 | Diluted wines measurement |
| 13 | Proline | Proline amino acid content |

---

## 🐳 Docker Deployment

### Services Overview

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `mlflow_server` | `ghcr.io/mlflow/mlflow:v2.8.0` | 5000 | Tracking server with SQLite backend |
| `model_api` | Custom (Python 3.9-slim) | 8000 | FastAPI inference server |

### Commands

```bash
# Build and start all services
docker-compose up --build -d

# View real-time logs
docker-compose logs -f

# Stop all services
docker-compose down

# Stop and remove all data volumes
docker-compose down -v

# Rebuild a specific service
docker-compose up --build model_api -d
```

### Shared Volume

Both services share a `mlflow_data` Docker volume for artifact access. This ensures the API container can read models and scalers logged by the training script.

---

## 🧪 Testing

Run the full test suite:

```bash
pytest tests/ -v
```

> 💡 Tests use **mocked** model and scaler objects — no running MLflow server required.

### Test Coverage

| Test Class | Tests | What It Validates |
|------------|-------|-------------------|
| `TestHealthEndpoint` | 2 | GET `/health` returns 200 + correct JSON |
| `TestPredictEndpointValid` | 3 | POST `/predict` with valid 13-feature input |
| `TestPredictEndpointErrors` | 6 | Missing keys, wrong count, invalid types → proper HTTP codes |

**Expected output:**
```
tests/test_api.py::TestHealthEndpoint::test_health_returns_200 PASSED
tests/test_api.py::TestHealthEndpoint::test_health_returns_healthy_status PASSED
tests/test_api.py::TestPredictEndpointValid::test_predict_valid_features PASSED
tests/test_api.py::TestPredictEndpointValid::test_predict_response_has_prediction_key PASSED
tests/test_api.py::TestPredictEndpointValid::test_predict_returns_integer PASSED
tests/test_api.py::TestPredictEndpointErrors::test_predict_missing_features_key PASSED
tests/test_api.py::TestPredictEndpointErrors::test_predict_empty_body PASSED
tests/test_api.py::TestPredictEndpointErrors::test_predict_wrong_feature_count PASSED
tests/test_api.py::TestPredictEndpointErrors::test_predict_wrong_feature_count_error_message PASSED
tests/test_api.py::TestPredictEndpointErrors::test_predict_invalid_content_type PASSED
tests/test_api.py::TestPredictEndpointErrors::test_predict_features_not_numeric PASSED

======================= 11 passed in 1.51s ========================
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow tracking server URI |
| `MODEL_NAME` | `ClassificationModel` | Model name in MLflow Registry |
| `MAX_RETRIES` | `5` | Max startup retries for MLflow connection |
| `RETRY_DELAY` | `5` | Seconds between connection retries |

Copy `.env.example` to `.env` and customize as needed.

---

## ❓ Troubleshooting

<details>
<summary><b>API container fails: "Could not load model"</b></summary>

The MLflow Model Registry is empty. Run the training script first:
```bash
python -m src.model_trainer
```
</details>

<details>
<summary><b>Why does the API need the scaler?</b></summary>

The model was trained on **scaled** data. Raw features like `[14.23, 1.71, ...]` must be transformed to the same scale (e.g., `[1.5, -0.2, ...]`) before prediction. The `StandardScaler` artifact ensures this transformation is identical to what was used during training.
</details>

<details>
<summary><b>How do I access the MLflow UI?</b></summary>

Navigate to **http://localhost:5000** in your browser after starting the MLflow server (locally or via Docker Compose).
</details>

<details>
<summary><b>How do MLflow and the API share data in Docker?</b></summary>

Both services mount the same Docker volume (`mlflow_data`). The MLflow server writes models and artifacts to this volume, and the API container reads them via the MLflow tracking URI.
</details>

---

## 📄 License

This project is provided for educational and assessment purposes.

---

<p align="center">
  Built with ❤️ using MLflow, FastAPI, and Docker
</p>