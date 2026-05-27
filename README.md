# AI Log Anomaly Detector

A production-ready REST API that detects anomalies in application logs using an **Isolation Forest** machine learning model. Built with FastAPI and scikit-learn — no GPU or external AI API required.

## Features

- 🔍 **Real-time anomaly detection** on individual log entries
- 📦 **Batch processing** — analyze thousands of logs at once
- 🧠 **Isolation Forest** — unsupervised ML, no labeled data needed
- 📊 **Feature extraction** — level severity, response time, HTTP status codes, error keywords, stack traces
- 🐳 **Docker-ready** — single command deployment
- 📈 **Stats endpoint** — track anomaly rates over time

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Uvicorn |
| ML Model | scikit-learn Isolation Forest |
| Feature Engineering | NumPy, regex |
| Containerization | Docker + Docker Compose |
| Testing | pytest + httpx |

## Architecture

```
POST /train  ──►  Feature Extraction  ──►  Isolation Forest (fit)
POST /detect ──►  Feature Extraction  ──►  Isolation Forest (predict)  ──►  AnomalyResult
```

**Features extracted from each log:**
- Log level score (DEBUG=0 → CRITICAL=4)
- Error keyword count (exception, timeout, crash, etc.)
- Stack trace presence
- Message length
- Response time (ms)
- HTTP 4xx / 5xx flags
- Error count
- Large numeric values in message

## Quick Start

### With Docker (recommended)
```bash
docker-compose up --build
```

### Local
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs available at: `http://localhost:8000/docs`

## Usage

### 1. Train the model
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "logs": [
      {"message": "User login successful", "level": "INFO", "response_time_ms": 120, "status_code": 200, "error_count": 0},
      {"message": "GET /api/users returned 200", "level": "INFO", "response_time_ms": 85, "status_code": 200, "error_count": 0}
    ]
  }'
```

### 2. Detect an anomaly
```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "message": "CRITICAL: database connection timeout exception — traceback null pointer",
    "level": "CRITICAL",
    "response_time_ms": 30000,
    "status_code": 500,
    "error_count": 50
  }'
```

**Response:**
```json
{
  "is_anomaly": true,
  "anomaly_score": -0.4821,
  "confidence": "HIGH",
  "features_used": { "level_score": 4, "keyword_hits": 4, "response_time_ms": 30000, ... },
  "message": "⚠️ Anomaly detected"
}
```

### 3. Batch detection
```bash
curl -X POST http://localhost:8000/detect/batch \
  -H "Content-Type: application/json" \
  -d '{"logs": [...]}'
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check + model status |
| POST | `/train` | Train model on normal logs |
| POST | `/detect` | Detect anomaly in one log |
| POST | `/detect/batch` | Detect anomalies in batch |
| GET | `/stats` | Detection statistics |

## Running Tests
```bash
pytest tests/ -v
```

## Real-World Use Cases
- Monitor microservice logs for production incidents
- Alert on abnormal database query times
- Detect unusual API error spikes
- Flag potential security events in auth logs

## Author
**Bhargav Alluri** — [LinkedIn](https://linkedin.com/in/bhargav-alluri-engineer) · [GitHub](https://github.com/bhargavalluri)
