from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import LogEntry, LogBatch, AnomalyResult, TrainRequest, HealthResponse
from app.detector import AnomalyDetector
from app.config import settings
import uvicorn

app = FastAPI(
    title="AI Log Anomaly Detector",
    description="Real-time log anomaly detection using Isolation Forest ML model",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = AnomalyDetector()


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "model_trained": detector.is_trained}


@app.post("/train")
def train(request: TrainRequest):
    """Train the model on a batch of normal logs."""
    if len(request.logs) < 10:
        raise HTTPException(status_code=400, detail="Need at least 10 log entries to train.")
    detector.train([log.dict() for log in request.logs])
    return {"message": f"Model trained on {len(request.logs)} log entries."}


@app.post("/detect", response_model=AnomalyResult)
def detect(entry: LogEntry):
    """Detect if a single log entry is anomalous."""
    if not detector.is_trained:
        raise HTTPException(status_code=400, detail="Model not trained yet. POST to /train first.")
    result = detector.predict(entry.dict())
    return result


@app.post("/detect/batch")
def detect_batch(batch: LogBatch):
    """Detect anomalies across a batch of log entries."""
    if not detector.is_trained:
        raise HTTPException(status_code=400, detail="Model not trained yet. POST to /train first.")
    results = [detector.predict(entry.dict()) for entry in batch.logs]
    anomalies = [r for r in results if r["is_anomaly"]]
    return {
        "total": len(results),
        "anomalies_found": len(anomalies),
        "anomaly_rate": round(len(anomalies) / len(results), 3),
        "results": results,
    }


@app.get("/stats")
def stats():
    """Return model and detection statistics."""
    return detector.get_stats()


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
