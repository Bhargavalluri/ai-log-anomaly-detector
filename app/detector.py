import re
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List
import joblib
import os

MODEL_PATH = "models/isolation_forest.joblib"
SCALER_PATH = "models/scaler.joblib"

LEVEL_MAP = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}

ERROR_KEYWORDS = [
    "exception", "error", "fail", "timeout", "crash", "refused",
    "unavailable", "panic", "fatal", "killed", "oom", "null",
]


def extract_features(log: Dict[str, Any]) -> np.ndarray:
    message = log.get("message", "").lower()
    level = log.get("level", "INFO")
    response_time = log.get("response_time_ms") or 0.0
    status_code = log.get("status_code") or 200
    error_count = log.get("error_count") or 0

    level_score = LEVEL_MAP.get(level, 1)
    keyword_hits = sum(1 for kw in ERROR_KEYWORDS if kw in message)
    has_stack_trace = 1 if ("traceback" in message or "at line" in message or "exception in" in message) else 0
    msg_length = len(message)
    is_5xx = 1 if status_code >= 500 else 0
    is_4xx = 1 if 400 <= status_code < 500 else 0
    numbers = re.findall(r"\d+", message)
    large_number = max((int(n) for n in numbers if int(n) > 1000), default=0)

    return np.array([
        level_score,
        keyword_hits,
        has_stack_trace,
        msg_length,
        response_time,
        is_5xx,
        is_4xx,
        error_count,
        min(large_number, 100000),
    ], dtype=float)


class AnomalyDetector:
    def __init__(self):
        self.model: IsolationForest | None = None
        self.scaler: StandardScaler | None = None
        self.is_trained = False
        self.total_predictions = 0
        self.total_anomalies = 0
        self._try_load()

    def _try_load(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.is_trained = True

    def train(self, logs: List[Dict[str, Any]]):
        os.makedirs("models", exist_ok=True)
        X = np.array([extract_features(log) for log in logs])
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)
        self.is_trained = True
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)

    def predict(self, log: Dict[str, Any]) -> Dict[str, Any]:
        features = extract_features(log)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        score = self.model.decision_function(features_scaled)[0]
        prediction = self.model.predict(features_scaled)[0]
        is_anomaly = prediction == -1

        self.total_predictions += 1
        if is_anomaly:
            self.total_anomalies += 1

        if score < -0.3:
            confidence = "HIGH"
        elif score < -0.1:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        feature_names = [
            "level_score", "keyword_hits", "has_stack_trace", "msg_length",
            "response_time_ms", "is_5xx", "is_4xx", "error_count", "large_number",
        ]

        return {
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": round(float(score), 4),
            "confidence": confidence if is_anomaly else "N/A",
            "features_used": dict(zip(feature_names, features.tolist())),
            "message": "⚠️ Anomaly detected" if is_anomaly else "✅ Normal log entry",
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "is_trained": self.is_trained,
            "total_predictions": self.total_predictions,
            "total_anomalies": self.total_anomalies,
            "anomaly_rate": (
                round(self.total_anomalies / self.total_predictions, 3)
                if self.total_predictions > 0 else 0
            ),
        }
