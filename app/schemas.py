from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogEntry(BaseModel):
    message: str
    level: LogLevel = LogLevel.INFO
    response_time_ms: Optional[float] = None
    status_code: Optional[int] = None
    error_count: Optional[int] = 0

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Database connection timeout after 30s",
                "level": "ERROR",
                "response_time_ms": 30000,
                "status_code": 500,
                "error_count": 5,
            }
        }


class LogBatch(BaseModel):
    logs: List[LogEntry]


class TrainRequest(BaseModel):
    logs: List[LogEntry]


class AnomalyResult(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    confidence: str
    features_used: dict
    message: str


class HealthResponse(BaseModel):
    status: str
    model_trained: bool
