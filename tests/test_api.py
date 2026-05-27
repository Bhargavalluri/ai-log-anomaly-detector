from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

NORMAL_LOGS = [
    {"message": "User login successful", "level": "INFO", "response_time_ms": 120, "status_code": 200, "error_count": 0},
    {"message": "GET /api/users returned 200", "level": "INFO", "response_time_ms": 85, "status_code": 200, "error_count": 0},
    {"message": "Cache hit for key user_123", "level": "DEBUG", "response_time_ms": 5, "status_code": 200, "error_count": 0},
    {"message": "Scheduled job completed", "level": "INFO", "response_time_ms": 300, "status_code": 200, "error_count": 0},
    {"message": "POST /api/orders returned 201", "level": "INFO", "response_time_ms": 200, "status_code": 201, "error_count": 0},
] * 4  # 20 total


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_train():
    r = client.post("/train", json={"logs": NORMAL_LOGS})
    assert r.status_code == 200
    assert "trained" in r.json()["message"]


def test_detect_normal():
    client.post("/train", json={"logs": NORMAL_LOGS})
    r = client.post("/detect", json={
        "message": "GET /api/health returned 200",
        "level": "INFO",
        "response_time_ms": 95,
        "status_code": 200,
        "error_count": 0,
    })
    assert r.status_code == 200
    data = r.json()
    assert "is_anomaly" in data


def test_detect_anomaly():
    client.post("/train", json={"logs": NORMAL_LOGS})
    r = client.post("/detect", json={
        "message": "CRITICAL: database connection timeout exception traceback null pointer",
        "level": "CRITICAL",
        "response_time_ms": 30000,
        "status_code": 500,
        "error_count": 50,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["is_anomaly"] is True


def test_detect_batch():
    client.post("/train", json={"logs": NORMAL_LOGS})
    r = client.post("/detect/batch", json={"logs": NORMAL_LOGS[:5]})
    assert r.status_code == 200
    assert r.json()["total"] == 5
