import sys
sys.path.insert(0, '/app')

from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    response = client.post("/predict", json={
        "income": 5000,
        "total_expenses": 3500,
        "num_transactions": 45,
        "groceries": 400,
        "restaurants": 250,
        "utilities": 200,
        "shopping": 150,
        "mortgage": 1200
    })
    assert response.status_code == 200
    assert "predicted_savings" in response.json()
    assert "recommendation" in response.json()
