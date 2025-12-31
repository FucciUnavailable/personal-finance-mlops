from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

app = FastAPI(title="Personal Finance Savings Predictor")

# Load model at startup
model = joblib.load("models/savings_predictor.pkl")

class TransactionInput(BaseModel):
    income: float
    total_expenses: float
    num_transactions: int
    groceries: float
    restaurants: float
    utilities: float
    shopping: float
    mortgage: float

class PredictionResponse(BaseModel):
    predicted_savings: float
    recommendation: str

@app.get("/")
def root():
    return FileResponse("src/static/index.html")

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: TransactionInput):
    """Predict monthly savings based on spending patterns"""

    features = np.array([[
        data.income,
        data.total_expenses,
        data.num_transactions,
        data.groceries,
        data.restaurants,
        data.utilities,
        data.shopping,
        data.mortgage
    ]])

    prediction = model.predict(features)[0]

    # Generate recommendation
    savings_rate = (prediction / data.income * 100) if data.income > 0 else 0

    if savings_rate > 20:
        recommendation = "Great job! You're saving well."
    elif savings_rate > 10:
        recommendation = "Good progress. Consider cutting discretionary spending."
    elif savings_rate > 0:
        recommendation = "You're saving, but there's room to improve."
    else:
        recommendation = "Warning: You're projected to overspend. Review your budget!"

    return PredictionResponse(
        predicted_savings=round(prediction, 2),
        recommendation=recommendation
    )

@app.get("/example")
def example():
    """Example input for testing"""
    return {
        "income": 5000,
        "total_expenses": 3500,
        "num_transactions": 45,
        "groceries": 400,
        "restaurants": 250,
        "utilities": 200,
        "shopping": 150,
        "mortgage": 1200
    }
