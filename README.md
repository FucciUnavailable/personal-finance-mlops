## Setup
1. Get Kaggle API credentials
2. Run: `uv run kaggle datasets download -d bukolafatunde/personal-finance -p data/raw --unzip`

I used this data because it seemed the closest to what I need with my current setup

# Personal Finance MLOps Pipeline

A complete MLOps project for predicting monthly savings based on transaction patterns. Built with industry best practices: Docker, MLflow, FastAPI, and automated testing.

## 🎯 Project Goal

Train a machine learning model to predict monthly savings and provide budget recommendations based on spending patterns.

## 🏗️ Architecture
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Kaggle    │────▶│   Training   │────▶│   MLflow    │
│   Dataset   │     │   Pipeline   │     │  Tracking   │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Saved Model  │
                    │   (.pkl)     │
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐     ┌─────────────┐
                    │   FastAPI    │────▶│   Client    │
                    │   Service    │     │  (HTTP)     │
                    └──────────────┘     └─────────────┘
```

## 📊 Model Details

- **Algorithm**: Random Forest Regressor
- **Features**: Income, expenses, transaction count, category breakdowns
- **Target**: Monthly savings ($)
- **Performance**:
  - MAE: $1,423
  - RMSE: $2,429
  - R² Score: 0.642

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Kaggle API credentials

### Setup

1. **Clone the repository**
```bash
git clone <your-repo>
cd personal-finance-mlops
```

2. **Set up Kaggle credentials**
```bash
mkdir -p ~/.kaggle
# Add your kaggle.json with API credentials
chmod 600 ~/.kaggle/kaggle.json
```

3. **Download dataset**
```bash
docker-compose run --rm train uv run kaggle datasets download -d bukolafatunde/personal-finance -p data/raw --unzip
```

### Usage

**Train the model:**
```bash
docker-compose run --rm train
```

**View MLflow tracking:**
```bash
docker-compose --profile mlflow up
# Open http://localhost:5000
```

**Run the API:**
```bash
docker-compose --profile api up
# Open http://localhost:8000/docs
```

**Run tests:**
```bash
docker-compose run --rm train uv run pytest tests/ -v
```

**Explore data (Jupyter):**
```bash
docker-compose up jupyter
# Open http://localhost:8888
```

## 📡 API Endpoints

### `POST /predict`
Predict monthly savings based on spending patterns.

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "income": 5000,
    "total_expenses": 3500,
    "num_transactions": 45,
    "groceries": 400,
    "restaurants": 250,
    "utilities": 200,
    "shopping": 150,
    "mortgage": 1200
  }'
```

**Response:**
```json
{
  "predicted_savings": 901.53,
  "recommendation": "Good progress. Consider cutting discretionary spending."
}
```

## 🛠️ Tech Stack

- **Python 3.12** - Programming language
- **uv** - Fast Python package manager
- **scikit-learn** - Machine learning
- **MLflow** - Experiment tracking & model registry
- **FastAPI** - REST API framework
- **Docker** - Containerization
- **pytest** - Testing
- **Jupyter** - Data exploration

## 📁 Project Structure
```
personal-finance-mlops/
├── data/
│   ├── raw/              # Kaggle dataset
│   └── processed/        # Processed features
├── models/               # Trained models
│   └── savings_predictor.pkl
├── notebooks/            # Jupyter notebooks
│   └── 01_explore_data.ipynb
├── src/
│   ├── train.py         # Training pipeline
│   ├── api.py           # FastAPI service
│   └── preprocessing.py # Data preprocessing
├── tests/
│   └── test_api.py      # API tests
├── Dockerfile           # Container definition
├── docker-compose.yml   # Multi-service orchestration
└── README.md
```

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ End-to-end ML pipeline design
- ✅ Experiment tracking with MLflow
- ✅ Model serving with FastAPI
- ✅ Containerization with Docker
- ✅ Automated testing
- ✅ Clean project structure
- ✅ Reproducible workflows

## 🔜 Future Improvements

- [ ] Add DVC for data versioning
- [ ] Implement CI/CD with GitHub Actions
- [ ] Deploy to cloud (AWS/GCP)
- [ ] Add monitoring & logging
- [ ] Build web frontend
- [ ] Implement A/B testing framework

## 📝 License

MIT

## 👤 Author
Fucci
Built as a learning project for MLOps certification preparation.
