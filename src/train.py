import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib

def load_and_prepare_data():
    """Load and prepare monthly aggregated features"""
    transactions = pd.read_csv('data/raw/personal_transactions.csv')
    transactions['Date'] = pd.to_datetime(transactions['Date'])
    transactions['YearMonth'] = transactions['Date'].dt.to_period('M')

    income = transactions[transactions['Transaction Type'] == 'credit']
    expenses = transactions[transactions['Transaction Type'] == 'debit']

    monthly_data = []
    for month in transactions['YearMonth'].unique():
        month_trans = transactions[transactions['YearMonth'] == month]
        month_income = income[income['YearMonth'] == month]['Amount'].sum()
        month_expenses = expenses[expenses['YearMonth'] == month]['Amount'].sum()

        category_spending = expenses[expenses['YearMonth'] == month].groupby('Category')['Amount'].sum()

        features = {
            'income': month_income,
            'total_expenses': month_expenses,
            'num_transactions': len(month_trans),
            'groceries': category_spending.get('Groceries', 0),
            'restaurants': category_spending.get('Restaurants', 0),
            'utilities': category_spending.get('Utilities', 0),
            'shopping': category_spending.get('Shopping', 0),
            'mortgage': category_spending.get('Mortgage & Rent', 0),
            'savings': month_income - month_expenses
        }
        monthly_data.append(features)

    return pd.DataFrame(monthly_data)

def train_model():
    mlflow.set_experiment("personal-finance-savings")

    with mlflow.start_run():
        df = load_and_prepare_data()

        feature_cols = ['income', 'total_expenses', 'num_transactions',
                       'groceries', 'restaurants', 'utilities', 'shopping', 'mortgage']
        X = df[feature_cols]
        y = df['savings']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mlflow.log_params({"n_estimators": 100, "model_type": "RandomForest"})
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2_score": r2})

        joblib.dump(model, "models/savings_predictor.pkl")
        mlflow.sklearn.log_model(model, "model")

        print(f"Model trained successfully!")
        print(f"MAE: ${mae:.2f}")
        print(f"RMSE: ${rmse:.2f}")
        print(f"R² Score: {r2:.3f}")

if __name__ == "__main__":
    train_model()
