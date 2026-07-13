import joblib
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

from feature_engineering import build_features  # same pipeline used in training

def fetch_live_data(ticker: str = "AAPL", lookback_days: int = 300) -> pd.DataFrame:
    """
    Pull recent data from Yahoo Finance. lookback_days needs to be generous
    (300+ calendar days) because SMA_200 needs 200 TRADING days of history,
    and trading days are fewer than calendar days (weekends/holidays).
    """
    end = datetime.today()
    start = end - timedelta(days=lookback_days)

    raw = yf.download(ticker, start=start, end=end)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    return raw

def get_latest_features(ticker: str = "AAPL") -> pd.DataFrame:
    """Fetch live data and run it through the same feature pipeline as training."""
    raw = fetch_live_data(ticker)
    df = build_features(raw)
    df = df.dropna().reset_index(drop=True)
    return df

def load_model(model_name: str = "random_forest"):
    """model_name: 'random_forest' or 'xgboost'"""
    filename = "random_forest.pkl" if model_name == "random_forest" else "xgboost.pkl"
    model = joblib.load(MODEL_DIR / filename)
    feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
    return model, feature_cols

def predict_next_close(model_name: str = "random_forest", ticker: str = "AAPL"):
    """
    Returns (predicted_price, current_price, as_of_date).
    Model predicts a RETURN, which is converted back to a price here.
    """
    model, feature_cols = load_model(model_name)
    df = get_latest_features(ticker)

    latest_row = df[feature_cols].tail(1)
    predicted_return = model.predict(latest_row)[0]

    current_price = df["Close"].iloc[-1]
    predicted_price = current_price * (1 + predicted_return)
    as_of_date = df["Date"].iloc[-1]

    return predicted_price, current_price, as_of_date

if __name__ == "__main__":
    price, current, date = predict_next_close("random_forest")
    print(f"As of {date.date()}: current close = ${current:.2f}")
    print(f"Predicted next close = ${price:.2f}")
