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
 