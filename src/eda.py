import pandas as pd
import numpy as np


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Daily_Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    for window in [10, 20, 50, 200]:
        df[f"SMA_{window}"] = df["Close"].rolling(window).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    return df


def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    df["Volatility_20"] = df["Daily_Return"].rolling(20).std()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    sma = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()
    df["BB_Upper"] = sma + 2 * std
    df["BB_Lower"] = sma - 2 * std
    return df


def add_lag_features(df: pd.DataFrame, lags=(1, 2, 3, 5, 10)) -> pd.DataFrame:
    for lag in lags:
        df[f"Close_lag_{lag}"] = df["Close"].shift(lag)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df["Target_Next_Close"] = df["Close"].shift(-1)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_return_features(df)
    df = add_moving_averages(df)
    df = add_volatility(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_lag_features(df)
    df = add_calendar_features(df)
    df = add_target(df)
    return df


if __name__ == "__main__":
    from preprocessing import load_and_clean

    df = load_and_clean("../data/raw/AAPL.csv")
    df = build_features(df)
    df = df.dropna().reset_index(drop=True)
    df.to_csv("../data/processed/processed_stock.csv", index=False)
    print(f"Processed dataset: {df.shape} -> ../data/processed/processed_stock.csv")
