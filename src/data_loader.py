import yfinance as yf
import pandas as pd

TICKER = "AAPL"
START = "2010-01-01"
END = "2025-12-31"
RAW_PATH = "data/raw/AAPL.csv"


def download_data(ticker: str = TICKER, start: str = START, end: str = END) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    return df


def save_raw(df: pd.DataFrame, path: str = RAW_PATH) -> None:
    df.to_csv(path, index=False)
    print(f"Saved raw data: {df.shape} -> {path}")


if __name__ == "__main__":
    df = download_data()
    save_raw(df)
