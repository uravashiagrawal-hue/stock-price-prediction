import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    missing = df.isnull().sum()
    if missing.any():
        print("Missing values found:\n", missing[missing > 0])

    gaps = df["Date"].diff().dt.days
    large_gaps = gaps[gaps > 5]
    if not large_gaps.empty:
        print(f"Note: {len(large_gaps)} date gaps larger than 5 days (holidays/splits/etc).")

    return df


def time_based_split(df: pd.DataFrame, target_col: str, feature_cols: list, split_ratio: float = 0.85):
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    return X_train, X_test, y_train, y_test, train_df, test_df


def scale_features(X_train, X_test, save_path: str = "../models/feature_scaler.joblib"):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, save_path)
    return X_train_scaled, X_test_scaled, scaler
