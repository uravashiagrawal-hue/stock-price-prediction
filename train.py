import pandas as pd
import joblib

from src.preprocessing import load_and_clean, time_based_split, scale_features
from src.feature_engineering import build_features

from xgboost import XGBRegressor

DATA_PATH = "data/raw/AAPL.csv"

df = load_and_clean(DATA_PATH)
df = build_features(df)

df = df.dropna().reset_index(drop=True)

df["Target_Next_Return"] = df["Close"].shift(-1) / df["Close"] - 1
df = df.dropna().reset_index(drop=True)

feature_cols = [
    c for c in df.columns
    if c not in ["Date", "Target_Next_Close", "Target_Next_Return"]
]

X_train, X_test, y_train, y_test, train_df, test_df = time_based_split(
    df,
    target_col="Target_Next_Return",
    feature_cols=feature_cols
)

X_train_scaled, X_test_scaled, scaler = scale_features(
    X_train,
    X_test
)

model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "models/stock_model.joblib")
joblib.dump(scaler, "models/feature_scaler.joblib")

print("SUCCESS")
print("Model saved to models/stock_model.joblib")