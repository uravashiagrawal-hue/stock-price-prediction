import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

st.title("📈 Stock Price Prediction")
st.write("This project predicts the next-day stock return using a trained XGBoost model.")

# Load model
model = joblib.load("models/stock_model.joblib")

# Load processed dataset
df = pd.read_csv("data/processed/processed_stock.csv")

st.subheader("Dataset Preview")
st.dataframe(df.tail())

# Prepare features
feature_cols = [
    c for c in df.columns
    if c not in ["Date", "Target_Next_Close", "Target_Next_Return"]
]

latest = df.iloc[-1][feature_cols]

if st.button("Predict Next Day Return"):
    prediction = model.predict([latest])[0]

    st.success(f"Predicted Next Day Return: {prediction:.4f}")

    current_price = df.iloc[-1]["Close"]
    predicted_price = current_price * (1 + prediction)

    st.metric("Current Close Price", f"${current_price:.2f}")
    st.metric("Predicted Next Close Price", f"${predicted_price:.2f}")

st.subheader("Closing Price Trend")
st.line_chart(df.set_index("Date")["Close"])