"""
app/app.py
Streamlit demo -- predicts tomorrow's AAPL close using LIVE market data.

Run with: streamlit run app.py   (from inside the app/ folder)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st
from predict import predict_next_close, get_latest_features

st.set_page_config(page_title="AAPL Next-Day Price Predictor", layout="centered")
st.title("AAPL Next-Day Close Price Predictor")
st.caption("Predictions use live market data, refreshed hourly.")


# Cache the live fetch for 1 hour so we don't call Yahoo Finance on every
# page load / every visitor -- avoids rate limiting and keeps the app fast.
@st.cache_data(ttl=3600)
def cached_prediction(model_name: str):
    return predict_next_close(model_name)


@st.cache_data(ttl=3600)
def cached_chart_data():
    return get_latest_features()


df = cached_chart_data()
st.write(f"Latest data through: **{df['Date'].max().date()}**")
st.line_chart(df.set_index("Date")["Close"].tail(180))

model_choice = st.radio("Choose model:", ["Random Forest", "XGBoost"])
model_key = "random_forest" if model_choice == "Random Forest" else "xgboost"

if st.button("Predict Next Close"):
    with st.spinner("Fetching live data and predicting..."):
        predicted_price, current_price, as_of_date = cached_prediction(model_key)

    delta = predicted_price - current_price
    st.metric(
        label=f"Predicted Next Close (as of {as_of_date.date()})",
        value=f"${predicted_price:.2f}",
        delta=f"{delta:+.2f}",
    )
    st.caption(f"Current close: ${current_price:.2f}")

st.divider()
st.caption(
    "This is a college project demo, not financial advice. "
    "Predictions are based on historical patterns and technical indicators only."
)
