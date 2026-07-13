# 📈 Stock Price Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red.svg)]()
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

## 🌐 Live Demo

**Streamlit App:**  
https://stock-price-prediction-anshika.streamlit.app/

---

# 📌 Project Overview

This project is a Machine Learning-based Stock Price Prediction System developed using historical Apple (AAPL) stock market data.

The application performs feature engineering on historical stock prices, trains an XGBoost regression model, and predicts the **next-day stock return**, which is then converted into the predicted next closing price.

A Streamlit dashboard has been developed to visualize the dataset, display stock trends, and interactively generate predictions.

---

# 🎯 Objectives

- Analyze historical stock market data
- Perform feature engineering using technical indicators
- Train a machine learning model for next-day return prediction
- Predict the next day's closing price
- Deploy the model using Streamlit Cloud

---

# 📂 Dataset

**Dataset Used**

- Apple Inc. (AAPL) Historical Stock Prices
- Daily OHLCV Data

Features include:

- Date
- Open
- High
- Low
- Close
- Volume

The project generates additional engineered features before model training.

---

# 🧠 Feature Engineering

The following technical indicators are generated:

### Price Features

- Daily Return
- Log Return

### Trend Indicators

- SMA 10
- SMA 20
- SMA 50
- SMA 200
- EMA 20

### Volatility

- Rolling Volatility (20 Days)

### Momentum Indicators

- RSI (14)

### MACD

- MACD
- MACD Signal

### Bollinger Bands

- Upper Band
- Lower Band

### Lag Features

- Close Lag 1
- Close Lag 2
- Close Lag 3
- Close Lag 5
- Close Lag 10

### Calendar Features

- Day of Week
- Month

---

# 🤖 Machine Learning Model

Model Used:

- XGBoost Regressor

Model Parameters

- n_estimators = 100
- learning_rate = 0.1
- max_depth = 6
- random_state = 42

The model predicts:

> **Next-Day Stock Return**

The predicted return is converted into:

> **Predicted Next Closing Price**

using:

```
Predicted Price = Current Close × (1 + Predicted Return)
```

---

# 📊 Project Workflow

```
Raw Stock Data
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Train/Test Split
        │
        ▼
Feature Scaling
        │
        ▼
XGBoost Model Training
        │
        ▼
Model Saving
        │
        ▼
Streamlit Deployment
```

---

# 🖥️ Streamlit Dashboard Features

The deployed application provides:

- Dataset Preview
- Interactive Prediction Button
- Predicted Next-Day Return
- Current Closing Price
- Predicted Next Closing Price
- Historical Closing Price Trend Visualization

---

# 📁 Project Structure

```
stock-price-prediction/
│
├── data/
│   ├── raw/
│   │   └── AAPL.csv
│   │
│   └── processed/
│       └── processed_stock.csv
│
├── models/
│   ├── stock_model.joblib
│   └── feature_scaler.joblib
│
├── notebooks/
│   ├── EDA.ipynb
│   └── model_training.ipynb
│
├── reports/
│   ├── correlation_heatmap.png
│   ├── eda_acf_pacf.png
│   ├── eda_price_trend.png
│   ├── eda_return_distribution.png
│   ├── eda_seasonal_decomposition.png
│   └── eda_volatility.png
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── feature_engineering.py
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/vatsanshika923-prog/stock-price-prediction.git

cd stock-price-prediction
```

Create Virtual Environment

```bash
python -m venv venv
```

Activate Environment

Windows

```bash
venv\Scripts\activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 🚀 Deployment

The application is deployed using **Streamlit Community Cloud**.

Live Application:

https://stock-price-prediction-anshika.streamlit.app/

---

# 🛠️ Technologies Used

Programming Language

- Python

Machine Learning

- XGBoost
- Scikit-learn

Data Processing

- Pandas
- NumPy

Visualization

- Plotly
- Matplotlib

Deployment

- Streamlit

Model Persistence

- Joblib

---

# 📈 Future Improvements

Potential enhancements include:

- Live stock market data integration
- Multi-stock prediction
- Real-time predictions using APIs
- LSTM and Transformer-based forecasting
- Hyperparameter optimization
- Model explainability using SHAP
- Confidence interval estimation
- Downloadable prediction reports
- Portfolio analysis dashboard

---

# ⚠️ Disclaimer

This project is intended for educational and research purposes only.

The predictions generated by this application should **not** be considered financial advice or investment recommendations.

---

# ⭐ Support

If you found this project helpful, consider giving the repository a ⭐ on GitHub.

It helps support future development and improvements.