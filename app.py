import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from email.message import EmailMessage
import smtplib
from collections import Counter

from signal_engine import prediction_signal, rsi_signal, ma_signal, combine_signals
from backtest_engine import run_backtest

# --- Streamlit Setup ---
st.set_page_config("AI Stock Predictor", layout="wide")
st.title("📈 AI Stock Predictor + Multi-Signal Strategy + Backtest")

# --- Stock Selection ---
ph_stocks = {
    "Jollibee (JFC)": "JFC.PS", "Ayala Land (ALI)": "ALI.PS", "SM Prime (SMPH)": "SMPH.PS",
    "BDO Unibank (BDO)": "BDO.PS", "Ayala Corp (AC)": "AC.PS", "Globe Telecom (GLO)": "GLO.PS",
    "PLDT (TEL)": "TEL.PS", "URC (URC)": "URC.PS", "Meralco (MER)": "MER.PS"
}
us_sectors = {
    "Tech": ["AAPL", "MSFT", "NVDA"], "Finance": ["JPM", "BAC", "GS"], 
    "Energy": ["XOM", "CVX", "BP"]
}

c1, c2, c3 = st.columns(3)
with c1:
    ph_choice = st.selectbox("🇵🇭 PH Stock", list(ph_stocks.keys()))
with c2:
    sector = st.selectbox("🏦 US Sector", list(us_sectors.keys()))
with c3:
    us_choice = st.selectbox("🌍 US Stock", us_sectors[sector])

override = st.text_input("🔎 Custom Symbol (optional)", value="")
ticker = override.upper() if override else ph_stocks.get(ph_choice) or us_choice
st.markdown(f"**Ticker:** `{ticker}`")

# --- Date & Email ---
start = st.date_input("Start Date", pd.to_datetime("2010-01-01"))
end = st.date_input("End Date", pd.to_datetime("2025-01-01"))
sender = st.text_input("📤 Gmail")
receiver = st.text_input("📥 Recipient")
password = st.text_input("🔐 App Password", type="password")

# --- Strategy Toggles ---
st.markdown("### 🎛 Signal Strategies")
use_pred = st.checkbox("🧠 LSTM Prediction", True)
use_rsi = st.checkbox("📉 RSI Indicator", True)
use_ma = st.checkbox("📊 Moving Averages", True)

# --- Signal Sensitivity ---
st.markdown("### ⚙️ Sensitivity Controls")
threshold = st.slider("Prediction Threshold (%)", 1, 10, 2) / 100
rsi_low = st.slider("RSI Buy < ", 10, 40, 30)
rsi_high = st.slider("RSI Sell > ", 60, 90, 70)
ma_short = st.slider("Short-Term MA", 5, 30, 20)
ma_long = st.slider("Long-Term MA", 30, 100, 50)

# --- Run ---
if st.button("🚀 Predict & Backtest"):
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        st.error("No data found.")
        st.stop()

    df['MA_Short'] = df['Close'].rolling(ma_short).mean()
    df['MA_Long'] = df['Close'].rolling(ma_long).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean()))
    df.dropna(inplace=True)

    # LSTM inputs
    features = df[['Close', 'MA_Long', 'RSI']].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)

    # Train/Test split
    if len(X) < 100:
        st.error("Not enough data to train.")
        st.stop()
    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # LSTM model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    preds = model.predict(X_test)
    pred_price = scaler.inverse_transform(np.hstack((preds, np.zeros((len(preds), 2)))))[:, 0]
    actual_price = scaler.inverse_transform(np.hstack((y_test.reshape(-1,1), np.zeros((len(y_test), 2)))))[:, 0]

    # --- Signal generation ---
    signals = []
    if use_pred: signals.append(prediction_signal(actual_price, pred_price, threshold))
    if use_rsi: signals.append(rsi_signal(df['RSI'].values[-len(pred_price):], rsi_low, rsi_high))
    if use_ma: signals.append(ma_signal(df['MA_Short'].values[-len(pred_price):], df['MA_Long'].values[-len(pred_price):]))
    if not signals:
        st.error("Please enable at least one strategy.")
        st.stop()

    final_signal = combine_signals(*signals)

    # --- Backtest ---
    results = run_backtest(final_signal, actual_price)
    st.metric("💼 Final Portfolio", f"{results['final_value']:,.2f}")
    st.metric("📈 ROI", f"{results['roi']:.2f}%")
    st.metric("🎯 Win Rate", f"{results['win_rate']:.1f}%")
    st.metric("📉 Max Drawdown", f"{results['max_drawdown']:.2f}%")

    # --- Plot signals & equity ---
    fig1, ax1 = plt.subplots(figsize=(12,5))
    ax1.plot(actual_price, label="Actual", color="blue")
    ax1.plot(pred_price, label="Predicted", color="orange")
    colors = ['green' if s == 'Buy' else 'red' if s == 'Sell' else 'gray' for s in final_signal]
    ax1.scatter(range(len(pred_price)), pred_price, c=colors, alpha=0.4, label="Signal")
    ax1.set_title("Price vs Prediction")
    ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10,3))
    ax2.plot(results['history'], color="purple")
    ax2.set_title("📊 Equity Curve")
    st.pyplot(fig2)

    # --- Table preview ---
    st.dataframe(pd.DataFrame({
        "Actual": actual_price[-10:],
        "Predicted": pred_price[-10:],
        "Signal": final_signal[-10:],
        "Action": results['actions'][-10:]
    }))

    # --- Email Signal Alert ---
    if sender and receiver and password and final_signal[-1] in ['Buy', 'Sell']:
        msg = f"Signal: {final_signal[-1]}\nActual: {actual_price[-1]:.2f}\nPredicted: {pred_price[-1]:.2f}\nROI: {results['roi']:.2f}%"
        try:
            email = EmailMessage()
            email['Subject'] = f"[{ticker}] Signal Alert: {final_signal[-1]}"
            email['From'] = sender
            email['To'] = receiver
            email.set_content(msg)
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(sender, password)
                smtp.send_message(email)
            st.success("📩 Signal email sent!")
        except:
            st.warning("Email failed. Check credentials.")
