import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from email.message import EmailMessage
import smtplib

from signal_engine import prediction_signal, rsi_signal, ma_signal, combine_signals
from backtest_engine import run_backtest
from company_lookup import get_ph_companies, get_global_companies

# --- Config ---
st.set_page_config(page_title="AI Stock Predictor", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("📈 AI Stock Predictor & Backtester")
st.caption("Smart signals powered by LSTM, RSI, MA + portfolio simulation")

st.markdown("### 🔎 Stock Selection")

# Market toggle
market = st.radio("🌏 Choose Market", ["Philippines", "Global"], horizontal=True)

ph_companies = get_ph_companies()
global_companies = get_global_companies()

if market == "Philippines":
    company_selected = st.selectbox("🇵🇭 Select PH Company", list(ph_companies.keys()))
    symbol = ph_companies[company_selected]
else:
    company_selected = st.selectbox("🌐 Select Global Company", list(global_companies.keys()))
    symbol = global_companies[company_selected]

override = st.text_input("🔍 Custom Symbol (optional)").strip().upper()
ticker = override if override else symbol

st.markdown(f"🧾 Final Ticker: `{ticker}`")

# Dates and email alerts
st.markdown("### 📅 Dates & Alerts")
col1, col2, col3 = st.columns(3)
with col1:
    start = st.date_input("Start Date", pd.to_datetime("2010-01-01"))
with col2:
    end = st.date_input("End Date", pd.to_datetime("2025-01-01"))
with col3:
    enable_email = st.checkbox("📨 Enable Email Alerts")

if enable_email:
    col4, col5, col6 = st.columns(3)
    with col4:
        sender = st.text_input("Gmail Sender")
    with col5:
        receiver = st.text_input("Recipient Email")
    with col6:
        password = st.text_input("App Password", type="password")
else:
    sender = receiver = password = ""

# Strategy toggles
st.markdown("### 🎛 Signal Strategies")
use_pred = st.checkbox("🧠 LSTM Prediction", True)
use_rsi = st.checkbox("📉 RSI Indicator", True)
use_ma = st.checkbox("📊 Moving Averages", True)

st.markdown("### ⚙️ Strategy Sensitivity")
c1, c2, c3, c4 = st.columns(4)
with c1:
    threshold = st.slider("Prediction Threshold %", 1, 10, 2) / 100
with c2:
    rsi_low = st.slider("RSI Buy <", 10, 40, 30)
with c3:
    rsi_high = st.slider("RSI Sell >", 60, 90, 70)
with c4:
    ma_short = st.slider("Short MA", 5, 30, 20)

ma_long = st.slider("Long MA", 30, 100, 50)

# Predict & Backtest
if st.button("🚀 Predict & Backtest"):
    with st.spinner("🔄 Fetching data, training model, and running backtest..."):
        df = yf.download(ticker, start=start, end=end)
        if df.empty or len(df) < max(ma_long + 60, 100):
            st.error("❗ Not enough data for indicators. Try an earlier start date or another ticker.")
            st.stop()

        df['MA_Short'] = df['Close'].rolling(ma_short).mean()
        df['MA_Long'] = df['Close'].rolling(ma_long).mean()
        df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean()))
        df.dropna(inplace=True)

        features = df[['Close', 'MA_Long', 'RSI']].dropna().values
        if features.shape[0] == 0:
            st.error("⚠️ Indicators didn't produce usable data. Try a longer date range or another ticker.")
            st.stop()

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(features)

        X, y = [], []
        for i in range(60, len(scaled)):
            X.append(scaled[i-60:i])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)

        if len(X) < 100:
            st.error("⚠️ Not enough sequences to train LSTM.")
            st.stop()

        split = int(len(X)*0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

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

        signals = []
        if use_pred:
            signals.append(prediction_signal(actual_price, pred_price, threshold))
        if use_rsi:
            signals.append(rsi_signal(df['RSI'].values[-len(pred_price):], rsi_low, rsi_high))
        if use_ma:
            signals.append(ma_signal(df['MA_Short'].values[-len(pred_price):], df['MA_Long'].values[-len(pred_price):]))

        if not signals:
            st.error("❗ No strategies enabled.")
            st.stop()

        final_signal = combine_signals(*signals)
        results = run_backtest(final_signal, actual_price)

        st.markdown("### 💼 Performance Overview")
        colA, colB, colC = st.columns(3)
        colA.metric("Final Portfolio", f"₱{results['final_value']:,.2f}")
        colB.metric("ROI", f"{results['roi']:.2f}%")
        colC.metric("Drawdown", f"{results['max_drawdown']:.2f}%")

        st.markdown("### 📉 Equity Curve")
        fig2, ax2 = plt.subplots(figsize=(10,3))
        ax2.plot(results['history'], color="purple")
        ax2.set_title("Backtest Performance")
        st.pyplot(fig2)

        st.markdown("### 🧾 Latest Signals")
        preview = pd.DataFrame({
            "Actual": actual_price[-10:],
            "Predicted": pred_price[-10:],
            "Signal": final_signal[-10:],
            "Action": results['actions'][-10:]
        })
        st.dataframe(preview)

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
            except Exception as e:
                st.warning(f"⚠️ Email failed: {e}")

    st.success("✅ Prediction and backtest complete!")
