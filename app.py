import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from signals import add_signals, apply_voting_strategy

st.set_page_config(page_title="🧠 AI Stock Predictor", layout="wide")

# Philippine Stocks
ph_tickers = {
    "Ayala Corp (AC)": "AC.PS",
    "SM Investments (SM)": "SM.PS",
    "BDO Unibank (BDO)": "BDO.PS",
    "Jollibee Foods (JFC)": "JFC.PS"
}

# Global Stocks
global_tickers = {
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Amazon (AMZN)": "AMZN",
    "Google (GOOG)": "GOOG"
}

# Sidebar Controls
st.sidebar.header("📈 Ticker & Date")
st.sidebar.subheader("🔍 Ticker Input Method")
symbol_mode = st.sidebar.radio("Choose input type:", ["Dropdown", "Manual"])

if symbol_mode == "Dropdown":
    region = st.sidebar.radio("Market Region", ["Global", "Philippines"])

    if region == "Global":
        symbol = st.sidebar.selectbox("Global Stocks", list(global_tickers.keys()))
        ticker = global_tickers[symbol]
    else:
        symbol = st.sidebar.selectbox("PH Stocks", list(ph_tickers.keys()))
        ticker = ph_tickers[symbol]

else:
    ticker = st.sidebar.text_input("Enter Custom Symbol", "AAPL")

start = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end = st.sidebar.date_input("End Date", pd.Timestamp.today())

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Signal Strategies")
show_macd = st.sidebar.checkbox("MACD", True)
show_bb = st.sidebar.checkbox("Bollinger Bands", True)
show_stoch = st.sidebar.checkbox("Stochastic Oscillator", True)
show_rsi = st.sidebar.checkbox("RSI", True)
show_adx = st.sidebar.checkbox("ADX", False)

# Load Data
data = yf.download(ticker, start=start, end=end)

if data.empty:
    st.error("⚠️ No data found for that ticker and date range.")
    st.stop()

# Add Indicators & Signals
st.write("✅ Data type:", type(data))
st.write("✅ Data columns:", data.columns if hasattr(data, "columns") else "No columns")
if 'Close' in data:
    st.write("✅ Type of data['Close']:", type(data['Close']))
    st.write("✅ Sample of data['Close']:", data['Close'].head())
else:
    st.write("❌ 'Close' column is missing")
st.write("✅ Full data preview:")
st.write(data.head())

data = add_signals(data)
data = apply_voting_strategy(data, show_macd, show_bb, show_stoch, show_rsi, show_adx)

# Plot
st.title(f"📊 {ticker} Signal Dashboard")
fig, ax = plt.subplots(figsize=(14,6))
ax.plot(data.index, data['Close'], label="Close Price", linewidth=2)

if show_bb:
    ax.plot(data.index, data['BB_High'], linestyle='--', label='BB High', alpha=0.5)
    ax.plot(data.index, data['BB_Low'], linestyle='--', label='BB Low', alpha=0.5)

# Mark Buy Signals
buy_signals = data[data['Composite_Signal'] == 1]
ax.scatter(buy_signals.index, buy_signals['Close'], color='green', label='Buy Signal', marker='^', s=100)

ax.set_title(f"{ticker} Price with Signals")
ax.legend()
st.pyplot(fig)

# Snapshot Table
st.subheader("🔍 Recent Signal Snapshot")
st.dataframe(data[['Close', 'MACD', 'MACD_Signal', '%K', '%D', 'RSI', 'ADX', 'Votes', 'Composite_Signal']].tail(10))
