import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from signals import add_signals, apply_voting_strategy

st.set_page_config(page_title="🧠 AI Stock Predictor", layout="wide")

# Sidebar Controls
st.sidebar.header("📈 Ticker & Date")
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
