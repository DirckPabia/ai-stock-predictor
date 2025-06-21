import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Stock Predictor", layout="wide")

# 1. Sidebar Inputs
st.sidebar.title("Trading Dashboard")
ticker = st.sidebar.text_input("Enter Ticker", "AAPL")
start = st.sidebar.date_input("From", pd.to_datetime("2023-01-01"))
end = st.sidebar.date_input("To", pd.Timestamp.today())

# Signal toggles
st.sidebar.header("Signals")
show_macd = st.sidebar.checkbox("MACD", True)
show_bb = st.sidebar.checkbox("Bollinger Bands", True)
show_stoch = st.sidebar.checkbox("Stochastic Oscillator", True)
show_rsi = st.sidebar.checkbox("RSI", True)
show_adx = st.sidebar.checkbox("ADX", False)

# 2. Load Data
data = yf.download(ticker, start=start, end=end)
data = data.dropna()

# 3. Technical Indicators
data['EMA12'] = data['Close'].ewm(span=12).mean()
data['EMA26'] = data['Close'].ewm(span=26).mean()
data['MACD'] = data['EMA12'] - data['EMA26']
data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()

bb = ta.volatility.BollingerBands(close=data['Close'], window=20)
data['BB_High'] = bb.bollinger_hband()
data['BB_Low'] = bb.bollinger_lband()

stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'])
data['%K'] = stoch.stoch()
data['%D'] = stoch.stoch_signal()

data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()

# 4. Voting Signal
data['Votes'] = 0
if show_macd:
    data['Votes'] += (data['MACD'] > data['MACD_Signal']) & (data['MACD'].shift(1) <= data['MACD_Signal'].shift(1))
if show_bb:
    data['Votes'] += data['Close'] < data['BB_Low']
if show_stoch:
    data['Votes'] += (data['%K'] > data['%D']) & (data['%K'] < 20)
if show_rsi:
    data['Votes'] += data['RSI'] < 30
if show_adx:
    data['Votes'] += data['ADX'] > 25

data['Composite_Signal'] = 0
data.loc[data['Votes'] >= 2, 'Composite_Signal'] = 1  # Buy
data.loc[data['Votes'] <= -2, 'Composite_Signal'] = -1  # Sell

# 5. Plotting
st.title(f"📈 {ticker} Technical Dashboard")
st.write(f"Showing data from **{start}** to **{end}**")

fig, ax = plt.subplots(figsize=(14,6))
ax.plot(data.index, data['Close'], label='Close Price', linewidth=2)

if show_bb:
    ax.plot(data.index, data['BB_High'], label='Bollinger High', linestyle='--', alpha=0.5)
    ax.plot(data.index, data['BB_Low'], label='Bollinger Low', linestyle='--', alpha=0.5)

if 'Prediction' in data.columns:
    ax.plot(data.index, data['Prediction'], label='LSTM Forecast', color='magenta')

buy_signals = data[data['Composite_Signal'] == 1]
ax.scatter(buy_signals.index, buy_signals['Close'], label='Buy Signal', marker='^', color='green', s=100)

ax.legend()
ax.set_title(f"{ticker} Price and Signals")
st.pyplot(fig)

# 6. Optional: Signal Table
st.subheader("📋 Signal Vote Summary")
st.dataframe(data[['Close', 'MACD', 'MACD_Signal', '%K', '%D', 'RSI', 'ADX', 'Votes', 'Composite_Signal']].tail(10))
