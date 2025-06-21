import streamlit as st
import pandas as pd
import yfinance as yf

from signals import add_signals, apply_voting_strategy

st.title("📈 AI Stock Predictor")

# Get user input
ticker = st.text_input("Enter a stock ticker (e.g. AAPL):", value="AAPL")
start = st.date_input("Start date", value=pd.to_datetime("2023-01-01"))
end = st.date_input("End date", value=pd.to_datetime("today"))

if st.button("Run Strategy"):
    try:
        # ✅ Download full OHLCV data
        data = yf.download(ticker, start=start, end=end)

        # ✅ Optional: Display raw data preview
        st.subheader("Raw Price Data")
        st.dataframe(data.tail())

        # ✅ Add technical signals (this will validate structure)
        data = add_signals(data)

        # ✅ Apply voting strategy (can add toggles here later)
        data = apply_voting_strategy(data)

        st.subheader("Signal-Enhanced Data")
        st.dataframe(data.tail())

        # Optional: Plot MACD or signals here
    except Exception as e:
        st.error(f"🚨 Something went wrong: {e}")
