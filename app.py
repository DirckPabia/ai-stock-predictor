import streamlit as st
import pandas as pd
import yfinance as yf

from signals import add_signals, apply_voting_strategy

st.title("📈 AI Stock Predictor")

# Get user input
# Radio selector for market type
market_type = st.radio(
    "Choose a market type:",
    ["🇵🇭 PH Stocks", "🌍 Global", "📝 Custom"],
    horizontal=True
)

# Dropdown options per market
if market_type == "🇵🇭 PH Stocks":
    ticker = st.selectbox("Select a PH stock:", [
        "ALI.PS",  # Ayala Land
        "AC.PS",   # Ayala Corporation
        "SM.PS",   # SM Investments
        "BPI.PS",  # Bank of the Philippine Islands
        "TEL.PS"   # PLDT
    ])
elif market_type == "🌍 Global":
    ticker = st.selectbox("Select a global stock:", [
        "AAPL",  # Apple
        "MSFT",  # Microsoft
        "GOOGL", # Alphabet
        "AMZN",  # Amazon
        "TSLA"   # Tesla
    ])
else:
    ticker = st.text_input("Enter a custom stock ticker (e.g. NVDA, JFC.PS):", value="AAPL")

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
