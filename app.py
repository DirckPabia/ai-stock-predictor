import streamlit as st
import pandas as pd
import yfinance as yf

from signals import add_signals, apply_voting_strategy

st.title("📈 AI Stock Predictor")

# 1️⃣ Market selection
market_type = st.radio(
    "Choose a market type:",
    ["🇵🇭 PH Stocks", "🌍 Global", "📝 Custom"],
    horizontal=True
)

# 2️⃣ Ticker selection based on market
if market_type == "🇵🇭 PH Stocks":
    ticker = st.selectbox("Select a PH stock:", [
        "ALI.PS", "AC.PS", "SM.PS", "BPI.PS", "TEL.PS"
    ])
elif market_type == "🌍 Global":
    ticker = st.selectbox("Select a global stock:", [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"
    ])
else:
    ticker = st.text_input("Enter a custom stock ticker (e.g. NVDA, JFC.PS):", value="AAPL")

# 3️⃣ Date range
start = st.date_input("Start date", value=pd.to_datetime("2023-01-01"))
end = st.date_input("End date", value=pd.to_datetime("today"))

# 4️⃣ Run strategy
if st.button("Run Strategy"):
    try:
        st.info("📡 Downloading data...")
        data = yf.download(ticker, start=start, end=end)

        st.subheader("Raw Price Data")
        st.dataframe(data.tail())

        # 🎯 Add signals (structure validated inside)
        data = add_signals(data)

        # 🧠 Apply voting strategy
        data = apply_voting_strategy(data)

        st.subheader("Signal-Enhanced Data")
        st.dataframe(data.tail())

        st.success("✅ Analysis complete!")

    except Exception as e:
        st.error(f"🚨 Something went wrong:\n\n{e}")
