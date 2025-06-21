import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

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

        if data.empty:
            st.warning("No data returned for this ticker and date range.")
        else:
            st.subheader("Raw Price Data")
            st.dataframe(data.tail())

            # 🧠 Add signals & apply voting
            data = add_signals(data)
            data = apply_voting_strategy(data)

            st.subheader("Signal-Enhanced Data")
            st.dataframe(data.tail())

            # 📊 Optional plot
            def plot_signals(df):
                fig = go.Figure()

                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name="Price"
                ))

                # Buy signals
                buys = df[df['Composite_Signal'] == 1]
                fig.add_trace(go.Scatter(
                    x=buys.index,
                    y=buys['Close'],
                    mode='markers',
                    marker=dict(size=10, color='green', symbol='triangle-up'),
                    name='Buy Signal'
                ))

                # Optional: MACD
                if 'MACD' in df:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['MACD'],
                        line=dict(color='blue', dash='dot'),
                        name='MACD'
                    ))

                fig.update_layout(title="📊 Composite Signals Chart", xaxis_rangeslider_visible=False)
                return fig

            st.plotly_chart(plot_signals(data), use_container_width=True)
            st.success("✅ Analysis complete!")

    except Exception as e:
        st.error(f"🚨 Something went wrong:\n\n{e}")
