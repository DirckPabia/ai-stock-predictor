import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

from signals import add_signals, apply_voting_strategy

st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.title("📈 AI Stock Predictor")

# 1️⃣ Market selection
market_type = st.radio("Choose a market type:", ["🇵🇭 PH Stocks", "🌍 Global", "📝 Custom"], horizontal=True)

# 2️⃣ Ticker selection
if market_type == "🇵🇭 PH Stocks":
    ticker = st.selectbox("Select a PH stock:", ["Jollibee (JFC)": "JFC.PS", "Ayala Land (ALI)": "ALI.PS", "SM Prime (SMPH)": "SMPH.PS",
    "BDO Unibank (BDO)": "BDO.PS", "Ayala Corp (AC)": "AC.PS", "Globe Telecom (GLO)": "GLO.PS",
    "PLDT (TEL)": "TEL.PS", "URC (URC)": "URC.PS", "Meralco (MER)": "MER.PS"])
elif market_type == "🌍 Global":
    ticker = st.selectbox("Select a global stock:", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"])
else:
    ticker = st.text_input("Enter a custom ticker (e.g. NVDA, JFC.PS):", value="AAPL")

# 3️⃣ Date range
start = st.date_input("Start date", value=pd.to_datetime("2023-01-01"))
end = st.date_input("End date", value=pd.to_datetime("today"))

# 4️⃣ Strategy toggles
st.markdown("### 🧠 Strategy Configuration")
use_macd = st.checkbox("Use MACD", True)
use_bb = st.checkbox("Use Bollinger Bands", True)
use_stoch = st.checkbox("Use Stochastic Oscillator", True)
use_rsi = st.checkbox("Use RSI", True)
use_adx = st.checkbox("Use ADX", False)

# 5️⃣ Run
if st.button("Run Strategy"):
    try:
        st.info(f"📡 Downloading data for {ticker}...")
        data = yf.download(ticker, start=start, end=end)
        


        if data.empty:
            st.warning("⚠️ No data found. Please check your ticker or date range.")
        else:
            st.subheader("📄 Raw Price Data")
            st.dataframe(data.tail())

            # ➕ Add signals
            data = add_signals(data)

            # 🧠 Apply strategy
            data = apply_voting_strategy(
                data,
                use_macd=use_macd,
                use_bb=use_bb,
                use_stoch=use_stoch,
                use_rsi=use_rsi,
                use_adx=use_adx
            )

            st.subheader("📊 Signal-Enhanced Data")
            st.dataframe(data.tail())

            # 📈 Plot signals
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

                # Optional MACD
                if 'MACD' in df:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['MACD'],
                        name='MACD',
                        line=dict(color='blue', dash='dot'),
                        yaxis='y2'
                    ))

                fig.update_layout(
                    title="📈 Price & Buy Signals",
                    xaxis_rangeslider_visible=False,
                    yaxis=dict(title='Price'),
                    yaxis2=dict(
                        title='MACD',
                        overlaying='y',
                        side='right',
                        showgrid=False
                    ),
                    height=600
                )
                return fig

            st.plotly_chart(plot_signals(data), use_container_width=True)

            # 📣 Summary
            signals_fired = data[data['Composite_Signal'] == 1]
            st.markdown(f"### ✅ Buy signals fired: **{len(signals_fired)}** times")
            if not signals_fired.empty:
                st.dataframe(signals_fired[['Close', 'Votes']].tail())

    except Exception as e:
        st.error(f"🚨 Something went wrong:\n\n{e}")
