import pandas as pd
import numpy as np
import ta

def add_signals(df):
    df = df.copy()

    # --- MACD ---
    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

    # --- Bollinger Bands (safe fallback) ---
    bb_ready = False

    if 'Close' in df.columns:
        close_col = df['Close']
        try:
            if bool(close_col.isna().sum() == 0) and len(df) >= 20:
                bb_ready = True
        except Exception:
            bb_ready = False

    if bb_ready:
        try:
            bb = ta.volatility.BollingerBands(close=df['Close'], window=20)
            df['BB_High'] = bb.bollinger_hband()
            df['BB_Low'] = bb.bollinger_lband()
        except Exception:
            df['BB_High'] = np.nan
            df['BB_Low'] = np.nan
    else:
        df['BB_High'] = np.nan
        df['BB_Low'] = np.nan

    # --- Stochastic Oscillator ---
    if len(df) >= 14:
        try:
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df['%K'] = stoch.stoch()
            df['%D'] = stoch.stoch_signal()
        except Exception:
            df['%K'] = np.nan
            df['%D'] = np.nan
    else:
        df['%K'] = np.nan
        df['%D'] = np.nan

    # --- RSI ---
    try:
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    except Exception:
        df['RSI'] = np.nan

    # --- ADX ---
    try:
        df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
    except Exception:
        df['ADX'] = np.nan

    return df.dropna()


def apply_voting_strategy(df, use_macd=True, use_bb=True, use_stoch=True, use_rsi=True, use_adx=False):
    df = df.copy()
    df['Votes'] = 0

    if use_macd:
        df['Votes'] += ((df['MACD'] > df['MACD_Signal']) & 
                        (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))).astype(int)

    if use_bb and 'BB_Low' in df.columns:
        try:
            bb_votes = (df['Close'] < df['BB_Low']).astype(int)
            df['Votes'] = df['Votes'].add(bb_votes, fill_value=0)
        except Exception:
            pass  # Optional: log or warn


    if use_stoch:
        df['Votes'] += ((df['%K'] > df['%D']) & (df['%K'] < 20)).astype(int)

    if use_rsi:
        df['Votes'] += (df['RSI'] < 30).astype(int)

    if use_adx:
        df['Votes'] += (df['ADX'] > 25).astype(int)

    df['Composite_Signal'] = 0
    df.loc[df['Votes'] >= 2, 'Composite_Signal'] = 1  # Buy
    df.loc[df['Votes'] <= -2, 'Composite_Signal'] = -1  # Sell

    return df
