import pandas as pd
import numpy as np
import ta  # Technical Analysis library

def add_signals(df):
    df = df.copy()

    required_cols = ['Close', 'High', 'Low']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Ensure all are Series and numeric
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            print(f"DEBUG: df[{col}] = {df[col]}")
            raise TypeError(f"Failed to convert column '{col}' to numeric: {e}")


    df.dropna(subset=required_cols, inplace=True)

    # MACD
    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

    # Bollinger Bands
    df['BB_High'] = np.nan
    df['BB_Low'] = np.nan
    if len(df) >= 20:
        try:
            bb = ta.volatility.BollingerBands(close=df['Close'], window=20)
            df['BB_High'] = bb.bollinger_hband()
            df['BB_Low'] = bb.bollinger_lband()
        except Exception:
            pass

    # Stochastic Oscillator
    df['%K'], df['%D'] = np.nan, np.nan
    if len(df) >= 14:
        try:
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df['%K'] = stoch.stoch()
            df['%D'] = stoch.stoch_signal()
        except Exception:
            pass

    # RSI
    try:
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    except Exception:
        df['RSI'] = np.nan

    # ADX
    try:
        df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
    except Exception:
        df['ADX'] = np.nan

    return df

def apply_voting_strategy(df, use_macd=True, use_bb=True, use_stoch=True, use_rsi=True, use_adx=False):
    df = df.copy()
    df['Votes'] = 0

    # MACD crossover
    if use_macd and 'MACD' in df and 'MACD_Signal' in df:
        macd_cross = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
        df['Votes'] += macd_cross.astype(int)

    # Bollinger Bounce
    if use_bb and 'BB_Low' in df:
        df['Votes'] += (df['Close'] < df['BB_Low']).astype(int)

    # Stochastic Oversold Crossover
    if use_stoch and '%K' in df and '%D' in df:
        k_cross = (df['%K'] > df['%D']) & (df['%K'] < 20)
        df['Votes'] += k_cross.astype(int)

    # RSI Oversold
    if use_rsi and 'RSI' in df:
        df['Votes'] += (df['RSI'] < 30).astype(int)

    # ADX Threshold
    if use_adx and 'ADX' in df:
        df['Votes'] += (df['ADX'] > 25).astype(int)

    df['Composite_Signal'] = 0
    df.loc[df['Votes'] >= 2, 'Composite_Signal'] = 1
    df.loc[df['Votes'] <= -2, 'Composite_Signal'] = -1

    return df
