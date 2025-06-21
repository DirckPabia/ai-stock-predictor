import pandas as pd
import numpy as np
import ta

def add_signals(df):
    df = df.copy()

    # Clean essential columns
    required_cols = ['Close', 'High', 'Low']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        series = df[col]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
            if not isinstance(col, str):
                raise TypeError(f"Column name must be a string, got {type(col)}")
            df[col] = pd.to_numeric(df[col].squeeze(), errors='coerce')



    df.dropna(subset=required_cols, inplace=True)

    # MACD
    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

    # Bollinger Bands
    df['BB_High'], df['BB_Low'] = np.nan, np.nan
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

    return df.dropna()

def apply_voting_strategy(df, use_macd=True, use_bb=True, use_stoch=True, use_rsi=True, use_adx=False):
    df = df.copy()
    df['Votes'] = 0

    # MACD logic
    if use_macd and {'MACD', 'MACD_Signal'}.issubset(df.columns):
        mask = df['MACD'].notna() & df['MACD_Signal'].notna()
        crossover = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
        df.loc[mask, 'Votes'] += crossover[mask].astype(int)

    # Bollinger logic
    if use_bb and 'BB_Low' in df.columns:
        bb_votes = (df['Close'] < df['BB_Low']).astype(int)
        df['Votes'] = df['Votes'].add(bb_votes, fill_value=0)

    # Stochastic logic
    if use_stoch and {'%K', '%D'}.issubset(df.columns):
        mask = df['%K'].notna() & df['%D'].notna()
        stoch_cond = (df['%K'] > df['%D']) & (df['%K'] < 20)
        df.loc[mask, 'Votes'] += stoch_cond[mask].astype(int)

    # RSI logic
    if use_rsi and 'RSI' in df.columns:
        rsi_votes = (df['RSI'] < 30).astype(int)
        df['Votes'] = df['Votes'].add(rsi_votes, fill_value=0)

    # ADX logic
    if use_adx and 'ADX' in df.columns:
        adx_votes = (df['ADX'] > 25).astype(int)
        df['Votes'] = df['Votes'].add(adx_votes, fill_value=0)

    # Composite Signal
    df['Composite_Signal'] = 0
    df.loc[df['Votes'] >= 2, 'Composite_Signal'] = 1
    df.loc[df['Votes'] <= -2, 'Composite_Signal'] = -1

    return df
