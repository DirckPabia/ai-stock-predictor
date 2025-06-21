import pandas as pd
import numpy as np
import ta

def add_signals(df):
    df = df.copy()

    # Ensure required columns exist and are numeric
    required_cols = ['Close', 'High', 'Low']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        df[col] = pd.to_numeric(df[col], errors='coerce')

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
    df['%K'] = np.nan
    df['%D'] = np.nan
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

    if use_macd and 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        macd_mask = df['MACD'].notna() & df['MACD_Signal'].notna()
        df.loc[macd_mask, 'Votes'] += ((df.loc[macd_mask, 'MACD'] > df.loc[macd_mask, 'MACD_Signal']) &
                                       (df.loc[macd_mask, 'MACD'].shift(1) <= df.loc[macd_mask, 'MACD_Signal'].shift(1))).astype(int)

    if use_bb and 'BB_Low' in df.columns:
        bb_mask = df['Close'].notna() & df['BB_Low'].notna()
        df.loc[bb_mask, 'Votes'] += (df.loc[bb_mask, 'Close'] < df.loc[bb_mask, 'BB_Low']).astype(int)

    if use_stoch and '%K' in df.columns and '%D' in df.columns:
        stoch_mask = df['%K'].notna() & df['%D'].notna()
        df.loc[stoch_mask, 'Votes'] += ((df.loc[stoch_mask, '%K'] > df.loc[stoch_mask, '%D']) &
                                        (df.loc[stoch_mask, '%K'] < 20)).astype(int)

    if use_rsi and 'RSI' in df.columns:
        rsi_mask = df['RSI'].notna()
        df.loc[rsi_mask, 'Votes'] += (df.loc[rsi_mask, 'RSI'] < 30).astype(int)

    if use_adx and 'ADX' in df.columns:
        adx_mask = df['ADX'].notna()
        df.loc[adx_mask, 'Votes'] += (df.loc[adx_mask, 'ADX'] > 25).astype(int)

    df['Composite_Signal'] = 0
    df.loc[df['Votes'] >= 2, 'Composite_Signal'] = 1
    df.loc[df['Votes'] <= -2, 'Composite_Signal'] = -1

    return df
