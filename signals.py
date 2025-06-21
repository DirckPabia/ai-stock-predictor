import pandas as pd
import numpy as np
import ta  # Make sure 'ta' is in your requirements.txt

def add_signals(df):
    df = df.copy()
    required_cols = ['Close', 'High', 'Low']

    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

        series = df[col]
        if isinstance(series, pd.DataFrame):
            raise TypeError(f"Expected Series for '{col}', got DataFrame. Use df['{col}'], not df[['{col}']]")

        try:
            df[col] = pd.to_numeric(series, errors='coerce')
        except Exception as e:
            raise TypeError(f"Failed to convert column '{col}' to numeric: {e}")

    df.dropna(subset=required_cols, inplace=True)

    # MACD
    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

    # Bollinger Bands
    try:
        bb = ta.volatility.BollingerBands(close=df['Close'], window=20)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
    except Exception:
        df['BB_High'], df['BB_Low'] = np.nan, np.nan

    # Stochastic Oscillator
    try:
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['%K'] = stoch.stoch()
        df['%D'] = stoch.stoch_signal()
    except Exception:
        df['%K'], df['%D'] = np.nan, np.nan

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

    if use_macd and 'MACD' in df and 'MACD_Signal' in df:
        macd_cross = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
        df['Votes'] += macd_cross.astype(int)

    if use_bb and 'BB_Low' in df:
        df['Votes'] += (df['Close'] < df['BB_Low']).astype(int)

    if use_stoch and '%K' in df and '%D' in df:
        stoch_cross = (df['%K'] > df['%D']) & (df['%K'] < 20)
        df['Votes'] += stoch_cross.astype(int)

    if use_rsi and 'RSI' in df:
        df['Votes'] += (df['RSI'] < 30).astype(int)

    if use_adx and 'ADX' in df:
        df['Votes'] += (df['ADX'] > 25).astype(int)

    df['Composite_Signal'] = 0
    df.loc[df['Votes'] >= 2, 'Composite_Signal'] = 1
    df.loc[df['Votes'] <= -2, 'Composite_Signal'] = -1

    return df
