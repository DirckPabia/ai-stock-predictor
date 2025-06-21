from collections import Counter

def prediction_signal(actual, predicted, threshold=0.02):
    return [
        'Buy' if p > a * (1 + threshold) else
        'Sell' if p < a * (1 - threshold) else
        'Hold'
        for a, p in zip(actual, predicted)
    ]

def rsi_signal(rsi_series, rsi_low=30, rsi_high=70):
    return [
        'Buy' if r < rsi_low else
        'Sell' if r > rsi_high else
        'Hold'
        for r in rsi_series
    ]

def ma_signal(short_ma, long_ma):
    return [
        'Buy' if s > l else
        'Sell' if s < l else
        'Hold'
        for s, l in zip(short_ma, long_ma)
    ]

def combine_signals(*layers):
    return [
        Counter(signals).most_common(1)[0][0]
        for signals in zip(*layers)
    ]
