def run_backtest(final_signal, actual_price, starting_capital=100000):
    balance = starting_capital
    shares = 0
    history = []
    actions = []
    entry_price = None
    wins, total_trades = 0, 0
    peak = starting_capital
    drawdowns = []

    for i, signal in enumerate(final_signal):
        price = actual_price[i]

        if signal == 'Buy' and shares == 0:
            shares = balance / price
            entry_price = price
            balance = 0
            actions.append('BUY')
        elif signal == 'Sell' and shares > 0:
            balance = shares * price
            if entry_price:
                if price > entry_price:
                    wins += 1
            shares = 0
            total_trades += 1
            entry_price = None
            actions.append('SELL')
        else:
            actions.append('HOLD')

        portfolio_value = balance + shares * price
        history.append(portfolio_value)
        if portfolio_value > peak:
            peak = portfolio_value
        drawdowns.append((peak - portfolio_value) / peak)

    roi = (history[-1] - starting_capital) / starting_capital * 100
    max_dd = max(drawdowns) * 100 if drawdowns else 0
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    return {
        "history": history,
        "actions": actions,
        "final_value": history[-1],
        "roi": roi,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "trades": total_trades
    }
