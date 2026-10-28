method: signal-trade-ledger
family: backtest-conventions
keywords: trade ledger, trade analysis, signal to trade, next open, forced close, round trip, exit reason, golden cross, death cross, sma crossover, ema crossover, bollinger band strategy, trailing stop, cumulative return
---
## Formulas
1. Signal s_t uses data to close t; fill at open t+1; a last-bar signal is never filled.
2. q = floor((cash - F)/(O(1+c))); buy cash -= qO(1+c)+F; sell cash += qO(1-c)-F.
3. V_t = cash + q C_t; r_0 = 0; cumulative_t = prod(1+r), first row 1.0.
4. One record per round trip: entry/exit dates and prices, pnl = sell net - buy total, exit_reason.
5. Forced close at the last close (signal date = exec date = last day), with costs.
6. Cross up: f_{t-1} <= s_{t-1} and f_t > s_t; NaN on either day -> no cross.
## Variants to decide
- V1 fill_bar: execution bar of a close-t signal. Options: (a) next bar open; (b) next bar close; (c) same close. Default when silent: (a) - Pardo 2008 (no look-ahead).
- V2 forced_close: position open at the end. Options: (a) SELL at last close with pnl, counted as a trade; (b) mark to market only. Default when silent: (a) - Chan 2009.
- V3 cross_signal: crossover entry logic. Options: (a) event on cross, NaN warm-up gives no cross; (b) state signal fast > slow, enters on the first valid day. Default when silent: (a) - Murphy 1999.
- V4 cum_base: cumulative series base. Options: (a) prod(1+r) starting at 1.0; (b) prod(1+r)-1 starting at 0. Default when silent: (a) - Bacon 2008 (wealth relative).
- V5 sizing: share count. Options: (a) all-in whole shares on cash; (b) fractional equity units. Default when silent: (a) - Chan 2009.
## Invariants
- BUY/SELL alternate, counts equal; final capital = initial + sum pnl; analysis lists every round trip.
## Pitfalls
- Toolkit ledgers that drop the forced close; side labels not as asked.
## Sources
- Pardo 2008; Chan 2009; Murphy 1999; Bacon 2008.
