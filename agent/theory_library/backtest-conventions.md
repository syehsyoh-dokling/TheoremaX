family: backtest-conventions
category: backtesting
keywords: backtest, signal, next open, execution, crossover, golden cross, death cross, sma, ema, bollinger, rsi, macd, momentum, sharpe, sortino, calmar, max drawdown, cagr, win rate, hit rate, turnover, transaction cost, whole shares, trailing stop, walk-forward, adjusted close, split, dividend, forced close, trade ledger, exit reason, cumulative return, bollinger bands, sma crossover, ema crossover, round trip
---
## Worked frame
DIKETAHUI: prices [T] or [T x N] (currency), scalar parameters (windows in bars, capital, costs in bps). Check: sort dates, drop duplicate dates keep last, then fill.
DITANYAKAN: daily table [T] (position, equity, return), trades one row per fill, metrics one key each (fractions).
RUMUS: card Core 7-11 (signals) -> 12-13 (sizing, costs) -> 1-6 (metrics); toolkit signals_to_positions, backtest_positions, performance_summary.
JAWAB: 1) clean prices [T]. 2) indicators [T], NaN until window full. 3) signal [T] -> position lagged one bar [T]. 4) equity [T], returns [T] with first = 0. 5) trades [n_trades]. 6) bps -> fraction; annualise -> scalars. 7) check: prod(1+r) = V_T/V_0, BUY/SELL alternate, cash >= 0; then write.

## Core formulas
1. Mark-to-market V_t = cash + shares*close_t; r_t = V_t/V_{t-1}-1 with r_0=0; keep flat days (r=0) in the series.
2. Total return = V_T/V_0-1 = prod(1+r)-1. CAGR = (V_T/V_0)^(P/N)-1 (N periods, P per year) or ^(365.25/days).
3. Arithmetic ann. return = mean(r)*P; vol = std(r,ddof=1)*sqrt(P). Sharpe = mean(r-rf_p)/std(r-rf_p)*sqrt(P), rf_p = rf_annual/P (simple) unless compounding stated.
4. Sortino = mean(r-MAR)*P / (sqrt(mean(min(r-MAR,0)^2))*sqrt(P)); downside mean over ALL periods, MAR=0.
5. MDD = max_t (peak_t-W_t)/peak_t, peak = running max of wealth; positive fraction. Calmar = ann return/MDD (text's return definition).
6. Win rate = trades with pnl>0 / closed trades (forced close included); hit rate = periods with r>0 / periods.
7. SMA = rolling(w).mean(), NaN until full. EMA: a=2/(span+1), E_t=a P_t+(1-a)E_{t-1}, E_0=P_0 (pandas ewm adjust=False); adjust=True is a different series.
8. RSI (Wilder 1978): first avg gain/loss = SMA of first n changes, then avg=(prev*(n-1)+x)/n; RS=AG/AL; RSI=100-100/(1+RS); AL=0 -> 100. ewm(alpha=1/n) seeded at the first value differs early on.
9. Bollinger: mid=SMA20, bands mid+-2*sd. Bollinger's definition and TA-Lib use population sd (ddof=0); pandas rolling std defaults ddof=1. Text decides; pandas-style statements mean ddof=1.
10. MACD = EMA12-EMA26, signal = EMA9 of MACD (adjust=False).
11. Cross up at i: f_{i-1}<=s_{i-1} and f_i>s_i; down: f_{i-1}>=s_{i-1} and f_i<s_i. Needs both days defined.
12. Sizing: q=floor(cash/P); with costs q=floor((cash-F)/(P(1+c))); risk sizing q=floor(risk_pct*equity/stop_dist), stop_dist<=0 -> 0 shares.
13. Costs: buy = qP(1+c)+F; sell = qP(1-c)-F; trade pnl = sell net - buy total. Weight-based: cost_t = sum|w_t-w_{t-1}|*bps/1e4 on the return of that rebalance.
14. Back-adjustment: split a:b -> prices before ex-date x b/a, volumes x a/b; cash dividend D -> prices before ex-date x (P_prev-D)/P_prev, P_prev = UNADJUSTED close of last day before ex-date (CRSP). Apply to dates strictly before; process actions chronologically.

## Conventions and defaults when the task is silent
- Signal on close t, fill at open t+1 (or weight_t earns return t+1). Same-bar fill is look-ahead.
- Adjusted close for signals and valuation, adjusted open for fills when both exist.
- Intraday order: fill pending exit at open -> fill pending entry at open -> update trailing peak -> evaluate exits then entries on close -> mark at close.
- A signal is consumed on the next bar even if it sizes to 0 shares; never retry it later.
- One position at a time; long-only unless shorting stated; signal on the last bar has no next bar -> ignored; open position force-closed at last close.
- Trailing-stop peak starts at signal-day close if stated; else entry fill. Exit priority as stated.
- Daily value series marks at close without exit costs; final capital is cash after the forced close.
- Periods/year: 252 daily, 12 monthly, 52 weekly, 365 crypto. No rf given -> rf=0.
- Month-end rebalance = last trading day present in data; month-start = first day whose month differs from the prior row.
- Cleaning order: sort by date -> drop duplicate dates keep last -> ffill -> dropna.
- Walk-forward: fit on [t-W, t-1], predict t; never refit on test data; windows count rows, not calendar days.
- Simple returns aggregate across assets; log returns sum across time.
- Rounding only at output; if text says total_return from rounded final capital, round first.

## Completing missing inputs
- num_trades = number of BUY fills (round trips); pnl only on SELL rows (empty cell on BUY rows).
- Initial capital, fee, commission, window, k: from statement/params; never hard-code alternatives.
- Portfolio value for sizing while flat = cash.
- Ex-date on a non-trading day: prior trading day close via the strictly-before rule.

## Invariants to self-check
- V_T/V_0 = prod(1+r); cum[0]=1.0; last cum-1 = total return (up to forced-close costs).
- total pnl = final capital - initial (all cash at end); cash never negative; shares integer.
- BUY/SELL rows alternate, counts equal; every exec_date > signal_date except forced close.
- 0 <= MDD < 1 long-only; Sharpe equals ratio of reported components.
- Adjusted = unadjusted on the last row; close*volume invariant under split adjustment.
- Turnover >= 0; row count of daily file = trading days.

## Variants to decide
- V1 exec_lag: when a close-t signal is executed. Options: (a) fill at open t+1; (b) close t+1 (weight_t earns close-to-close r_{t+1}); (c) same close t (look-ahead). Default when silent: (a) - no-look-ahead practice (Pardo 2008; Bailey et al. 2014).
- V2 return_type: strategy return aggregation over time. Options: (a) simple returns compounded prod(1+r); (b) log returns summed. Default when silent: (a) - Bacon 2008 (simple returns aggregate across assets).
- V3 periods_per_year: annualisation factor. Options: (a) 252 trading days (12 monthly, 52 weekly); (b) 365 calendar days; (c) 260 weekdays. Default when silent: (a) - Sharpe 1994 practice; (b) only for 24/7 crypto data.
- V4 sharpe_sd: Sharpe ratio volatility estimate. Options: (a) std ddof=1 of periodic excess returns x sqrt(P), rf=0 if none given; (b) ddof=0; (c) CAGR / annual vol. Default when silent: (a) - Sharpe 1994.
- V5 sortino_dd: downside deviation. Options: (a) sqrt(mean(min(r-MAR,0)^2)) over ALL periods, MAR=0; (b) std of negative returns only. Default when silent: (a) - Sortino-Price 1994.
- V6 drawdown_base: drawdown path start. Options: (a) include initial capital as first running peak; (b) start from first marked value. Default when silent: (a) - Bacon 2008 / Magdon-Ismail 2004.
- V7 cagr_time: CAGR time base. Options: (a) (V_T/V_0)^(P/n_returns)-1 using periods per year; (b) calendar years (days/365.25). Default when silent: (a) - Bacon 2008 (consistent with the periods-per-year convention).
- V8 forced_close_trade: open position at the last bar. Options: (a) SELL at last close with costs, counted in trades, pnl and win rate; (b) mark to market only, no trade row. Default when silent: (a) - Chan 2009.
- V9 cross_warmup: crossover on the first bar where both averages exist. Options: (a) no cross (previous day NaN compares False); (b) state entry if fast > slow. Default when silent: (a) - Murphy 1999 (a cross needs two defined days).
- V10 cum_series_base: cumulative column. Options: (a) prod(1+r) starting at 1.0 with r_0 = 0; (b) prod(1+r)-1 starting at 0. Default when silent: (a) - Bacon 2008 (wealth relative).
- V11 metric_source: where reported metrics come from. Options: (a) recomputed from the daily value series per the text; (b) toolkit or library summary. Default when silent: (a) - the statement defines the metric.

## Pitfalls
- ewm adjust=True default; numpy std ddof=0 vs pandas ddof=1.
- Excluding the r_0=0 day or flat days from mean/std.
- CAGR where the text says mean*P (or vice versa); sqrt(252) on monthly data.
- Drawdown reported negative when positive is asked (and the reverse).
- Off-by-one: t-12..t-2 is iloc[t-12:t-1]; first valid month t=12.
- ffill before dedup; dedup keep first.
- Stop distance from the fill-day band instead of the signal-day band.
- Mathematically equal floats that differ in the last bit reorder ties: round (~10 dp) before sort keys and equality tests.
- Using the ex-date close (already ex) for the dividend factor.
- Crossover rules ('crosses above': f[i-1] <= s[i-1] and f[i] > s[i]) are EVENTS: a state signal f > s buys on the first bar without a cross, and an event array read as state sells the day after buying. crossover_backtest implements the events, next-open fills, forced close, mean*252 metrics and BUY/SELL labels.

- Bollinger with risk sizing and a trailing stop: bollinger_risk_stop_backtest; crossover_backtest takes window ints or SMA arrays.
## Sources
Sharpe 1966/1994; Sortino-Price 1994; Young 1991 (Calmar); Wilder 1978; Bollinger 2001; Appel (MACD); CRSP adjustment methodology; Bailey-Lopez de Prado (backtest overfitting); Pardo (walk-forward).
