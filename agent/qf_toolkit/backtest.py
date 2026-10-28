"""Signal -> position -> equity backtesting with every convention as a keyword.

Why it exists: re-derived backtests compounded wrongly or mixed shares with returns (a Bollinger
run reported total_return 1005 and final capital 100.6M from 100k). The timing rule used
everywhere in this module:

    signal[t]    is computed from data up to and including the close of bar t
    position[t]  is what is HELD during bar t (earns the return from close t-1 to close t)
    signals_to_positions(signal, lag=1) sets position[t] = signal[t-1]  -> no look-ahead;
    a signal on the LAST bar has no next bar and never trades.

backtest_positions() takes positions that are ALREADY lagged; it never shifts them again.

    from qf_toolkit import backtest
    sig = backtest.bollinger_signals(close, window=20, k=2.0)
    pos = backtest.signals_to_positions(sig)                      # lag=1
    bt = backtest.backtest_positions(close, pos, initial_capital=100000.0, cost_bps=10)
    summ = backtest.performance_summary(bt["strategy_return"], equity=bt["equity"],
                                        trades=backtest.trade_list(pos, close))
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .stats import annualized_mean, annualized_vol, cagr, max_drawdown, sharpe

__all__ = ["signals_to_positions", "bollinger_bands", "bollinger_signals", "sma_crossover_signals",
           "momentum_signal", "backtest_positions", "trade_list", "performance_summary"]


# ----------------------------------------------------------------------------------------------
# signals and positions
# ----------------------------------------------------------------------------------------------
def signals_to_positions(signal, *, lag=1, long_only=False, hold="until_exit", fill_value=0.0):
    """Turn a signal (known at the close of bar t) into the position HELD during bar t+lag.

    signal : Series/DataFrame/array indexed by bar.
    hold="until_exit" (default): the signal is an EVENT series - NaN = no event (keep the
        previous position), 0 = exit to flat, +x = go long x, -x = go short x. Events are
        forward-filled into a state (the entry/exit state machine), start flat.
    hold="signal": the signal already IS the desired position each bar (NaN -> flat); only
        the lag is applied. Use this for the output of bollinger_signals / sma_crossover_signals
        (they already return a state).
    long_only=False: if True, negative targets become 0 (flat), i.e. a short entry only exits a long.
    lag=1 (default): position[t] = state[t-1]. lag=0 is same-bar execution = LOOK-AHEAD; only use
        it when the task explicitly says the trade happens on the signal bar's own return.
    fill_value=0.0: position for the first `lag` bars (nothing known yet).
    Returns the same type as the input (ndarray in -> ndarray out).
    """
    is_np = not isinstance(signal, (pd.Series, pd.DataFrame))
    s = pd.Series(np.asarray(signal, dtype=float)) if is_np else signal.astype(float)
    if hold == "until_exit":
        state = s.ffill().fillna(0.0)
    elif hold == "signal":
        state = s.fillna(0.0)
    else:
        raise ValueError("hold must be 'until_exit' or 'signal'")
    if long_only:
        state = state.clip(lower=0.0)
    pos = state.shift(int(lag)).fillna(fill_value) if lag else state
    return pos.to_numpy() if is_np else pos


def bollinger_bands(price, window=20, k=2.0, *, ddof=0, min_periods=None) -> pd.DataFrame:
    """Bollinger bands -> DataFrame[mid, upper, lower, std] on the price index.

    mid = rolling SMA(window); upper/lower = mid +- k * rolling std.
    ddof=0 (default) = POPULATION std, Bollinger's own definition and TA-Lib's. NOTE pandas
    `.rolling().std()` defaults to ddof=1 (sample std): pass ddof=1 when the task's reference
    uses pandas defaults or says "sample std". First window-1 rows are NaN (min_periods=window).
    """
    p = pd.Series(price, dtype=float) if not isinstance(price, pd.Series) else price.astype(float)
    mp = window if min_periods is None else min_periods
    roll = p.rolling(window, min_periods=mp)
    mid = roll.mean()
    sd = roll.std(ddof=ddof)
    return pd.DataFrame({"mid": mid, "upper": mid + k * sd, "lower": mid - k * sd, "std": sd})


def bollinger_signals(price, window=20, k=2.0, *, mode="mean_reversion", exit="mid", ddof=0,
                      long_only=False, inclusive=False) -> pd.Series:
    """Band strategy STATE (desired position after the close of bar t): +1 long, -1 short, 0 flat.

    Feed the result to signals_to_positions(sig, hold="signal") (or the default - a state with no
    NaN is identical under both) to get lagged positions.

    mode="mean_reversion" (default): enter long when close < lower, short when close > upper.
    mode="breakout": enter long when close > upper, short when close < lower.
    exit="mid" (default): a long exits when close crosses the middle band (>= mid for mean
        reversion, <= mid for breakout); shorts symmetric. exit="opposite": hold until the opposite
        entry condition, then flip (or go flat when long_only).
    inclusive=False: strict band comparisons (< lower, > upper); True uses <= / >=.
    long_only=False: if True short entries are ignored (they only close a long under exit="opposite").
    ddof: see bollinger_bands (default 0 = population std).
    Bars before the bands exist are flat. Exit and entry on the same bar: exit is checked first,
    then a new entry may fire on that same close.
    """
    p = pd.Series(price, dtype=float) if not isinstance(price, pd.Series) else price.astype(float)
    bb = bollinger_bands(p, window, k, ddof=ddof)
    if mode not in ("mean_reversion", "breakout"):
        raise ValueError("mode must be 'mean_reversion' or 'breakout'")
    if exit not in ("mid", "opposite"):
        raise ValueError("exit must be 'mid' or 'opposite'")
    px, lo, up, mid = p.to_numpy(), bb["lower"].to_numpy(), bb["upper"].to_numpy(), bb["mid"].to_numpy()
    below = (lambda a, b: a <= b) if inclusive else (lambda a, b: a < b)
    above = (lambda a, b: a >= b) if inclusive else (lambda a, b: a > b)
    out = np.zeros(len(px))
    state = 0.0
    for i in range(len(px)):
        if not (np.isfinite(px[i]) and np.isfinite(lo[i]) and np.isfinite(up[i])):
            out[i] = state
            continue
        if mode == "mean_reversion":
            long_entry, short_entry = below(px[i], lo[i]), above(px[i], up[i])
            long_exit_mid, short_exit_mid = px[i] >= mid[i], px[i] <= mid[i]
        else:
            long_entry, short_entry = above(px[i], up[i]), below(px[i], lo[i])
            long_exit_mid, short_exit_mid = px[i] <= mid[i], px[i] >= mid[i]
        if exit == "mid":
            if state > 0 and long_exit_mid:
                state = 0.0
            elif state < 0 and short_exit_mid:
                state = 0.0
        if long_entry:
            state = 1.0
        elif short_entry:
            state = 0.0 if long_only else -1.0
        out[i] = state
    return pd.Series(out, index=p.index, name="signal")


def sma_crossover_signals(price, fast, slow, *, ma="sma", long_only=True, output="state") -> pd.Series:
    """Moving-average crossover.

    ma="sma" (rolling mean, NaN until full) | "ema" (ewm(span, adjust=False)).
    output="state" (default): +1 while fast > slow, -1 (or 0 if long_only=True, the default) while
        fast < slow; ties keep the previous state; 0 until both averages exist. Feed to
        signals_to_positions(..., hold="signal").
    output="cross": event series, +1 on a golden cross bar (f[i-1] <= s[i-1] and f[i] > s[i]),
        -1 on a death cross (f[i-1] >= s[i-1] and f[i] < s[i]), 0 otherwise; needs both bars defined.
    """
    p = pd.Series(price, dtype=float) if not isinstance(price, pd.Series) else price.astype(float)
    if ma == "sma":
        f, s = p.rolling(fast).mean(), p.rolling(slow).mean()
    elif ma == "ema":
        f, s = p.ewm(span=fast, adjust=False).mean(), p.ewm(span=slow, adjust=False).mean()
    else:
        raise ValueError("ma must be 'sma' or 'ema'")
    if output == "cross":
        fp, sp = f.shift(1), s.shift(1)
        up = (fp <= sp) & (f > s)
        dn = (fp >= sp) & (f < s)
        return pd.Series(np.where(up, 1.0, np.where(dn, -1.0, 0.0)), index=p.index, name="cross")
    if output != "state":
        raise ValueError("output must be 'state' or 'cross'")
    neg = 0.0 if long_only else -1.0
    raw = pd.Series(np.where(f > s, 1.0, np.where(f < s, neg, np.nan)), index=p.index)
    both = f.notna() & s.notna()
    raw[~both] = 0.0
    return raw.ffill().fillna(0.0).rename("signal")


def momentum_signal(returns, lookback, skip=0, *, method="compound", min_periods=None):
    """Time-series / cross-sectional momentum known at the close of row t.

    Uses the `lookback` returns ending at row t, EXCLUDING the most recent `skip` of them:
    returns r[t-lookback+1 .. t-skip]  (lookback - skip returns).
    Classic 12-1 on monthly data: lookback=12, skip=1 -> r[t-11..t-1]; for the month being
    predicted (h = t+1) that is h-12..h-2. Hold it with lag=1 (signal t, return t+1).
    method="compound" (prod(1+r)-1, default) | "sum" (sum r) | "log" (sum log1p r).
    Works on Series or DataFrame (per column). NaN until the full window exists
    (min_periods = lookback - skip by default).
    """
    r = returns if isinstance(returns, (pd.Series, pd.DataFrame)) else pd.Series(returns, dtype=float)
    r = r.astype(float)
    n = int(lookback) - int(skip)
    if n <= 0:
        raise ValueError("lookback must exceed skip")
    mp = n if min_periods is None else min_periods
    base = r.shift(int(skip)) if skip else r
    if method == "compound":
        out = np.expm1(np.log1p(base).rolling(n, min_periods=mp).sum())
    elif method == "sum":
        out = base.rolling(n, min_periods=mp).sum()
    elif method == "log":
        out = np.log1p(base).rolling(n, min_periods=mp).sum()
    else:
        raise ValueError("method must be 'compound', 'sum' or 'log'")
    return out


# ----------------------------------------------------------------------------------------------
# backtest engine
# ----------------------------------------------------------------------------------------------
def _as_series(x, index=None):
    if isinstance(x, pd.Series):
        return x.astype(float)
    return pd.Series(np.asarray(x, dtype=float), index=index)


def backtest_positions(price_or_returns, positions, *, initial_capital=100000.0, cost_bps=0.0,
                       input="price", returns_are="simple", position_is="weight", fractional=True,
                       fill_price=None, fee_per_trade=0.0, liquidate_at_end=False) -> pd.DataFrame:
    """Backtest ALREADY-LAGGED positions (position[t] is held during bar t; see module doc).

    price_or_returns : Series of prices (input="price", default) or of per-bar returns
        (input="returns"; returns_are="simple" default | "log", converted with expm1).
        DataFrame (multi-asset) is supported for position_is="weight", fractional=True only.
    positions : same index. position_is="weight" (default): fraction of equity (1 = fully long,
        -1 = fully short, 0.5 = half). position_is="shares": number of shares held.
    cost_bps=0.0 : proportional cost in basis points of TRADED value (bps/1e4 x |change|).
    fee_per_trade=0.0 : fixed currency fee per bar with a non-zero trade (share accounting only).

    Accounting (asset_return[t] = price[t]/price[t-1]-1, asset_return[0] = 0; with input="returns"
    asset_return[t] = the given return of bar t, row 0 included, NaN -> 0):
      weight, fractional=True (vectorised, weight held constant through the bar):
          cost_return[t] = |w[t]-w[t-1]| * cost_bps/1e4   (w[-1] = 0: the first entry pays)
          strategy_return[t] = w[t]*asset_return[t] - cost_return[t]
      shares, or weight with fractional=False or with fill_price (weights -> shares, re-sized
      ONLY when the target weight changes: q = w * equity_at_fill / (fill * (1+c)), truncated
      toward zero to whole shares when fractional=False), cash accounting:
          trade dq[t] = q[t]-q[t-1] executed at fill[t]; cash -= dq*fill + |dq|*fill*c + fee
          equity[t] = cash[t] + q[t]*price[t];  strategy_return[t] = equity[t]/equity[t-1]-1
      fill[t] = fill_price[t] if given (e.g. the OPEN of bar t: signal close t-1 -> fill open t),
          else price[t-1] (the close at which the lagged signal became known; price[0] for t=0).
      In both: equity[t] = equity[t-1] * (1 + strategy_return[t]), equity[-1] = initial_capital.
      So buy-and-hold (weight 1 from bar 0) ends at initial_capital * price[-1]/price[0].
    liquidate_at_end=False : if True the last row also pays the cost of closing the position
        (final equity = cash after the forced close).

    Returns DataFrame[position, asset_return, strategy_return, cost, cost_return, equity, turnover]
      cost = currency cost paid that bar; cost_return = cost / equity[t-1];
      turnover = |traded value| / equity[t-1] (= |w[t]-w[t-1]| in the vectorised weight mode).
      Share accounting adds columns shares, cash. Multi-asset: position = net weight sum,
      asset_return = gross portfolio return sum(w*r).
    """
    c = float(cost_bps) / 1e4
    if returns_are not in ("simple", "log"):
        raise ValueError("returns_are must be 'simple' or 'log'")
    if input not in ("price", "returns"):
        raise ValueError("input must be 'price' or 'returns'")
    if position_is not in ("weight", "shares"):
        raise ValueError("position_is must be 'weight' or 'shares'")

    # ---- multi-asset, vectorised weights
    if isinstance(price_or_returns, pd.DataFrame):
        if position_is != "weight" or not fractional:
            raise ValueError("DataFrame input supports position_is='weight', fractional=True only")
        X = price_or_returns.astype(float)
        R = X.pct_change() if input == "price" else (np.expm1(X) if returns_are == "log" else X)
        R = R.fillna(0.0)
        W = pd.DataFrame(positions, index=X.index, columns=X.columns).astype(float).fillna(0.0)
        dW = W.diff()
        dW.iloc[0] = W.iloc[0]
        turn = dW.abs().sum(axis=1)
        if liquidate_at_end:
            turn.iloc[-1] += W.iloc[-1].abs().sum()
        gross = (W * R).sum(axis=1)
        cost_ret = turn * c
        sr = gross - cost_ret
        eq = initial_capital * (1 + sr).cumprod()
        prev_eq = eq.shift(1).fillna(initial_capital)
        return pd.DataFrame({"position": W.sum(axis=1), "asset_return": gross, "strategy_return": sr,
                             "cost": cost_ret * prev_eq, "cost_return": cost_ret, "equity": eq,
                             "turnover": turn})

    x = _as_series(price_or_returns)
    pos = _as_series(positions, index=x.index).reindex(x.index).fillna(0.0)
    if input == "price":
        price = x
        r = price.pct_change().fillna(0.0)
    else:
        r = (np.expm1(x) if returns_are == "log" else x).fillna(0.0)
        price = None

    use_shares = position_is == "shares" or not fractional or fill_price is not None
    if not use_shares:
        w = pos
        dw = w.diff()
        dw.iloc[0] = w.iloc[0]
        turn = dw.abs()
        if liquidate_at_end:
            turn.iloc[-1] += abs(w.iloc[-1])
        cost_ret = turn * c
        sr = w * r - cost_ret
        eq = initial_capital * (1 + sr).cumprod()
        prev_eq = eq.shift(1).fillna(initial_capital)
        return pd.DataFrame({"position": w, "asset_return": r, "strategy_return": sr,
                             "cost": cost_ret * prev_eq, "cost_return": cost_ret, "equity": eq,
                             "turnover": turn})

    # ---- share accounting loop
    if price is None:  # returns input: synthesise a price index starting at 1
        price = (1 + r).cumprod()
        if position_is == "shares":
            raise ValueError("position_is='shares' needs input='price'")
    P = price.to_numpy(dtype=float)
    F = (_as_series(fill_price, index=x.index).reindex(x.index).to_numpy(dtype=float)
         if fill_price is not None else np.concatenate([[P[0]], P[:-1]]))
    tgt = pos.to_numpy(dtype=float)
    n = len(P)
    cash, q_prev, eq_prev, w_prev = float(initial_capital), 0.0, float(initial_capital), 0.0
    out = {k: np.zeros(n) for k in ("shares", "cash", "equity", "cost", "turnover", "prev")}
    for t in range(n):
        f = F[t]
        if position_is == "shares":
            q = tgt[t] if fractional else float(math.trunc(tgt[t]))
        else:
            if tgt[t] != w_prev or t == 0:
                eq_at_fill = cash + q_prev * f
                qq = tgt[t] * eq_at_fill / (f * (1 + c)) if f > 0 else 0.0
                q = qq if fractional else float(math.trunc(qq))
            else:
                q = q_prev
            w_prev = tgt[t]
        dq = q - q_prev
        cost = abs(dq) * f * c + (fee_per_trade if dq != 0 else 0.0)
        cash -= dq * f + cost
        traded = abs(dq) * f
        if liquidate_at_end and t == n - 1 and q != 0:
            lc = abs(q) * P[t] * c + fee_per_trade
            cash += q * P[t] - lc
            cost += lc
            traded += abs(q) * P[t]
            q_hold_mark = 0.0
        else:
            q_hold_mark = q
        eq = cash + q_hold_mark * P[t]
        out["shares"][t], out["cash"][t], out["equity"][t] = q, cash, eq
        out["cost"][t], out["turnover"][t] = cost, traded / eq_prev if eq_prev else np.nan
        out["prev"][t] = eq_prev
        q_prev, eq_prev = q, eq
    eq = pd.Series(out["equity"], index=x.index)
    prev = pd.Series(out["prev"], index=x.index)
    sr = eq / prev - 1
    return pd.DataFrame({"position": pos if position_is == "weight" else pd.Series(out["shares"], index=x.index),
                         "asset_return": r, "strategy_return": sr,
                         "cost": out["cost"], "cost_return": out["cost"] / prev.to_numpy(),
                         "equity": eq, "turnover": out["turnover"],
                         "shares": out["shares"], "cash": out["cash"]}, index=x.index)


# ----------------------------------------------------------------------------------------------
# trades and summary
# ----------------------------------------------------------------------------------------------
def trade_list(positions, price, *, fill_price=None, cost_bps=0.0, force_close=True) -> pd.DataFrame:
    """Round trips from LAGGED positions (position[t] held during bar t) -> DataFrame, one row per trade:
    [direction, size, entry_date, entry_price, exit_date, exit_price, return, return_net,
     bars_held, holding_days, forced_close].

    A trade opens when the position goes from 0 (or the opposite sign) to non-zero and closes when
    it returns to 0 or flips sign (a flip closes one trade and opens another). Size changes that
    keep the sign stay in the same trade (size = entry size).
    Execution (same rule as backtest_positions): a change entering bar t fills at fill_price[t] on
    date index[t] if given, else at price[t-1] on date index[t-1] (the signal bar's close).
    An open position at the end is closed at the last price on the last date (forced_close=True)
    when force_close=True (default), else dropped.
    return = direction * (exit/entry - 1) (gross); return_net with proportional cost c=cost_bps/1e4:
      long exit(1-c)/(entry(1+c)) - 1; short (entry(1-c) - exit(1+c)) / (entry(1+c)).
    holding_days = calendar days between entry and exit dates (DatetimeIndex) else bar count.
    """
    p = _as_series(price)
    pos = _as_series(positions, index=p.index).reindex(p.index).fillna(0.0).to_numpy()
    P = p.to_numpy()
    idx = p.index
    n = len(P)
    c = float(cost_bps) / 1e4
    if fill_price is not None:
        Fp = _as_series(fill_price, index=idx).reindex(idx).to_numpy()

    def exec_at(t):
        if fill_price is not None:
            return idx[t], Fp[t], t
        j = max(t - 1, 0)
        return idx[j], P[j], j

    rows = []
    open_tr = None
    prev = 0.0
    for t in range(n):
        cur = pos[t]
        if np.sign(cur) != np.sign(prev):
            if open_tr is not None:
                d, px, j = exec_at(t)
                open_tr.update(exit_date=d, exit_price=px, _exit_i=j, forced_close=False)
                rows.append(open_tr)
                open_tr = None
            if cur != 0:
                d, px, j = exec_at(t)
                open_tr = {"direction": int(np.sign(cur)), "size": float(abs(cur)), "entry_date": d,
                           "entry_price": px, "_entry_i": j}
        prev = cur
    if open_tr is not None and force_close:
        open_tr.update(exit_date=idx[-1], exit_price=P[-1], _exit_i=n - 1, forced_close=True)
        rows.append(open_tr)
    cols = ["direction", "size", "entry_date", "entry_price", "exit_date", "exit_price", "return",
            "return_net", "bars_held", "holding_days", "forced_close"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    e, x, dr = df["entry_price"].astype(float), df["exit_price"].astype(float), df["direction"]
    df["return"] = dr * (x / e - 1)
    df["return_net"] = np.where(dr > 0, x * (1 - c) / (e * (1 + c)) - 1,
                                (e * (1 - c) - x * (1 + c)) / (e * (1 + c)))
    df["bars_held"] = df["_exit_i"] - df["_entry_i"]
    if isinstance(idx, pd.DatetimeIndex):
        df["holding_days"] = (pd.to_datetime(df["exit_date"]) - pd.to_datetime(df["entry_date"])).dt.days
    else:
        df["holding_days"] = df["bars_held"]
    return df[cols].reset_index(drop=True)


def performance_summary(strategy_returns, equity=None, periods_per_year=252, rf=0.0, *, trades=None,
                        initial_capital=None, annualized_return="cagr", mar=None, ddof=1,
                        trade_return_col="return") -> dict:
    """Standard backtest statistics from simple periodic strategy returns -> dict of floats.

    Keep flat bars (r=0) and the first bar in the series: they count as periods.
    rf = ANNUAL risk-free rate (default 0); per period rf/periods_per_year (simple).
    total_return = prod(1+r)-1 ;  cagr = (1+total)^(ppy/N)-1 (stats.cagr) ;
    annualized_mean = mean(r)*ppy (stats.annualized_mean) ;
    annualized_return = cagr (default, annualized_return="cagr") or "arithmetic" (= annualized_mean);
    annualized_volatility = std(r, ddof=1)*sqrt(ppy) ; sharpe = stats.sharpe(r, ppy, rf/ppy, ddof) ;
    sortino = mean(r-MAR)*ppy / (sqrt(mean(min(r-MAR,0)^2))*sqrt(ppy)), downside over ALL periods,
        MAR per period = mar if given else rf/ppy ;
    max_drawdown = POSITIVE fraction (0.23 = 23% peak-to-trough; stats.max_drawdown is negative),
        from the equity curve with the starting value (initial capital / 1.0) included as a peak ;
    calmar = annualized_return / max_drawdown (nan if no drawdown) ;
    hit_rate = share of periods with r > 0 ;
    win_rate = share of closed trades with trades[trade_return_col] > 0 when `trades` (trade_list
        output) is given (win_rate_basis="trades", n_trades set); otherwise = hit_rate
        (win_rate_basis="periods") ;
    final_capital = equity[-1] if equity given, else initial_capital*(1+total_return) if given,
        else nan.
    """
    r = pd.Series(strategy_returns, dtype=float).dropna()
    ppy = periods_per_year
    rf_p = float(rf) / ppy
    total = float(np.prod(1 + r.to_numpy()) - 1)
    g = cagr(r, ppy)
    am = annualized_mean(r, ppy)
    if annualized_return == "cagr":
        ann = g
    elif annualized_return == "arithmetic":
        ann = am
    else:
        raise ValueError("annualized_return must be 'cagr' or 'arithmetic'")
    m = rf_p if mar is None else float(mar)
    ex = r.to_numpy() - m
    dd = math.sqrt(float(np.mean(np.minimum(ex, 0.0) ** 2)))
    sortino = float(np.mean(ex) * ppy / (dd * math.sqrt(ppy))) if dd > 0 else float("nan")
    curve = np.concatenate([[1.0], np.cumprod(1 + r.to_numpy())])
    mdd = -max_drawdown(equity=curve)
    out = {
        "total_return": total,
        "cagr": g,
        "annualized_mean": am,
        "annualized_return": ann,
        "annualized_volatility": annualized_vol(r, ppy, ddof=ddof),
        "sharpe": sharpe(r, ppy, rf_per_period=rf_p, ddof=ddof),
        "sortino": sortino,
        "max_drawdown": float(mdd),
        "calmar": float(ann / mdd) if mdd > 0 else float("nan"),
        "hit_rate": float((r > 0).mean()) if len(r) else float("nan"),
        "n_periods": int(len(r)),
    }
    if trades is not None:
        tr = pd.DataFrame(trades)
        out["n_trades"] = int(len(tr))
        out["win_rate"] = float((tr[trade_return_col] > 0).mean()) if len(tr) else float("nan")
        out["win_rate_basis"] = "trades"
    else:
        out["win_rate"] = out["hit_rate"]
        out["win_rate_basis"] = "periods"
    if equity is not None:
        out["final_capital"] = float(pd.Series(equity, dtype=float).dropna().iloc[-1])
    elif initial_capital is not None:
        out["final_capital"] = float(initial_capital) * (1 + total)
    else:
        out["final_capital"] = float("nan")
    return out
