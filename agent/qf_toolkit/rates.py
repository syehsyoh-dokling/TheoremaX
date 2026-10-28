"""Discounting, curve bootstrap and bond analytics with the compounding stated explicitly."""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq


def discount_factor(rate, t, compounding="continuous"):
    """DF for a rate: 'continuous', 'simple', or an int m (m times per year; 1 = annual)."""
    if compounding == "continuous":
        return math.exp(-rate * t)
    if compounding == "simple":
        return 1.0 / (1.0 + rate * t)
    m = int(compounding)
    return (1.0 + rate / m) ** (-m * t)


def zero_rate(df, t, compounding="continuous"):
    """Inverse of discount_factor."""
    if compounding == "continuous":
        return -math.log(df) / t
    if compounding == "simple":
        return (1.0 / df - 1.0) / t
    m = int(compounding)
    return m * (df ** (-1.0 / (m * t)) - 1.0)


def forward_rate(df1, t1, df2, t2, compounding="continuous"):
    """Forward rate between t1 < t2 implied by discount factors df1 = DF(t1), df2 = DF(t2).
    compounding: 'continuous' | 'simple' | m (per year). t1 = 0 means df1 = 1."""
    ratio = df1 / df2
    tau = t2 - t1
    if tau <= 0:
        raise ValueError("need t2 > t1")
    if compounding == "continuous":
        return math.log(ratio) / tau
    if compounding == "simple":
        return (ratio - 1.0) / tau
    m = int(compounding)
    return m * (ratio ** (1.0 / (m * tau)) - 1.0)


def par_curve(maturities, par_rates, freq=1):
    """Bootstrap from par rates quoted at ANY maturities: par rates are linearly interpolated
    (flat before the first quote) onto the coupon grid k/freq up to the last maturity, then
    bootstrapped. Returns a pandas DataFrame with columns
    t, par_rate, df, zero_continuous, zero_compounded (compounded `freq` times a year)."""
    import pandas as pd

    T = np.asarray(maturities, dtype=float)
    c = np.asarray(par_rates, dtype=float)
    order = np.argsort(T)
    T, c = T[order], c[order]
    n = int(round(T[-1] * freq))
    grid = np.arange(1, n + 1) / freq
    par = np.interp(grid, T, c)
    dfs = bootstrap_par(grid, par, freq)
    return pd.DataFrame({
        "t": grid, "par_rate": par, "df": dfs,
        "zero_continuous": -np.log(dfs) / grid,
        "zero_compounded": freq * (dfs ** (-1.0 / (freq * grid)) - 1.0),
    })


def bootstrap_par(maturities, par_rates, freq=1):
    """Discount factors from par coupon rates on a regular grid.

    maturities must be exactly k/freq for k = 1..n (every coupon date quoted); par bond with
    coupon c/freq per period prices at 1. Returns np.ndarray of DFs aligned with maturities.
    """
    T = np.asarray(maturities, dtype=float)
    c = np.asarray(par_rates, dtype=float)
    expected = np.arange(1, len(T) + 1) / freq
    if not np.allclose(T, expected):
        # 8 Oct: three units died on this ValueError and never recovered (yield-curve-bond,
        # zero-coupon, mtm-xccy). Irregular quotes are routed to par_curve (interpolated par rates on
        # the coupon grid) and the DFs are read back at the quoted maturities, log-linearly.
        curve = par_curve(T, c, freq)
        t_grid = curve["t"].to_numpy(dtype=float)
        log_df = np.log(curve["df"].to_numpy(dtype=float))
        return np.exp(np.interp(T, t_grid, log_df, left=log_df[0] * (T[0] / t_grid[0]) if t_grid[0] > 0 else 0.0))
    dfs = np.empty(len(T))
    for i in range(len(T)):
        coupon = c[i] / freq
        dfs[i] = (1.0 - coupon * dfs[:i].sum()) / (1.0 + coupon)
    return dfs


def bond_price(ytm, coupon_rate, maturity, freq=2, face=100.0):
    """Price of a bullet bond on a coupon date, yield compounded `freq` times a year."""
    n = int(round(maturity * freq))
    cpn = face * coupon_rate / freq
    y = ytm / freq
    k = np.arange(1, n + 1)
    return float(np.sum(cpn / (1 + y) ** k) + face / (1 + y) ** n)


def bond_ytm(price, coupon_rate, maturity, freq=2, face=100.0):
    """Yield to maturity solving bond_price(ytm) = price."""
    return brentq(lambda y: bond_price(y, coupon_rate, maturity, freq, face) - price, -0.99 * freq, 10.0,
                  xtol=1e-14)


def duration_convexity(ytm, coupon_rate, maturity, freq=2, face=100.0) -> dict:
    """Macaulay and modified duration (years) and convexity (years^2) for the yield convention used."""
    n = int(round(maturity * freq))
    cpn = face * coupon_rate / freq
    y = ytm / freq
    k = np.arange(1, n + 1)
    cf = np.full(n, cpn)
    cf[-1] += face
    pv = cf / (1 + y) ** k
    price = pv.sum()
    t = k / freq
    macaulay = float((t * pv).sum() / price)
    modified = macaulay / (1 + y)
    convexity = float((cf * k * (k + 1) / (1 + y) ** (k + 2)).sum() / (price * freq ** 2))
    return {"price": float(price), "macaulay": macaulay, "modified": modified, "convexity": convexity}


def yield_curve_from_dfs(maturities, dfs):
    """Build yield curve DataFrame from maturities and discount factors."""
    import pandas as pd
    T = np.asarray(maturities, dtype=float)
    D = np.asarray(dfs, dtype=float)
    order = np.argsort(T)
    T, D = T[order], D[order]
    zr = -np.log(D) / T
    fwd = np.empty_like(zr)
    fwd[0] = zr[0]
    for i in range(1, len(T)):
        dt = T[i] - T[i-1]
        fwd[i] = (math.log(D[i-1] / D[i])) / dt
    return pd.DataFrame({
        "maturity": T,
        "discount_factor": D,
        "zero_rate": zr,
        "forward_rate": fwd
    })

