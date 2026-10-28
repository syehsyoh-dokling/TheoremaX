"""Option pricing primitives with explicit conventions.

Units, unless a function says otherwise: rates and dividend yield continuously compounded per
year, T in years, sigma per sqrt(year). Greeks are per unit of the underlying input: vega per
1.00 of vol (divide by 100 for per vol point), theta per YEAR (divide by 365 or 252 for per
day), rho per 1.00 of rate.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1d2(S, K, T, r, q, sigma):
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    vs = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vs
    return d1, d1 - vs


def bs_price(S, K, T, r, sigma, kind="call", q=0.0):
    """Black-Scholes-Merton price with continuous dividend yield q. kind: 'call' | 'put'."""
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    dq, dr = np.exp(-q * np.asarray(T)), np.exp(-r * np.asarray(T))
    if kind == "call":
        return S * dq * norm.cdf(d1) - K * dr * norm.cdf(d2)
    if kind == "put":
        return K * dr * norm.cdf(-d2) - S * dq * norm.cdf(-d1)
    raise ValueError("kind must be 'call' or 'put'")


def bs_greeks(S, K, T, r, sigma, kind="call", q=0.0) -> dict:
    """delta, gamma, vega (per 1.00 vol), theta (per year), rho (per 1.00 rate)."""
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    T = np.asarray(T)
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    pdf = norm.pdf(d1)
    gamma = dq * pdf / (S * sigma * np.sqrt(T))
    vega = S * dq * pdf * np.sqrt(T)
    common = -S * dq * pdf * sigma / (2 * np.sqrt(T))
    if kind == "call":
        delta = dq * norm.cdf(d1)
        theta = common - r * K * dr * norm.cdf(d2) + q * S * dq * norm.cdf(d1)
        rho = K * T * dr * norm.cdf(d2)
    elif kind == "put":
        delta = dq * (norm.cdf(d1) - 1.0)
        theta = common + r * K * dr * norm.cdf(-d2) - q * S * dq * norm.cdf(-d1)
        rho = -K * T * dr * norm.cdf(-d2)
    else:
        raise ValueError("kind must be 'call' or 'put'")
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def black76_price(F, K, T, r, sigma, kind="call"):
    """Black-76 on a forward/futures price F, discounted at r."""
    return bs_price(F, K, T, r, sigma, kind, q=r)


def garman_kohlhagen_price(S, K, T, r_dom, r_for, sigma, kind="call"):
    """FX option (price in domestic currency per unit of foreign)."""
    return bs_price(S, K, T, r_dom, sigma, kind, q=r_for)


def implied_vol(price, S, K, T, r, kind="call", q=0.0, lo=1e-6, hi=5.0, xtol=1e-10):
    """Black-Scholes implied volatility by bracketing root search; nan outside no-arbitrage bounds."""
    intrinsic_lo = bs_price(S, K, T, r, lo, kind, q)
    intrinsic_hi = bs_price(S, K, T, r, hi, kind, q)
    if not (intrinsic_lo - 1e-12 <= price <= intrinsic_hi + 1e-12):
        return float("nan")
    return brentq(lambda s: float(bs_price(S, K, T, r, s, kind, q)) - price, lo, hi, xtol=xtol,
                  maxiter=500)


def crr_price(S, K, T, r, sigma, steps=500, kind="call", american=True, q=0.0):
    """Cox-Ross-Rubinstein binomial tree, European or American exercise."""
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("risk-neutral probability outside (0,1): increase steps")
    disc = math.exp(-r * dt)
    j = np.arange(steps + 1)
    prices = S * u ** j * d ** (steps - j)
    sign = 1.0 if kind == "call" else -1.0
    values = np.maximum(sign * (prices - K), 0.0)
    for step in range(steps - 1, -1, -1):
        k = np.arange(step + 1)
        prices = S * u ** k * d ** (step - k)
        values = disc * (p * values[1: step + 2] + (1 - p) * values[: step + 1])
        if american:
            values = np.maximum(values, sign * (prices - K))
    return float(values[0])


def baw_american(S, K, T, r, sigma, kind="call", q=0.0):
    """Barone-Adesi & Whaley (1987) quadratic approximation for an American option.

    Cost of carry b = r - q. An American call on a non-dividend underlying (q <= 0) equals the
    European call. Returns (price, critical_price).
    """
    b = r - q
    euro = float(bs_price(S, K, T, r, sigma, kind, q))
    if kind == "call" and q <= 0:
        return euro, float("inf")
    M = 2 * r / sigma ** 2
    N = 2 * b / sigma ** 2
    Kt = 1 - math.exp(-r * T)
    disc_carry = math.exp((b - r) * T)

    def d1(x):
        return (math.log(x / K) + (b + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    if kind == "call":
        q2 = (-(N - 1) + math.sqrt((N - 1) ** 2 + 4 * M / Kt)) / 2

        def f(x):
            return x - K - float(bs_price(x, K, T, r, sigma, "call", q)) \
                - (1 - disc_carry * norm.cdf(d1(x))) * x / q2

        hi = K * 1.5
        while f(hi) < 0:
            hi *= 1.5
        s_star = brentq(f, K, hi, xtol=1e-10)
        if S >= s_star:
            return S - K, s_star
        A2 = (s_star / q2) * (1 - disc_carry * norm.cdf(d1(s_star)))
        return euro + A2 * (S / s_star) ** q2, s_star
    if kind == "put":
        q1 = (-(N - 1) - math.sqrt((N - 1) ** 2 + 4 * M / Kt)) / 2

        def g(x):
            return K - x - float(bs_price(x, K, T, r, sigma, "put", q)) \
                + (1 - disc_carry * norm.cdf(-d1(x))) * x / q1

        lo = K * 0.5
        while g(lo) < 0 and lo > 1e-8:
            lo *= 0.5
        s_star = brentq(g, lo, K * (1 - 1e-12), xtol=1e-10)
        if S <= s_star:
            return K - S, s_star
        A1 = -(s_star / q1) * (1 - disc_carry * norm.cdf(-d1(s_star)))
        return euro + A1 * (S / s_star) ** q1, s_star
    raise ValueError("kind must be 'call' or 'put'")
