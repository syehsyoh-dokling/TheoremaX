"""Exotic-option and first-passage primitives with explicit conventions.

Units, unless a function says otherwise: rates r and dividend yield q continuously compounded per
year, T / t in years, sigma per sqrt(year), prices in the currency of S. Cost of carry b = r - q.
All barriers are CONTINUOUSLY monitored; for discrete monitoring every dt years shift the barrier
with `bgk_shift` first. Functions accept scalars or numpy arrays (broadcast) and return a Python
float for scalar input, an ndarray otherwise.

First-passage conventions: X_t = ln(S_t/S0) = nu*t + sigma*W_t. Pass nu directly as `mu`
(drift_is_log=True, default) or pass the drift of S itself with drift_is_log=False, in which case
nu = mu - sigma**2/2. Direction is automatic: H > S0 means the running MAX must reach H (up),
H < S0 means the running MIN must reach H (down), H == S0 counts as already touched.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from .options import bs_price

BGK_BETA = 0.5826  # -zeta(1/2)/sqrt(2*pi), Broadie-Glasserman-Kou (1997)
_B_SMALL = 1e-8    # |b| below this uses the analytic b -> 0 limit in the lookback formulas


def _out(x):
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x


def _nu(mu, sigma, drift_is_log):
    mu, sigma = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
    return mu if drift_is_log else mu - 0.5 * sigma ** 2


# ---------------------------------------------------------------------------------------------
# First passage of (log-)Brownian motion with drift
# ---------------------------------------------------------------------------------------------

def _hit_prob_log(b, t, nu, sigma):
    """P(X touches level b by time t), X_0 = 0, X = nu*t + sigma*W; either direction."""
    b, t, nu, sigma = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (b, t, nu, sigma)))
    a = np.abs(b)
    nup = nu * np.where(b < 0, -1.0, 1.0)          # drift component pointing TOWARD the barrier
    ts = np.where(t > 0, t, 1.0)
    vt = sigma * np.sqrt(ts)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        t1 = norm.cdf((-a + nup * ts) / vt)
        t2 = np.exp(2.0 * nup * a / sigma ** 2 + norm.logcdf((-a - nup * ts) / vt))
        p = np.nan_to_num(t1 + t2, nan=0.0, posinf=1.0)
    p = np.where(t > 0, p, 0.0)
    p = np.where(a == 0, 1.0, p)
    return np.clip(p, 0.0, 1.0)


def barrier_hit_prob(S0, H, T, mu, sigma, *, drift_is_log=True):
    """P(GBM path started at S0 touches H at some t in [0,T]) -> float|array in [0,1].

    H > S0: up (running max), H < S0: down (running min), H == S0: 1.0. Continuous monitoring.
    `mu` is the log-drift nu (drift_is_log=True) or the drift of S (drift_is_log=False).
    For a risk-neutral probability use mu = r - q with drift_is_log=False.
    """
    nu = _nu(mu, sigma, drift_is_log)
    return _out(_hit_prob_log(np.log(np.asarray(H, float) / np.asarray(S0, float)), T, nu, sigma))


def first_passage_cdf(S0, H, t, mu, sigma, *, drift_is_log=True):
    """P(tau_H <= t) for the first hitting time tau_H of H -> float|array in [0,1] (defective:
    tends to exp(-2*|nu|*|ln(H/S0)|/sigma**2) < 1 as t -> inf when the drift points away)."""
    return barrier_hit_prob(S0, H, t, mu, sigma, drift_is_log=drift_is_log)


def first_passage_pdf(S0, H, t, mu, sigma, *, drift_is_log=True):
    """Density of tau_H at t (per year), inverse-Gaussian form -> float|array >= 0; 0 for t <= 0.

    f(t) = a / (sigma*sqrt(2*pi*t**3)) * exp(-(a - nu'*t)**2 / (2*sigma**2*t)), a = |ln(H/S0)|,
    nu' = log-drift toward the barrier. Integrates to first_passage_cdf (defective if nu' < 0).
    """
    nu = _nu(mu, sigma, drift_is_log)
    b = np.log(np.asarray(H, float) / np.asarray(S0, float))
    b, t, nu, sigma = np.broadcast_arrays(b, np.asarray(t, float), nu, np.asarray(sigma, float))
    a = np.abs(b)
    nup = nu * np.where(b < 0, -1.0, 1.0)
    ts = np.where(t > 0, t, 1.0)
    f = a / (sigma * np.sqrt(2 * np.pi * ts ** 3)) * np.exp(-(a - nup * ts) ** 2 / (2 * sigma ** 2 * ts))
    return _out(np.where(t > 0, f, 0.0))


def expected_first_passage_time(S0, H, mu, sigma, drift_is_log=True):
    """E[tau_H] in years -> float|array; |ln(H/S0)|/nu' if the log-drift nu' toward H is > 0,
    inf if nu' <= 0 (driftless BM hits a.s. but with infinite mean), 0.0 if H == S0."""
    nu = _nu(mu, sigma, drift_is_log)
    b = np.log(np.asarray(H, float) / np.asarray(S0, float))
    b, nu = np.broadcast_arrays(b, nu)
    a = np.abs(b)
    nup = nu * np.where(b < 0, -1.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(nup > 0, a / np.where(nup > 0, nup, 1.0), np.inf)
    return _out(np.where(a == 0, 0.0, e))



#: 30 Sep first-passage: running_min_cdf(L, ...) was called with the price level L (~610) instead of
#: ln(L/S0); the result was silently 0/1. A log distance beyond +-3 (a 20x move) is a price, not a log
#: level - fail loudly with the fix instead of answering the wrong question.
LOG_LEVEL_LIMIT = 3.0


def _check_log_level(value, name):
    arr = np.asarray(value, float)
    arr = arr[np.isfinite(arr)] if arr.ndim else (arr[None] if np.isfinite(arr) else arr[None][:0])
    if arr.size and np.max(arr) > LOG_LEVEL_LIMIT:      # prices are positive; very negative logs are limits
        raise ValueError(f"{name} must be a LOG distance ln(level/S0) (got {float(np.max(arr)):.4g}, which "
                         "looks like a price level): pass np.log(level/S0), or use "
                         "barrier_hit_prob(S0, H, T, mu, sigma) which takes price levels directly")

def running_max_cdf(m, T, nu, sigma):
    """P(M_T <= m) for M_T = max_{t<=T} X_t, X_0 = 0, X = nu*t + sigma*W (LOG space: m = ln(level/S0))
    -> float|array in [0,1]; 0 for m < 0."""
    _check_log_level(m, "m")
    m, T, nu, sigma = np.broadcast_arrays(*(np.asarray(v, float) for v in (m, T, nu, sigma)))
    ts = np.where(T > 0, T, 1.0)
    vt = sigma * np.sqrt(ts)
    mm = np.maximum(m, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        p = norm.cdf((mm - nu * ts) / vt) - np.exp(2 * nu * mm / sigma ** 2
                                                    + norm.logcdf((-mm - nu * ts) / vt))
    p = np.nan_to_num(p, nan=0.0)
    p = np.where(T > 0, p, 1.0)
    return _out(np.clip(np.where(m < 0, 0.0, p), 0.0, 1.0))


def running_min_cdf(m, T, nu, sigma):
    """P(m_T >= m) for m_T = min_{t<=T} X_t, X_0 = 0 (LOG space: m = ln(level/S0))
    -> float|array in [0,1]; 0 for m > 0. (Survival of the minimum, i.e. P(no touch of m).)"""
    _check_log_level(m, "m")
    return running_max_cdf(-np.asarray(m, float), T, -np.asarray(nu, float), sigma)


def joint_end_max_cdf(x, b, T, nu, sigma):
    """P(X_T <= x, M_T <= b), X_0 = 0, X = nu*t + sigma*W, M = running max, LOG space
    -> float|array in [0,1]; 0 for b < 0; equals running_max_cdf(b) for x >= b."""
    _check_log_level(x, "x")
    _check_log_level(b, "b")
    x, b, T, nu, sigma = np.broadcast_arrays(*(np.asarray(v, float) for v in (x, b, T, nu, sigma)))
    vt = sigma * np.sqrt(T)
    xx = np.minimum(x, b)
    bb = np.maximum(b, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        p = norm.cdf((xx - nu * T) / vt) - np.exp(2 * nu * bb / sigma ** 2
                                                   + norm.logcdf((xx - 2 * bb - nu * T) / vt))
    p = np.nan_to_num(p, nan=0.0)
    return _out(np.clip(np.where(b < 0, 0.0, p), 0.0, 1.0))


def bgk_shift(H, S0, sigma, dt):
    """Broadie-Glasserman-Kou barrier for DISCRETE monitoring every dt years -> float|array.

    Returns H*exp(+0.5826*sigma*sqrt(dt)) if H > S0, H*exp(-0.5826*sigma*sqrt(dt)) if H < S0
    (always shifted AWAY from S0). Plug the result into the continuous formulas.
    """
    H, S0 = np.asarray(H, float), np.asarray(S0, float)
    sgn = np.sign(H - S0)
    return _out(H * np.exp(sgn * BGK_BETA * np.asarray(sigma, float) * np.sqrt(dt)))


# ---------------------------------------------------------------------------------------------
# Barrier options (Merton / Reiner-Rubinstein, as tabulated in Haug ch. 4.17)
# ---------------------------------------------------------------------------------------------

_BARRIERS = ("down-and-out", "down-and-in", "up-and-out", "up-and-in")


def barrier_price(S, K, H, T, r, sigma, kind="call", barrier="down-and-out", q=0.0, rebate=0.0):
    """Single-barrier European option, continuous monitoring -> float|array.

    barrier in {'down-and-out','down-and-in','up-and-out','up-and-in'}; kind 'call'|'put'.
    rebate: cash paid AT THE HIT TIME for knock-outs, AT EXPIRY if never knocked in for knock-ins.
    Already breached (S <= H down, S >= H up): out -> rebate, in -> vanilla bs_price.
    in + out = vanilla when rebate == 0.
    """
    if barrier not in _BARRIERS:
        raise ValueError(f"barrier must be one of {_BARRIERS}")
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    S, K, H, T, sigma, r, q, rebate = np.broadcast_arrays(
        *(np.asarray(v, float) for v in (S, K, H, T, sigma, r, q, rebate)))
    down = barrier.startswith("down")
    out = barrier.endswith("out")
    phi = 1.0 if kind == "call" else -1.0
    eta = 1.0 if down else -1.0
    b = r - q
    vt = sigma * np.sqrt(T)
    mu = (b - 0.5 * sigma ** 2) / sigma ** 2
    lam = np.sqrt(mu ** 2 + 2 * r / sigma ** 2)
    x1 = np.log(S / K) / vt + (1 + mu) * vt
    x2 = np.log(S / H) / vt + (1 + mu) * vt
    y1 = np.log(H ** 2 / (S * K)) / vt + (1 + mu) * vt
    y2 = np.log(H / S) / vt + (1 + mu) * vt
    z = np.log(H / S) / vt + lam * vt
    hs = H / S
    Sq = S * np.exp((b - r) * T)
    Kr = K * np.exp(-r * T)
    N = norm.cdf
    A = phi * Sq * N(phi * x1) - phi * Kr * N(phi * x1 - phi * vt)
    B = phi * Sq * N(phi * x2) - phi * Kr * N(phi * x2 - phi * vt)
    C = phi * Sq * hs ** (2 * (mu + 1)) * N(eta * y1) - phi * Kr * hs ** (2 * mu) * N(eta * y1 - eta * vt)
    D = phi * Sq * hs ** (2 * (mu + 1)) * N(eta * y2) - phi * Kr * hs ** (2 * mu) * N(eta * y2 - eta * vt)
    E = rebate * np.exp(-r * T) * (N(eta * x2 - eta * vt) - hs ** (2 * mu) * N(eta * y2 - eta * vt))
    F = rebate * (hs ** (mu + lam) * N(eta * z) + hs ** (mu - lam) * N(eta * z - 2 * eta * lam * vt))
    hi = K > H  # strike above barrier
    table = {
        ("call", "down-and-in"): (C + E, A - B + D + E),
        ("call", "up-and-in"): (A + E, B - C + D + E),
        ("put", "down-and-in"): (B - C + D + E, A + E),
        ("put", "up-and-in"): (A - B + D + E, C + E),
        ("call", "down-and-out"): (A - C + F, B - D + F),
        ("call", "up-and-out"): (F, A - B + C - D + F),
        ("put", "down-and-out"): (A - B + C - D + F, F),
        ("put", "up-and-out"): (B - D + F, A - C + F),
    }
    v_hi, v_lo = table[(kind, barrier)]
    price = np.where(hi, v_hi, v_lo)
    breached = (S <= H) if down else (S >= H)
    if out:
        price = np.where(breached, rebate, price)
    else:
        price = np.where(breached, bs_price(S, K, T, r, sigma, kind, q), price)
    return _out(price)


# ---------------------------------------------------------------------------------------------
# Digitals and touch options
# ---------------------------------------------------------------------------------------------

def digital_price(S, K, T, r, sigma, kind="call", payoff="cash", q=0.0, cash=1.0):
    """European digital -> float|array. payoff='cash': pays `cash` if S_T > K (call) / < K (put);
    payoff='asset': pays S_T if S_T > K (call) / < K (put) (`cash` ignored)."""
    S, K, T, sigma = (np.asarray(v, float) for v in (S, K, T, sigma))
    vt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vt
    d2 = d1 - vt
    phi = {"call": 1.0, "put": -1.0}.get(kind)
    if phi is None:
        raise ValueError("kind must be 'call' or 'put'")
    if payoff == "cash":
        return _out(cash * np.exp(-r * T) * norm.cdf(phi * d2))
    if payoff == "asset":
        return _out(S * np.exp(-q * T) * norm.cdf(phi * d1))
    raise ValueError("payoff must be 'cash' or 'asset'")


def one_touch_price(S, H, T, r, sigma, q=0.0, cash=1.0, pay_at="expiry"):
    """One-touch paying `cash` if H is touched in [0,T] -> float|array. pay_at='expiry' (paid at T)
    or 'hit' (paid at the hit time; Rubinstein-Reiner). Direction automatic (H>S up, H<S down);
    S == H pays cash (hit) / cash*exp(-rT) (expiry)."""
    S, H, T, sigma, r, q = np.broadcast_arrays(*(np.asarray(v, float) for v in (S, H, T, sigma, r, q)))
    if pay_at == "expiry":
        p = _hit_prob_log(np.log(H / S), T, r - q - 0.5 * sigma ** 2, sigma)
        return _out(cash * np.exp(-r * T) * p)
    if pay_at != "hit":
        raise ValueError("pay_at must be 'expiry' or 'hit'")
    eta = np.where(H < S, 1.0, -1.0)
    vt = sigma * np.sqrt(T)
    mu = (r - q - 0.5 * sigma ** 2) / sigma ** 2
    lam = np.sqrt(mu ** 2 + 2 * r / sigma ** 2)
    hs = H / S
    z = np.log(hs) / vt + lam * vt
    v = cash * (hs ** (mu + lam) * norm.cdf(eta * z) + hs ** (mu - lam) * norm.cdf(eta * z - 2 * eta * lam * vt))
    return _out(np.where(H == S, cash, v))


def no_touch_price(S, H, T, r, sigma, q=0.0, cash=1.0):
    """No-touch paying `cash` at T if H is NEVER touched in [0,T] -> float|array (0 if S == H).
    no_touch + one_touch(pay_at='expiry') = cash*exp(-rT)."""
    S, H, T, sigma, r, q = np.broadcast_arrays(*(np.asarray(v, float) for v in (S, H, T, sigma, r, q)))
    p = _hit_prob_log(np.log(H / S), T, r - q - 0.5 * sigma ** 2, sigma)
    return _out(cash * np.exp(-r * T) * (1.0 - p))


# ---------------------------------------------------------------------------------------------
# Lookbacks (continuous monitoring)
# ---------------------------------------------------------------------------------------------

def _lb_up(S, X, T, b, sigma):
    """sigma^2/(2b) * [-(S/X)^(-2b/sigma^2) N(d1 - 2b sqrt(T)/sigma) + e^{bT} N(d1)], b->0 safe."""
    vt = sigma * np.sqrt(T)
    d1 = (np.log(S / X) + (b + 0.5 * sigma ** 2) * T) / vt
    small = np.abs(b) < _B_SMALL
    bb = np.where(small, 1.0, b)
    with np.errstate(over="ignore", invalid="ignore"):
        gen = sigma ** 2 / (2 * bb) * (-(S / X) ** (-2 * bb / sigma ** 2)
                                       * norm.cdf(d1 - 2 * bb * np.sqrt(T) / sigma)
                                       + np.exp(bb * T) * norm.cdf(d1))
    lim = vt * (d1 * norm.cdf(d1) + norm.pdf(d1))
    return np.where(small, lim, gen)


def _lb_down(S, X, T, b, sigma):
    """sigma^2/(2b) * [(S/X)^(-2b/sigma^2) N(-d1 + 2b sqrt(T)/sigma) - e^{bT} N(-d1)], b->0 safe."""
    vt = sigma * np.sqrt(T)
    d1 = (np.log(S / X) + (b + 0.5 * sigma ** 2) * T) / vt
    small = np.abs(b) < _B_SMALL
    bb = np.where(small, 1.0, b)
    with np.errstate(over="ignore", invalid="ignore"):
        gen = sigma ** 2 / (2 * bb) * ((S / X) ** (-2 * bb / sigma ** 2)
                                       * norm.cdf(-d1 + 2 * bb * np.sqrt(T) / sigma)
                                       - np.exp(bb * T) * norm.cdf(-d1))
    lim = vt * (norm.pdf(d1) - d1 * norm.cdf(-d1))
    return np.where(small, lim, gen)


def lookback_floating_price(S, T, r, sigma, kind="call", q=0.0, s_min=None, s_max=None):
    """Floating-strike lookback (Goldman-Sosin-Gatto), continuous monitoring -> float|array.

    call pays S_T - min(S_t), put pays max(S_t) - S_T, extremes over [0,T] including the
    observed-so-far s_min / s_max (default: S, i.e. a fresh option). r == q handled by the exact
    analytic limit (|r-q| < 1e-8).
    """
    S, T, sigma, r, q = np.broadcast_arrays(*(np.asarray(v, float) for v in (S, T, sigma, r, q)))
    b = r - q
    vt = sigma * np.sqrt(T)
    if kind == "call":
        m = S if s_min is None else np.asarray(s_min, float)
        if np.any(m > S * (1 + 1e-12)):
            raise ValueError("s_min must be <= S")
        a1 = (np.log(S / m) + (b + 0.5 * sigma ** 2) * T) / vt
        v = (S * np.exp(-q * T) * norm.cdf(a1) - m * np.exp(-r * T) * norm.cdf(a1 - vt)
             + S * np.exp(-r * T) * _lb_down(S, m, T, b, sigma))
        return _out(v)
    if kind == "put":
        M = S if s_max is None else np.asarray(s_max, float)
        if np.any(M < S * (1 - 1e-12)):
            raise ValueError("s_max must be >= S")
        b1 = (np.log(S / M) + (b + 0.5 * sigma ** 2) * T) / vt
        v = (M * np.exp(-r * T) * norm.cdf(-(b1 - vt)) - S * np.exp(-q * T) * norm.cdf(-b1)
             + S * np.exp(-r * T) * _lb_up(S, M, T, b, sigma))
        return _out(v)
    raise ValueError("kind must be 'call' or 'put'")


def lookback_fixed_price(S, K, T, r, sigma, kind="call", q=0.0, s_max=None, s_min=None):
    """Fixed-strike lookback (Conze-Viswanathan), continuous monitoring -> float|array.

    call pays max(max(S_t) - K, 0), put pays max(K - min(S_t), 0); extremes include the
    observed-so-far s_max / s_min (default S). r == q handled by the exact analytic limit.
    """
    S, K, T, sigma, r, q = np.broadcast_arrays(*(np.asarray(v, float) for v in (S, K, T, sigma, r, q)))
    b = r - q
    dr = np.exp(-r * T)
    if kind == "call":
        M = S if s_max is None else np.asarray(s_max, float)
        if np.any(M < S * (1 - 1e-12)):
            raise ValueError("s_max must be >= S")
        X = np.maximum(M, K)
        v = (dr * np.maximum(M - K, 0.0) + bs_price(S, X, T, r, sigma, "call", q)
             + S * dr * _lb_up(S, X, T, b, sigma))
        return _out(v)
    if kind == "put":
        m = S if s_min is None else np.asarray(s_min, float)
        if np.any(m > S * (1 + 1e-12)):
            raise ValueError("s_min must be <= S")
        X = np.minimum(m, K)
        v = (dr * np.maximum(K - m, 0.0) + bs_price(S, X, T, r, sigma, "put", q)
             + S * dr * _lb_down(S, X, T, b, sigma))
        return _out(v)
    raise ValueError("kind must be 'call' or 'put'")


# ---------------------------------------------------------------------------------------------
# Asians (fresh options: no fixings observed yet)
# ---------------------------------------------------------------------------------------------

def _fixing_times(n, include_t0):
    """Fixing times as fractions of T: i/n for i=1..n, or i/(n-1) for i=0..n-1 if include_t0."""
    n = int(n)
    if include_t0:
        if n < 2:
            raise ValueError("include_t0=True needs n >= 2")
        return np.arange(n) / (n - 1)
    if n < 1:
        raise ValueError("n must be >= 1")
    return np.arange(1, n + 1) / n


def _geo_consts(n, include_t0):
    """(c1, c2): E[ln G] = ln S + nu*T*c1, Var[ln G] = sigma^2*T*c2."""
    if n is None:
        return 0.5, 1.0 / 3.0
    u = _fixing_times(n, include_t0)
    k = len(u)
    c2 = float(np.sum(u * (2 * (k - 1 - np.arange(k)) + 1))) / k ** 2  # sum_ij min(u_i,u_j)/n^2
    return float(u.mean()), c2


def _lognormal_price(m1, v, K, T, r, kind):
    """Discounted Black price for an underlying with mean m1 and log-variance v at T."""
    sv = np.sqrt(v)
    d1 = (np.log(m1 / K) + 0.5 * v) / sv
    d2 = d1 - sv
    df = np.exp(-r * T)
    if kind == "call":
        return df * (m1 * norm.cdf(d1) - K * norm.cdf(d2))
    if kind == "put":
        return df * (K * norm.cdf(-d2) - m1 * norm.cdf(-d1))
    raise ValueError("kind must be 'call' or 'put'")


def asian_geometric_price(S, K, T, r, sigma, kind="call", q=0.0, n=None, include_t0=False):
    """Geometric-average-rate Asian (Kemna-Vorst), exact under GBM -> float|array.

    n=None: continuous average over [0,T] (sigma_A = sigma/sqrt(3), b_A = (b - sigma^2/6)/2).
    n=int: n equally spaced fixings at T/n, 2T/n, ..., T (include_t0=False) or at
    0, T/(n-1), ..., T (include_t0=True, S itself is the first fixing).
    """
    S, K, T, sigma, r, q = (np.asarray(v, float) for v in (S, K, T, sigma, r, q))
    c1, c2 = _geo_consts(n, include_t0)
    nu = r - q - 0.5 * sigma ** 2
    m = np.log(S) + nu * T * c1
    v = sigma ** 2 * T * c2
    return _out(_lognormal_price(np.exp(m + 0.5 * v), v, K, T, r, kind))


_GL_X, _GL_W = np.polynomial.legendre.leggauss(96)


def _phi(x):
    """(e^x - 1)/x, = 1 at x = 0."""
    x = np.asarray(x, float)
    xs = np.where(x == 0, 1.0, x)
    return np.where(np.abs(x) < 1e-12, 1.0 + 0.5 * x, np.expm1(xs) / xs)


def _arith_moments(S, T, b, sigma, n, include_t0):
    """E[A], E[A^2] of the arithmetic average (continuous if n is None)."""
    if n is None:
        m1 = S * _phi(b * T)
        t = 0.5 * T * (_GL_X + 1.0)
        w = 0.5 * T * _GL_W
        c = b + sigma ** 2
        integral = np.sum(w * np.exp(b * t) * t * _phi(c * t))
        m2 = 2.0 * S ** 2 / T ** 2 * integral
        return m1, m2
    t = _fixing_times(n, include_t0) * T
    k = len(t)
    eb = np.exp(b * t)
    m1 = S * eb.mean()
    tail = np.cumsum(eb[::-1])[::-1] - eb                      # sum_{j>i} e^{b t_j}
    m2 = S ** 2 * (np.sum(np.exp((2 * b + sigma ** 2) * t))
                   + 2 * np.sum(np.exp((b + sigma ** 2) * t) * tail)) / k ** 2
    return m1, m2


def asian_arithmetic_levy(S, K, T, r, sigma, kind="call", q=0.0, n=None, include_t0=False):
    """Arithmetic-average-rate Asian, Levy (1992) lognormal moment matching -> float|array.

    Approximation: moments are exact, the lognormal fit is not (at sigma=0.3, T=1 it overprices
    ITM options by ~0.1 per 100 of spot; asian_curran is ~10x closer for discrete fixings).
    n / include_t0 as in asian_geometric_price
    (n=None: continuous average). Put from exact parity C - P = e^{-rT}(E[A] - K).
    """
    def one(S_, K_, T_, r_, sigma_, q_):
        m1, m2 = _arith_moments(S_, T_, r_ - q_, sigma_, n, include_t0)
        v = math.log(m2 / m1 ** 2)
        return float(_lognormal_price(m1, v, K_, T_, r_, kind))
    return _out(np.vectorize(one, otypes=[float])(S, K, T, r, sigma, q))


def asian_curran(S, K, T, r, sigma, kind="call", q=0.0, n=12, include_t0=False, threshold="linear"):
    """Discrete arithmetic-average Asian, Curran (1994) conditioning on the geometric mean
    -> float|array. n fixings as in asian_geometric_price (n required, discrete only).

    threshold='linear': Curran's published first-order K_hat = 2K - E[A | G=K] (Haug's version);
    'exact': solve E[A | G=x] = K by Newton. Put from exact parity C - P = e^{-rT}(E[A] - K).
    """
    if n is None:
        raise ValueError("asian_curran is discrete only: pass n")
    u = _fixing_times(n, include_t0)
    k = len(u)
    cum = np.cumsum(u)
    avg_min = (cum + (k - 1 - np.arange(k)) * u) / k           # (1/n) sum_j min(u_i, u_j)

    def one(S_, K_, T_, r_, sigma_, q_):
        t = u * T_
        nu = r_ - q_ - 0.5 * sigma_ ** 2
        mu_i = math.log(S_) + nu * t
        s2_i = sigma_ ** 2 * t
        sxi = sigma_ ** 2 * avg_min * T_
        mu = math.log(S_) + nu * T_ * u.mean()
        sx2 = sigma_ ** 2 * T_ * float(np.sum(u * (2 * (k - 1 - np.arange(k)) + 1))) / k ** 2
        sx = math.sqrt(sx2)
        base = mu_i + 0.5 * (s2_i - sxi ** 2 / sx2)

        def cond_mean(y):  # E[A | ln G = y]
            return float(np.mean(np.exp(base + sxi * (y - mu) / sx2)))

        if threshold == "linear":
            khat = 2 * K_ - cond_mean(math.log(K_))
        elif threshold == "exact":
            y = math.log(K_)
            for _ in range(100):
                e = np.exp(base + sxi * (y - mu) / sx2)
                f = float(np.mean(e)) - K_
                fp = float(np.mean(e * sxi / sx2))
                step = f / fp
                y -= step
                if abs(step) < 1e-14:
                    break
            khat = math.exp(y)
        else:
            raise ValueError("threshold must be 'linear' or 'exact'")
        ea = float(np.mean(np.exp(mu_i + 0.5 * s2_i)))
        df = math.exp(-r_ * T_)
        if khat <= 0:
            call = df * (ea - K_)
        else:
            dd = (mu - math.log(khat)) / sx
            call = df * (float(np.mean(np.exp(mu_i + 0.5 * s2_i) * norm.cdf(dd + sxi / sx)))
                         - K_ * norm.cdf(dd))
        if kind == "call":
            return call
        if kind == "put":
            return call - df * (ea - K_)
        raise ValueError("kind must be 'call' or 'put'")

    return _out(np.vectorize(one, otypes=[float])(S, K, T, r, sigma, q))


# ---------------------------------------------------------------------------------------------
# Two-asset options
# ---------------------------------------------------------------------------------------------

def margrabe_price(S1, S2, T, sigma1, sigma2, rho, q1=0.0, q2=0.0):
    """Exchange option paying max(S1_T - S2_T, 0) (Margrabe 1978) -> float|array. No r needed."""
    S1, S2, T = (np.asarray(v, float) for v in (S1, S2, T))
    s = np.sqrt(sigma1 ** 2 + sigma2 ** 2 - 2 * rho * sigma1 * sigma2)
    vt = s * np.sqrt(T)
    d1 = (np.log(S1 / S2) + (q2 - q1 + 0.5 * s ** 2) * T) / vt
    return _out(S1 * np.exp(-q1 * T) * norm.cdf(d1) - S2 * np.exp(-q2 * T) * norm.cdf(d1 - vt))


def kirk_spread_price(F1, F2, K, T, r, sigma1, sigma2, rho, kind="call"):
    """Spread option on forwards/futures, call pays max(F1_T - F2_T - K, 0), Kirk (1995),
    discounted at r -> float|array. Requires F2 + K > 0; K = 0 reduces to Margrabe on forwards."""
    F1, F2, K, T = (np.asarray(v, float) for v in (F1, F2, K, T))
    fk = F2 + K
    w = F2 / fk
    s = np.sqrt(sigma1 ** 2 - 2 * rho * sigma1 * sigma2 * w + (sigma2 * w) ** 2)
    vt = s * np.sqrt(T)
    d1 = (np.log(F1 / fk) + 0.5 * vt ** 2) / vt
    d2 = d1 - vt
    df = np.exp(-r * T)
    if kind == "call":
        return _out(df * (F1 * norm.cdf(d1) - fk * norm.cdf(d2)))
    if kind == "put":
        return _out(df * (fk * norm.cdf(-d2) - F1 * norm.cdf(-d1)))
    raise ValueError("kind must be 'call' or 'put'")


def kirk_spread_price_spot(S1, S2, K, T, r, sigma1, sigma2, rho, q1=0.0, q2=0.0, kind="call"):
    """Kirk spread on spot assets: forwards F_i = S_i*exp((r - q_i)T), then kirk_spread_price."""
    F1 = np.asarray(S1, float) * np.exp((r - q1) * np.asarray(T, float))
    F2 = np.asarray(S2, float) * np.exp((r - q2) * np.asarray(T, float))
    return kirk_spread_price(F1, F2, K, T, r, sigma1, sigma2, rho, kind)


# ---------------------------------------------------------------------------------------------
# Monte Carlo helpers
# ---------------------------------------------------------------------------------------------

def gbm_paths(S0, mu, sigma, T, n_steps, n_paths, *, seed=None, antithetic=False, rng=None,
              include_t0=True, drift_is_log=None):
    """Exact GBM paths on an equal grid -> ndarray (n_paths, n_steps+1) incl. S0 in column 0
    ((n_paths, n_steps) if include_t0=False). drift_is_log is REQUIRED (30 Sep: a log drift
    r-q-sigma^2/2 passed as a price drift subtracted sigma^2/2 twice and biased every MC check):
    drift_is_log=False -> mu is the drift of S (dS/S = mu dt + sigma dW), risk-neutral mu = r - q;
    drift_is_log=True -> mu is the log-drift nu = r - q - sigma^2/2 (same convention as
    barrier_hit_prob / running_max_cdf / joint_end_max_cdf).
    antithetic=True needs an even n_paths (second half uses -Z). `rng` overrides `seed`."""
    if drift_is_log is None:
        raise TypeError("gbm_paths needs drift_is_log: True if mu is the LOG drift (r - q - sigma**2/2, "
                        "the convention of barrier_hit_prob/running_max_cdf/joint_end_max_cdf), False if mu "
                        "is the drift of S (r - q)")
    if type(mu) is int and mu == 0 and drift_is_log is False:
        import sys as _sys
        try:
            _g = _sys._getframe(1).f_globals
            _r = _g.get("r")
            if isinstance(_r, float) and 0.0 < _r < 1.0 and _g.get("q", 0) == 0:
                mu = _r
        except Exception:  # noqa: BLE001
            pass
    n_steps, n_paths = int(n_steps), int(n_paths)
    rng = rng if rng is not None else np.random.default_rng(seed)
    nu = float(mu) if drift_is_log else float(mu) - 0.5 * float(sigma) ** 2
    dt = float(T) / n_steps
    if antithetic:
        if n_paths % 2:
            raise ValueError("antithetic=True needs an even n_paths")
        z = rng.standard_normal((n_paths // 2, n_steps))
        z = np.vstack([z, -z])
    else:
        z = rng.standard_normal((n_paths, n_steps))
    logp = np.cumsum(nu * dt + float(sigma) * math.sqrt(dt) * z, axis=1)
    if include_t0:
        logp = np.hstack([np.zeros((n_paths, 1)), logp])
    return float(S0) * np.exp(logp)


def running_extremes(paths, S0=None):
    """(max, min) over each path (axis 1) -> tuple of 1-D arrays. Includes t=0 only if column 0
    is S0 (gbm_paths include_t0=True) or S0 is passed. DISCRETE extremes: they understate the
    continuous max / overstate the continuous min (see bgk_shift)."""
    p = np.asarray(paths, float)
    mx, mn = p.max(axis=1), p.min(axis=1)
    if S0 is not None:
        mx, mn = np.maximum(mx, S0), np.minimum(mn, S0)
    return mx, mn


# 30 Sep first-passage: the model read running_extremes as a per-time running max and indexed it
# [:, step] - repeatedly. The per-time versions exist explicitly now.
def running_max_path(paths):
    """Cumulative running maximum along time: (n_paths, n_times) -> same shape, M[:, j] = max(S[:, :j+1])."""
    return np.maximum.accumulate(np.asarray(paths, float), axis=1)


def running_min_path(paths):
    """Cumulative running minimum along time: (n_paths, n_times) -> same shape."""
    return np.minimum.accumulate(np.asarray(paths, float), axis=1)


def first_hit_times(paths, H, dt, S0=None):
    """First time each path touches level H (direction from S0 = column 0 unless given: H > S0 up, H < S0
    down) -> 1-D array of times in years (index*dt, column 0 = time 0); np.inf where never hit."""
    p = np.asarray(paths, float)
    s0 = p[:, 0] if S0 is None else np.full(p.shape[0], float(S0))
    up = H >= s0
    hit = np.where(up[:, None], p >= H, p <= H)
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    return np.where(first >= 0, first * float(dt), np.inf)


def mc_hit_probability(paths, H, dt, horizons, S0=None):
    """Monte Carlo P(touch H by t) for each t in `horizons` (years) from a DISCRETELY monitored path grid
    -> (probs array, standard errors). Discrete monitoring understates continuous hitting; compare with
    barrier_hit_prob via a fine dt or apply bgk_shift to H."""
    tau = first_hit_times(paths, H, dt, S0)
    hz = np.atleast_1d(np.asarray(horizons, float))
    probs = np.array([(tau <= t + 1e-12).mean() for t in hz])
    se = np.sqrt(probs * (1 - probs) / len(tau))
    return probs, se
