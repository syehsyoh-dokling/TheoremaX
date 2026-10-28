"""Compound (Geske 1979) and chooser (Rubinstein 1991) options, closed form + Monte Carlo check.

Units: rates r and dividend yield q continuously compounded per year, cost of carry b = r - q,
times in years, sigma per sqrt(year). Scalar inputs, float outputs (the chooser_simple_price
also broadcasts arrays).

Compound option: at T1 the holder may pay K1 (outer call) or receive K1 (outer put) for a vanilla
European option (the "inner" option, strike K2, expiry T2 > T1) on S.  kind = "<outer>_on_<inner>":
    call_on_call  pays (C(S_T1) - K1)+    exercised when S_T1 > S*
    put_on_call   pays (K1 - C(S_T1))+    exercised when S_T1 < S*
    call_on_put   pays (P(S_T1) - K1)+    exercised when S_T1 < S*
    put_on_put    pays (K1 - P(S_T1))+    exercised when S_T1 > S*
where C/P = BS value with remaining life T2 - T1 and S* solves BS(S*, K2, T2 - T1) = K1.

Haug (2007) notation, rho = sqrt(T1/T2), M(a, b; rho) = standard bivariate normal cdf:
    y1 = [ln(S/S*) + (b + sigma^2/2) T1] / (sigma sqrt T1),   y2 = y1 - sigma sqrt T1
    z1 = [ln(S/K2) + (b + sigma^2/2) T2] / (sigma sqrt T2),   z2 = z1 - sigma sqrt T2
    CoC = S e^{-qT2} M(z1, y1; rho)   - K2 e^{-rT2} M(z2, y2; rho)   - K1 e^{-rT1} N(y2)
    PoC = K2 e^{-rT2} M(z2, -y2; -rho) - S e^{-qT2} M(z1, -y1; -rho) + K1 e^{-rT1} N(-y2)
    CoP = K2 e^{-rT2} M(-z2, -y2; rho) - S e^{-qT2} M(-z1, -y1; rho) - K1 e^{-rT1} N(-y2)
    PoP = S e^{-qT2} M(-z1, y1; -rho)  - K2 e^{-rT2} M(-z2, y2; -rho) + K1 e^{-rT1} N(y2)
Parity: CoX - PoX = BS(S, K2, T2, X) - K1 e^{-rT1}   (compound_parity_rhs).

Bivariate normal: Genz (2004) algorithm (Drezner-Wesolowsky 1990 Gauss-Legendre integration of
the Plackett identity, with Genz's refinement for |rho| >= 0.925), a port of Genz's BVNU. Absolute
accuracy ~1e-15 (double precision), deterministic - no Monte Carlo / QMC noise, unlike
scipy.stats.multivariate_normal.cdf whose default tolerance is 1e-5.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .options import bs_price

# ---------------------------------------------------------------------------------------------
# Bivariate normal cdf (Genz BVNU)
# ---------------------------------------------------------------------------------------------

_GL = {
    6: ([0.1713244923791705, 0.3607615730481384, 0.4679139345726904],
        [0.9324695142031522, 0.6612093864662647, 0.2386191860831970]),
    12: ([0.04717533638651177, 0.1069393259953183, 0.1600783285433464,
          0.2031674267230659, 0.2334925365383547, 0.2491470458134029],
         [0.9815606342467191, 0.9041172563704750, 0.7699026741943050,
          0.5873179542866171, 0.3678314989981802, 0.1252334085114692]),
    20: ([0.01761400713915212, 0.04060142980038694, 0.06267204833410906,
          0.08327674157670475, 0.1019301198172404, 0.1181945319615184,
          0.1316886384491766, 0.1420961093183821, 0.1491729864726037,
          0.1527533871307259],
         [0.9931285991850949, 0.9639719272779138, 0.9122344282513259,
          0.8391169718222188, 0.7463319064601508, 0.6360536807265150,
          0.5108670019508271, 0.3737060887154196, 0.2277858511416451,
          0.07652652113349733]),
}
_GLW = {n: (np.array(w + w), np.array([1 - v for v in x] + [1 + v for v in x]))
        for n, (w, x) in _GL.items()}


def _phid(x):
    return float(norm.cdf(x))


def _bvnu(dh, dk, r):
    """P(X > dh, Y > dk) for standard bivariate normal with correlation r (Genz BVNU)."""
    if dh == math.inf or dk == math.inf:
        return 0.0
    if dh == -math.inf:
        return 1.0 if dk == -math.inf else _phid(-dk)
    if dk == -math.inf:
        return _phid(-dh)
    if r == 0.0:
        return _phid(-dh) * _phid(-dk)
    if r >= 1.0:
        return _phid(-max(dh, dk))
    if r <= -1.0:
        return max(0.0, _phid(-dh) - _phid(dk))
    tp = 2.0 * math.pi
    h, k = dh, dk
    hk = h * k
    ar = abs(r)
    w, x = _GLW[6] if ar < 0.3 else (_GLW[12] if ar < 0.75 else _GLW[20])
    if ar < 0.925:
        hs = (h * h + k * k) / 2.0
        asr = math.asin(r) / 2.0
        sn = np.sin(asr * x)
        bvn = float(np.exp((sn * hk - hs) / (1.0 - sn ** 2)) @ w)
        bvn = bvn * asr / tp + _phid(-h) * _phid(-k)
    else:
        if r < 0:
            k = -k
            hk = -hk
        bvn = 0.0
        a_s = 1.0 - r * r
        a = math.sqrt(a_s)
        bs = (h - k) ** 2
        asr = -(bs / a_s + hk) / 2.0
        c = (4.0 - hk) / 8.0
        d = (12.0 - hk) / 80.0
        if asr > -100:
            bvn = a * math.exp(asr) * (1 - c * (bs - a_s) * (1 - d * bs) / 3 + c * d * a_s ** 2)
        if hk > -100:
            b = math.sqrt(bs)
            sp = math.sqrt(tp) * _phid(-b / a)
            bvn = bvn - math.exp(-hk / 2) * sp * b * (1 - c * bs * (1 - d * bs) / 3)
        a = a / 2.0
        xs = (a * x) ** 2
        asr_v = -(bs / xs + hk) / 2.0
        ix = asr_v > -100
        xs = xs[ix]
        sp = 1 + c * xs * (1 + 5 * d * xs)
        rs = np.sqrt(1 - xs)
        ep = np.exp(-(hk / 2) * xs / (1 + rs) ** 2) / rs
        bvn = (a * float((np.exp(asr_v[ix]) * (sp - ep)) @ w[ix]) - bvn) / tp
        if r > 0:
            bvn = bvn + _phid(-max(h, k))
        elif h >= k:
            bvn = -bvn
        else:
            L = _phid(k) - _phid(h) if h < 0 else _phid(-h) - _phid(-k)
            bvn = L - bvn
    return max(0.0, min(1.0, bvn))


def bivariate_normal_cdf(a, b, rho):
    """M(a, b; rho) = P(X <= a, Y <= b), standard bivariate normal, correlation rho -> float.

    Genz (2004) BVNU algorithm (Drezner-Wesolowsky Gauss-Legendre), accuracy ~1e-15; +-inf ok.
    """
    return _bvnu(-float(a), -float(b), float(rho))


# ---------------------------------------------------------------------------------------------
# Compound options (Geske 1979)
# ---------------------------------------------------------------------------------------------

_KINDS = {"call_on_call": ("call", "call"), "put_on_call": ("put", "call"),
          "call_on_put": ("call", "put"), "put_on_put": ("put", "put"),
          "coc": ("call", "call"), "poc": ("put", "call"), "cop": ("call", "put"),
          "pop": ("put", "put")}


def _parse_kind(kind):
    key = str(kind).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _KINDS:
        raise ValueError("kind must be 'call_on_call' | 'put_on_call' | 'call_on_put' | 'put_on_put'")
    return _KINDS[key]


def _check_inner(inner):
    inner = str(inner).strip().lower()
    if inner not in ("call", "put"):
        raise ValueError("inner must be 'call' or 'put'")
    return inner


def _check_times(T1, T2):
    if not (T1 > 0 and T2 > T1):
        raise ValueError("need 0 < T1 < T2 (T1 = compound expiry, T2 = underlying option expiry)")


def geske_critical_price(K1, K2, T1, T2, r, sigma, q=0.0, inner="call"):
    """S* with BS(S*, K2, T2 - T1, r, sigma, inner, q) = K1 -> float (the T1 exercise boundary).

    Solved with brentq in log(S) on an expanding bracket (rel. tol 1e-15). Edge cases:
    inner call: K1 <= 0 -> 0.0 (always worth exercising the outer call).
    inner put:  K1 <= 0 -> inf; K1 >= K2 e^{-r(T2-T1)} (the put's supremum as S -> 0) -> 0.0.
    """
    inner = _check_inner(inner)
    _check_times(T1, T2)
    tau = T2 - T1
    if K1 <= 0:
        return 0.0 if inner == "call" else math.inf
    if inner == "put" and K1 >= K2 * math.exp(-r * tau):
        return 0.0

    def f(x):
        return float(bs_price(math.exp(x), K2, tau, r, sigma, inner, q)) - K1

    x0 = math.log(K2)
    lo, hi = x0 - 1.0, x0 + 1.0
    # value(call) increases in S, value(put) decreases: expand until f changes sign
    for _ in range(200):
        flo, fhi = f(lo), f(hi)
        if flo * fhi <= 0:
            break
        if (inner == "call") == (flo > 0):   # root lies below lo
            lo -= 2.0 * (hi - lo)
        else:
            hi += 2.0 * (hi - lo)
    else:
        raise RuntimeError("geske_critical_price: could not bracket S*")
    return math.exp(brentq(f, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=500))


def compound_option_price(S, K1, K2, T1, T2, r, sigma, q=0.0, kind="call_on_call"):
    """Geske (1979) compound option price -> float.  See the module docstring for the formulas.

    S spot; K1 strike of the compound (paid/received at T1); K2 strike of the inner option that
    expires at T2 > T1; kind 'call_on_call' | 'put_on_call' | 'call_on_put' | 'put_on_put'
    (also 'coc','poc','cop','pop'). Continuous dividend yield q, b = r - q.
    Satisfies CoX - PoX = compound_parity_rhs(...) to ~1e-12.
    """
    outer, inner = _parse_kind(kind)
    _check_times(T1, T2)
    s_star = geske_critical_price(K1, K2, T1, T2, r, sigma, q=q, inner=inner)
    b = r - q
    v1, v2 = sigma * math.sqrt(T1), sigma * math.sqrt(T2)
    with np.errstate(divide="ignore"):
        lss = math.inf if s_star == 0.0 else (-math.inf if s_star == math.inf else math.log(S / s_star))
    y1 = (lss + (b + 0.5 * sigma ** 2) * T1) / v1 if math.isfinite(lss) else lss
    y2 = y1 - v1 if math.isfinite(y1) else y1
    z1 = (math.log(S / K2) + (b + 0.5 * sigma ** 2) * T2) / v2
    z2 = z1 - v2
    rho = math.sqrt(T1 / T2)
    M = bivariate_normal_cdf
    N = _phid
    sq, k2r, k1r = S * math.exp(-q * T2), K2 * math.exp(-r * T2), K1 * math.exp(-r * T1)
    if inner == "call" and outer == "call":
        v = sq * M(z1, y1, rho) - k2r * M(z2, y2, rho) - k1r * N(y2)
    elif inner == "call":
        v = k2r * M(z2, -y2, -rho) - sq * M(z1, -y1, -rho) + k1r * N(-y2)
    elif outer == "call":
        v = k2r * M(-z2, -y2, rho) - sq * M(-z1, -y1, rho) - k1r * N(-y2)
    else:
        v = sq * M(-z1, y1, -rho) - k2r * M(-z2, y2, -rho) + k1r * N(y2)
    return float(v)


def compound_parity_rhs(S, K1, K2, T1, T2, r, sigma, q=0.0, inner="call"):
    """Compound put-call parity RHS -> float:
    call_on_<inner> - put_on_<inner> = BS(S, K2, T2, inner) - K1 e^{-r T1}."""
    inner = _check_inner(inner)
    return float(bs_price(S, K2, T2, r, sigma, inner, q)) - K1 * math.exp(-r * T1)


def compound_option_mc(S, K1, K2, T1, T2, r, sigma, q=0.0, kind="call_on_call", n_paths=200000,
                       seed=None):
    """Monte Carlo check of compound_option_price -> tuple (price, stderr).

    S_T1 simulated exactly (GBM, drift r - q), inner option valued with Black-Scholes at T1
    (life T2 - T1), outer payoff (inner - K1)+ or (K1 - inner)+, discounted by e^{-r T1}.
    """
    outer, inner = _parse_kind(kind)
    _check_times(T1, T2)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(int(n_paths))
    st1 = S * np.exp((r - q - 0.5 * sigma ** 2) * T1 + sigma * math.sqrt(T1) * z)
    val = np.asarray(bs_price(st1, K2, T2 - T1, r, sigma, inner, q), dtype=float)
    pay = np.maximum(val - K1, 0.0) if outer == "call" else np.maximum(K1 - val, 0.0)
    disc = pay * math.exp(-r * T1)
    return float(disc.mean()), float(disc.std(ddof=1) / math.sqrt(disc.size))


# ---------------------------------------------------------------------------------------------
# Chooser options (Rubinstein 1991)
# ---------------------------------------------------------------------------------------------

def chooser_simple_price(S, K, t_choose, T, r, sigma, q=0.0):
    """Simple chooser (Rubinstein 1991): at t_choose pick a call OR put, same K, same expiry T.
    -> float|array.

    At t the holder gets max(C_t, P_t) = C_t + (K e^{-r(T-t)} - S_t e^{-q(T-t)})+   (put-call parity)
                                     = C_t + e^{-q(T-t)} (K e^{-(r-q)(T-t)} - S_t)+ ,
    so  V = C_BS(S, K, T) + e^{-q(T-t)} P_BS(S, K e^{-(r-q)(T-t)}, t)   (r, sigma, q in both).
    Equivalent (Haug): C_BS(K,T) + K e^{-rT} N(-y2) - S e^{-qT} N(-y1),
    y1 = [ln(S/K) + bT + sigma^2 t/2]/(sigma sqrt t), y2 = y1 - sigma sqrt t.
    t_choose = 0 -> max(call, put); t_choose = T -> call + put (straddle).
    """
    tau = T - t_choose
    call = bs_price(S, K, T, r, sigma, "call", q)
    if np.all(np.asarray(t_choose) <= 0):
        put = bs_price(S, K, T, r, sigma, "put", q)
        v = np.maximum(call, put)
    else:
        v = call + np.exp(-q * tau) * bs_price(S, K * np.exp(-(r - q) * tau), t_choose, r, sigma, "put", q)
    v = np.asarray(v, dtype=float)
    return float(v) if v.ndim == 0 else v


def chooser_complex_price(S, Kc, Kp, t_choose, Tc, Tp, r, sigma, q=0.0):
    """Complex chooser (Rubinstein 1991): at t_choose pick a call(Kc, Tc) or a put(Kp, Tp).
    -> float.  Needs 0 < t_choose < min(Tc, Tp).

    I solves C_BS(I, Kc, Tc - t) = P_BS(I, Kp, Tp - t) (unique: C - P is increasing in S);
    the call is chosen when S_t > I.  rho1 = sqrt(t/Tc), rho2 = sqrt(t/Tp),
    d1 = [ln(S/I) + (b + sigma^2/2) t]/(sigma sqrt t), d2 = d1 - sigma sqrt t,
    y1 = [ln(S/Kc) + (b + sigma^2/2) Tc]/(sigma sqrt Tc), y2 = y1 - sigma sqrt Tc,
    y3 = [ln(S/Kp) + (b + sigma^2/2) Tp]/(sigma sqrt Tp), y4 = y3 - sigma sqrt Tp,
    V = S e^{-qTc} M(d1, y1; rho1) - Kc e^{-rTc} M(d2, y2; rho1)
        - S e^{-qTp} M(-d1, -y3; rho2) + Kp e^{-rTp} M(-d2, -y4; rho2).
    Kc = Kp and Tc = Tp reproduces chooser_simple_price.
    """
    t = t_choose
    if not (0 < t < Tc and t < Tp):
        raise ValueError("need 0 < t_choose < min(Tc, Tp)")
    b = r - q

    def g(x):
        s = math.exp(x)
        return (float(bs_price(s, Kc, Tc - t, r, sigma, "call", q))
                - float(bs_price(s, Kp, Tp - t, r, sigma, "put", q)))

    x0 = math.log(math.sqrt(Kc * Kp))
    lo, hi = x0 - 1.0, x0 + 1.0
    for _ in range(200):
        glo, ghi = g(lo), g(hi)
        if glo * ghi <= 0:
            break
        if glo > 0:
            lo -= 2.0 * (hi - lo)
        else:
            hi += 2.0 * (hi - lo)
    else:
        raise RuntimeError("chooser_complex_price: could not bracket the critical price")
    crit = math.exp(brentq(g, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=500))
    vt = sigma * math.sqrt(t)
    d1 = (math.log(S / crit) + (b + 0.5 * sigma ** 2) * t) / vt
    d2 = d1 - vt
    y1 = (math.log(S / Kc) + (b + 0.5 * sigma ** 2) * Tc) / (sigma * math.sqrt(Tc))
    y2 = y1 - sigma * math.sqrt(Tc)
    y3 = (math.log(S / Kp) + (b + 0.5 * sigma ** 2) * Tp) / (sigma * math.sqrt(Tp))
    y4 = y3 - sigma * math.sqrt(Tp)
    r1, r2 = math.sqrt(t / Tc), math.sqrt(t / Tp)
    M = bivariate_normal_cdf
    v = (S * math.exp(-q * Tc) * M(d1, y1, r1) - Kc * math.exp(-r * Tc) * M(d2, y2, r1)
         - S * math.exp(-q * Tp) * M(-d1, -y3, r2) + Kp * math.exp(-r * Tp) * M(-d2, -y4, r2))
    return float(v)
