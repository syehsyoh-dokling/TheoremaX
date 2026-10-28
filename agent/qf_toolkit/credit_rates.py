"""Credit risk and short-rate / rate-option primitives with every convention as a keyword.

Units, unless a function says otherwise: T and t in years; r continuously compounded per year;
volatilities annualised decimals (0.20, not 20); CDS spreads DECIMAL per year (0.0124, not
124 bp: divide bp by 10000); PD, LGD, recovery as fractions in [0, 1]; normal (Bachelier) vol in
rate units per sqrt(year) (85 bp/yr -> 0.0085).

Sections
  structural : merton_model, merton_calibrate, black_cox_pd
  hazard/CDS : hazard_from_spread, pd_from_spread, survival_prob, cds_legs, cds_par_spread,
               cds_bootstrap
  portfolio  : vasicek_loss_quantile, vasicek_loss_distribution, basel_irb_capital,
               one_factor_mc_losses
  migration  : matrix_power_migration, generator_from_matrix, cohort_matrix,
               cumulative_default_probs
  short rate : vasicek_zcb, cir_zcb, hull_white_zcb, vasicek_zcb_option,
               hull_white_zcb_option, vasicek_swaption
  rate opts  : black_caplet, black_swaption, bachelier_price, bachelier_caplet,
               bachelier_swaption, swap_rate_and_annuity
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import brentq
from scipy.stats import norm

from .options import bs_price


def _out(x):
    """0-d arrays -> float; arrays unchanged."""
    a = np.asarray(x, dtype=float)
    return float(a) if a.ndim == 0 else a


def _df_fn(discount_curve_fn):
    """Callable t -> DF, or a number = flat continuously compounded rate."""
    if discount_curve_fn is None:
        raise ValueError("pass discount_curve_fn: a callable t -> DF(0,t), or a flat continuous "
                         "rate as a number")
    if callable(discount_curve_fn):
        return lambda t: float(discount_curve_fn(t))
    r = float(discount_curve_fn)
    return lambda t: math.exp(-r * t)


# ---------------------------------------------------------------------------------------------
# Structural credit
# ---------------------------------------------------------------------------------------------

def merton_model(V, sigma_V, D, T, r, *, mu=None) -> dict:
    """Merton (1974): equity = European call on assets V with strike = debt face D at T.

    Returns dict (all floats):
      equity, debt (= V - equity), d1, d2,
      pd_rn = N(-d2) risk-neutral default prob by T, dd = d2 (risk-neutral distance to default),
      dd_real, pd_real = N(-dd_real) using asset drift mu (only if mu is given, else nan),
      dd_kmv = (V - D)/(sigma_V V) (D read as the default point),
      sigma_E = N(d1) sigma_V V / equity, delta = N(d1),
      debt_yield = -ln(debt/D)/T, spread = debt_yield - r (continuous, decimal per year),
      expected_loss = 1 - debt/(D e^{-rT}) (risk-neutral, fraction of PV of promised payment),
      recovery = 1 - expected_loss/pd_rn = E[V_T | V_T < D]/D (risk-neutral),
      leverage = D e^{-rT}/V.
    """
    V, sigma_V, D, T, r = map(float, (V, sigma_V, D, T, r))
    sq = sigma_V * math.sqrt(T)
    d1 = (math.log(V / D) + (r + 0.5 * sigma_V ** 2) * T) / sq
    d2 = d1 - sq
    equity = float(bs_price(V, D, T, r, sigma_V, "call"))
    debt = V - equity
    pv_d = D * math.exp(-r * T)
    pd_rn = float(norm.cdf(-d2))
    el = 1.0 - debt / pv_d
    if mu is not None:
        dd_real = (math.log(V / D) + (float(mu) - 0.5 * sigma_V ** 2) * T) / sq
        pd_real = float(norm.cdf(-dd_real))
    else:
        dd_real = pd_real = float("nan")
    y = -math.log(debt / D) / T
    return {
        "equity": equity, "debt": debt, "d1": d1, "d2": d2,
        "pd_rn": pd_rn, "dd": d2, "dd_real": dd_real, "pd_real": pd_real,
        "dd_kmv": (V - D) / (sigma_V * V),
        "sigma_E": float(norm.cdf(d1)) * sigma_V * V / equity if equity > 0 else float("inf"),
        "delta": float(norm.cdf(d1)),
        "debt_yield": y, "spread": y - r,
        "expected_loss": el,
        "recovery": 1.0 - el / pd_rn if pd_rn > 0 else float("nan"),
        "leverage": pv_d / V,
    }


def merton_calibrate(E, sigma_E, D, T, r, *, mu=None, tol=1e-12) -> dict:
    """Solve the two Merton equations for (V, sigma_V) given equity value E and equity vol sigma_E:
        E = V N(d1) - D e^{-rT} N(d2)     and     sigma_E E = N(d1) sigma_V V.

    Robust nested 1-D solve (no fsolve starting-point failures): for each sigma_V the equity
    equation is solved for V in the exact bracket [E, E + D e^{-rT}]; the outer volatility
    equation is then bracketed by scanning sigma_V upward from ~0 and solved with brentq.
    Returns merton_model(V, sigma_V, ...) dict plus keys V, sigma_V, resid_equity, resid_vol.
    """
    E, sigma_E, D, T, r = map(float, (E, sigma_E, D, T, r))
    pv_d = D * math.exp(-r * T)

    def v_of(s):
        f = lambda v: float(bs_price(v, D, T, r, s, "call")) - E  # noqa: E731
        lo, hi = E, E + pv_d
        flo, fhi = f(lo), f(hi)
        if flo >= 0:
            return lo
        if fhi <= 0:
            return hi
        return brentq(f, lo, hi, xtol=tol * max(1.0, hi), maxiter=500)

    def g(s):
        v = v_of(s)
        d1 = (math.log(v / D) + (r + 0.5 * s * s) * T) / (s * math.sqrt(T))
        return float(norm.cdf(d1)) * s * v - sigma_E * E

    grid = np.concatenate([np.geomspace(1e-6, sigma_E, 60), sigma_E * np.array([1.5, 2, 4, 8, 16])])
    prev_s, prev_g = grid[0], g(grid[0])
    s_star = None
    for s in grid[1:]:
        gs = g(s)
        if prev_g == 0:
            s_star = prev_s
            break
        if prev_g * gs <= 0:
            s_star = brentq(g, prev_s, s, xtol=1e-14, maxiter=500)
            break
        prev_s, prev_g = s, gs
    if s_star is None:
        raise ValueError("merton_calibrate: no root for sigma_V; check units (sigma_E decimal, "
                         "E and D in the same currency units)")
    V = v_of(s_star)
    out = merton_model(V, s_star, D, T, r, mu=mu)
    out.update({"V": V, "sigma_V": s_star,
                "resid_equity": out["equity"] - E,
                "resid_vol": out["sigma_E"] * out["equity"] - sigma_E * E})
    return out


def black_cox_pd(V, barrier, sigma_V, T, mu, *, payout=0.0, barrier_growth=0.0):
    """First-passage default probability P(min_{t<=T} V_t/H(t) <= 1) for GBM assets.

    dV/V = (mu - payout) dt + sigma_V dW; barrier H(t) = barrier * exp(barrier_growth * t).
    mu = r for the risk-neutral PD, the real-world asset drift otherwise.
    Black-Cox (1976) barrier K e^{-gamma (T - t)}: pass barrier = K e^{-gamma T},
    barrier_growth = gamma. With m = mu - payout - sigma^2/2 - barrier_growth, x0 = ln(V/barrier):
        PD = N((-x0 - m T)/(s sqrt T)) + exp(-2 m x0 / s^2) N((-x0 + m T)/(s sqrt T)).
    Returns 1.0 if V <= barrier already. Always >= the Merton-style terminal PD with same barrier.
    """
    V, H, s, T = float(V), float(barrier), float(sigma_V), float(T)
    if V <= H:
        return 1.0
    m = float(mu) - float(payout) - 0.5 * s * s - float(barrier_growth)
    x0 = math.log(V / H)
    sq = s * math.sqrt(T)
    return float(norm.cdf((-x0 - m * T) / sq) + math.exp(-2 * m * x0 / (s * s)) * norm.cdf((-x0 + m * T) / sq))


# ---------------------------------------------------------------------------------------------
# Hazard rates and CDS
# ---------------------------------------------------------------------------------------------

def hazard_from_spread(spread, recovery=0.4):
    """Credit triangle: lambda = spread / (1 - recovery). spread decimal per year."""
    return _out(np.asarray(spread, float) / (1.0 - np.asarray(recovery, float)))


def pd_from_spread(spread, T, recovery=0.4):
    """Cumulative default prob by T from a flat spread: 1 - exp(-spread T/(1-R)) (continuous)."""
    return _out(1.0 - np.exp(-np.asarray(spread, float) * np.asarray(T, float) / (1.0 - recovery)))


def survival_prob(t, hazard_tenors, hazards):
    """S(t) = exp(-int_0^t lambda) with piecewise-constant hazards: hazards[k] applies on
    (hazard_tenors[k-1], hazard_tenors[k]] (first segment starts at 0); the last hazard is
    extrapolated flat beyond the last tenor. hazard_tenors=None with a scalar hazard = flat."""
    lam = np.atleast_1d(np.asarray(hazards, float))
    if hazard_tenors is None:
        knots = np.array([np.inf]) if lam.size == 1 else None
        if knots is None:
            raise ValueError("hazard_tenors required for more than one hazard")
    else:
        knots = np.atleast_1d(np.asarray(hazard_tenors, float)).copy()
        if knots.size != lam.size:
            raise ValueError("hazard_tenors and hazards must have equal length")
        knots[-1] = np.inf
    starts = np.concatenate([[0.0], knots[:-1]])
    tt = np.asarray(t, float)
    seg = np.clip(tt[..., None] - starts, 0.0, knots - starts)
    return _out(np.exp(-(seg * lam).sum(axis=-1)))


def _schedule(maturity, freq):
    n = max(1, int(math.ceil(maturity * freq - 1e-9)))
    times = np.arange(1, n + 1, dtype=float) / freq
    times[-1] = maturity
    return times


def cds_legs(hazards, hazard_tenors, maturity, recovery=0.4, discount_curve_fn=None, *, freq=4,
             accrual=True, protection_df="end", substeps=1) -> dict:
    """CDS legs per unit notional under piecewise-constant hazards (see survival_prob).

    Premium dates k/freq up to maturity (short final stub if maturity is not on the grid),
    accrual fraction = period length (no day count; convert dates to years first).
      premium_annuity = sum_i Delta_i DF(t_i) S(t_i)
      accrual_on_default (if accrual) = sum over sub-intervals (u,v] of
                                        (midpoint - period start) * DF_x * (S(u) - S(v))
      protection = (1 - R) sum over sub-intervals DF_x (S(u) - S(v))
    DF_x = DF(v) for protection_df="end" (theory-card formula) or DF((u+v)/2) for "mid" (Hull's
    mid-period default). substeps = sub-intervals per premium period (finer = closer to continuous).
    With accrual, substeps=1 gives the half-period accrual Delta/2.
    discount_curve_fn: callable t -> DF(0,t), or a number = flat continuous rate.
    Returns dict{rpv01 (= annuity + accrual), premium_annuity, accrual_on_default, protection,
    par_spread = protection / rpv01}. Protection-buyer PV at contract spread c = protection - c*rpv01;
    upfront (buyer pays) = (par_spread - c) * rpv01.
    """
    if protection_df not in ("end", "mid"):
        raise ValueError("protection_df must be 'end' or 'mid'")
    df = _df_fn(discount_curve_fn)
    S = lambda x: float(survival_prob(x, hazard_tenors, hazards))  # noqa: E731
    times = _schedule(float(maturity), freq)
    ann = acc = prot = 0.0
    a = 0.0
    for b in times:
        ann += (b - a) * df(b) * S(b)
        grid = np.linspace(a, b, int(substeps) + 1)
        for u, v in zip(grid[:-1], grid[1:]):
            dP = S(u) - S(v)
            dfx = df(v) if protection_df == "end" else df(0.5 * (u + v))
            prot += (1.0 - recovery) * dfx * dP
            if accrual:
                acc += (0.5 * (u + v) - a) * dfx * dP
        a = b
    rpv01 = ann + acc
    return {"rpv01": rpv01, "premium_annuity": ann, "accrual_on_default": acc,
            "protection": prot, "par_spread": prot / rpv01}


def cds_par_spread(hazards, hazard_tenors, maturity, recovery=0.4, discount_curve_fn=None, *,
                   freq=4, accrual=True, protection_df="end", substeps=1) -> float:
    """Par CDS spread (decimal per year) = protection / rpv01. Arguments as cds_legs."""
    return cds_legs(hazards, hazard_tenors, maturity, recovery, discount_curve_fn, freq=freq,
                    accrual=accrual, protection_df=protection_df, substeps=substeps)["par_spread"]


def cds_bootstrap(tenors, spreads, recovery=0.4, discount_curve_fn=None, freq=4, accrual=True, *,
                  protection_df="end", substeps=1, allow_negative=False):
    """Bootstrap piecewise-constant hazards so each tenor's CDS at its par spread has PV 0.

    tenors in years (increasing), spreads DECIMAL (bp/10000). Hazard k applies on
    (tenors[k-1], tenors[k]]. Solved in order with brentq on [1e-12, 5] (negative hazards only if
    allow_negative=True; otherwise an inverted curve raises ValueError instead of flooring).
    Returns pandas DataFrame[tenor, spread, hazard, survival, cum_pd, rpv01, repriced_spread];
    feed df.tenor, df.hazard into survival_prob / cds_par_spread.
    """
    import pandas as pd

    T = np.asarray(tenors, float)
    s = np.asarray(spreads, float)
    if np.any(np.diff(T) <= 0):
        raise ValueError("tenors must be strictly increasing")
    if np.any(s > 1.0):
        raise ValueError("spreads look like basis points; pass decimals (bp / 10000)")
    kw = dict(freq=freq, accrual=accrual, protection_df=protection_df, substeps=substeps)
    haz = []
    for k in range(len(T)):
        def pv(lam, k=k):
            legs = cds_legs(haz + [lam], T[:k + 1], T[k], recovery, discount_curve_fn, **kw)
            return s[k] * legs["rpv01"] - legs["protection"]
        lo = -2.0 if allow_negative else 1e-12
        if pv(lo) * pv(5.0) > 0:
            raise ValueError(f"no hazard in [{lo}, 5] reprices tenor {T[k]} (spread {s[k]}); "
                             "curve may imply a negative hazard (allow_negative=True to see it)")
        haz.append(brentq(pv, lo, 5.0, xtol=1e-15, maxiter=500))
    haz = np.array(haz)
    surv = np.asarray(survival_prob(T, T, haz), float)
    rows = [cds_legs(haz, T, t, recovery, discount_curve_fn, **kw) for t in T]
    return pd.DataFrame({"tenor": T, "spread": s, "hazard": haz, "survival": surv,
                         "cum_pd": 1.0 - surv, "rpv01": [x["rpv01"] for x in rows],
                         "repriced_spread": [x["par_spread"] for x in rows]})


# ---------------------------------------------------------------------------------------------
# Portfolio credit (ASRF / Vasicek / Basel IRB / one-factor MC)
# ---------------------------------------------------------------------------------------------

def vasicek_loss_quantile(pd, rho, level=0.999):
    """ASRF conditional PD at confidence `level` (bad factor M = -Phi^-1(level)):
        p_q = Phi((Phi^-1(PD) + sqrt(rho) Phi^-1(level)) / sqrt(1 - rho)).
    Multiply by LGD (and EAD) for the loss quantile; subtract PD*LGD for unexpected loss.
    p_q > PD for level > 0.5."""
    p = np.asarray(pd, float)
    rho = np.asarray(rho, float)
    return _out(norm.cdf((norm.ppf(p) + np.sqrt(rho) * norm.ppf(level)) / np.sqrt(1.0 - rho)))


def vasicek_loss_distribution(x, pd, rho, *, lgd=1.0, kind="cdf"):
    """Vasicek large homogeneous pool: distribution of the portfolio LOSS rate x (fraction of
    exposure). Default rate = x / lgd (x in [0, lgd]).
      kind="cdf": P(L <= x) = Phi((sqrt(1-rho) Phi^-1(x/lgd) - Phi^-1(PD)) / sqrt(rho))
      kind="pdf": density of L at x (per unit of loss rate)
      kind="quantile": x is the confidence level; returns lgd * vasicek_loss_quantile(pd, rho, x)
    """
    if kind == "quantile":
        return _out(lgd * np.asarray(vasicek_loss_quantile(pd, rho, x)))
    z = np.clip(np.asarray(x, float) / lgd, 0.0, 1.0)
    a = norm.ppf(z)
    k = norm.ppf(pd)
    if kind == "cdf":
        return _out(norm.cdf((math.sqrt(1 - rho) * a - k) / math.sqrt(rho)))
    if kind == "pdf":
        with np.errstate(over="ignore", invalid="ignore"):
            dens = math.sqrt((1 - rho) / rho) * np.exp(0.5 * a ** 2 - (math.sqrt(1 - rho) * a - k) ** 2
                                                      / (2 * rho)) / lgd
        return _out(np.nan_to_num(dens, nan=0.0, posinf=0.0))
    raise ValueError("kind must be 'cdf', 'pdf' or 'quantile'")


def basel_irb_capital(pd, lgd=0.45, rho=None, maturity=2.5, asset_class="corporate", sales=None, *,
                      ead=1.0, confidence=0.999, pd_floor=0.0003, maturity_adjustment=None,
                      clip_maturity=True) -> dict:
    """Basel II IRB capital (BCBS 128 paras 272-273, 328-330).

    asset_class:
      "corporate" (also sovereign/bank): R = 0.12 w + 0.24 (1-w), w = (1-e^{-50 PD})/(1-e^{-50});
                  SME firm-size adjustment R -= 0.04 (1 - (S-5)/45) when sales (EUR mn) < 50,
                  S clipped to [5, 50]
      "financial": large regulated / unregulated financial (Basel III): corporate R x 1.25
      "mortgage" R = 0.15; "qrre" R = 0.04;
      "other_retail": R = 0.03 w + 0.16 (1-w), w = (1-e^{-35 PD})/(1-e^{-35})
    rho given -> overrides the formula. Maturity adjustment (non-retail only, unless
    maturity_adjustment=True/False forces it): b = (0.11852 - 0.05478 ln PD)^2,
    MA = (1 + (M - 2.5) b)/(1 - 1.5 b), M clipped to [1, 5] if clip_maturity.
    K = LGD [Phi((Phi^-1(PD) + sqrt(R) Phi^-1(confidence))/sqrt(1-R)) - PD] MA.
    PD floored at pd_floor (0.03% Basel II; pass 0.0005 for Basel III 2017, 0 to disable).
    PD >= 1 (defaulted) -> K = 0 (F-IRB convention).
    Returns dict{K, risk_weight (= 12.5 K), rwa (= 12.5 K EAD), capital (= K EAD = 8% RWA),
    R, b, maturity_adjustment, conditional_pd, expected_loss (= PD LGD EAD), pd_used}.
    """
    p = np.maximum(np.asarray(pd, float), pd_floor)
    lgd_a = np.asarray(lgd, float)
    cls = asset_class.lower()
    retail = cls in ("mortgage", "qrre", "other_retail", "retail")
    if rho is not None:
        R = np.asarray(rho, float) * np.ones_like(p)
    elif cls in ("corporate", "sovereign", "bank", "sme", "financial"):
        w = (1 - np.exp(-50 * p)) / (1 - math.exp(-50))
        R = 0.12 * w + 0.24 * (1 - w)
        if sales is not None:
            S = np.clip(np.asarray(sales, float), 5.0, 50.0)
            R = R - 0.04 * (1 - (S - 5.0) / 45.0)
        if cls == "financial":
            R = R * 1.25
    elif cls == "mortgage":
        R = np.full_like(p, 0.15)
    elif cls == "qrre":
        R = np.full_like(p, 0.04)
    elif cls in ("other_retail", "retail"):
        w = (1 - np.exp(-35 * p)) / (1 - math.exp(-35))
        R = 0.03 * w + 0.16 * (1 - w)
    else:
        raise ValueError("asset_class must be corporate|sovereign|bank|sme|financial|mortgage|qrre|"
                         "other_retail")
    use_ma = (not retail) if maturity_adjustment is None else bool(maturity_adjustment)
    pc = np.minimum(p, 1 - 1e-15)
    b = (0.11852 - 0.05478 * np.log(pc)) ** 2
    M = np.clip(maturity, 1.0, 5.0) if clip_maturity else np.asarray(maturity, float)
    ma = (1 + (M - 2.5) * b) / (1 - 1.5 * b) if use_ma else np.ones_like(p)
    cond = norm.cdf((norm.ppf(pc) + np.sqrt(R) * norm.ppf(confidence)) / np.sqrt(1 - R))
    K = lgd_a * (cond - pc) * ma
    K = np.where(p >= 1.0, 0.0, K)
    ead_a = np.asarray(ead, float)
    return {"K": _out(K), "risk_weight": _out(12.5 * K), "rwa": _out(12.5 * K * ead_a),
            "capital": _out(K * ead_a), "R": _out(R), "b": _out(b), "maturity_adjustment": _out(ma),
            "conditional_pd": _out(cond), "expected_loss": _out(p * lgd_a * ead_a),
            "pd_used": _out(p)}


def one_factor_mc_losses(pds, eads, lgds, rho, n_sims=100_000, seed=0, *, return_defaults=False):
    """Gaussian one-factor default-mode Monte Carlo.

    A_i = sqrt(rho_i) M + sqrt(1-rho_i) eps_i; obligor i defaults iff A_i < Phi^-1(PD_i).
    RNG: np.random.default_rng(seed); draws M (n_sims,) FIRST, then eps (n_sims, n_obligors),
    C order. rho scalar or per obligor. Loss_s = sum_i EAD_i LGD_i 1{default}.
    Returns losses array (n_sims,), or (losses, default_indicator bool array) if return_defaults.
    VaR/ES from the losses: qf_toolkit.var_es_historical(-losses, level).
    """
    p = np.atleast_1d(np.asarray(pds, float))
    n = p.size
    e = np.broadcast_to(np.asarray(eads, float), (n,))
    lg = np.broadcast_to(np.asarray(lgds, float), (n,))
    r = np.broadcast_to(np.asarray(rho, float), (n,))
    rng = np.random.default_rng(seed)
    M = rng.standard_normal(n_sims)
    eps = rng.standard_normal((n_sims, n))
    A = np.sqrt(r) * M[:, None] + np.sqrt(1 - r) * eps
    with np.errstate(divide="ignore"):
        thr = norm.ppf(p)
    dflt = A < thr
    losses = dflt.astype(float) @ (e * lg)
    return (losses, dflt) if return_defaults else losses


# ---------------------------------------------------------------------------------------------
# Rating migration
# ---------------------------------------------------------------------------------------------

_SCALE = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D", "SD", "DEFAULT"]


def _as_matrix(P, percent):
    labels = None
    try:
        import pandas as pd
        if isinstance(P, pd.DataFrame):
            labels = (list(P.index), list(P.columns))
    except ImportError:  # pragma: no cover
        pass
    A = np.asarray(P, float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("transition matrix must be square (rows FROM, cols TO)")
    if percent is True or (percent == "auto" and np.allclose(A.sum(axis=1), 100.0, atol=1.0)):
        A = A / 100.0
    return A, labels


def _wrap(A, labels):
    if labels is None:
        return A
    import pandas as pd
    return pd.DataFrame(A, index=labels[0], columns=labels[1])


def generator_from_matrix(P, *, fix="DA", percent="auto"):
    """Generator Q with expm(Q) ~ P for a one-period matrix P (rows FROM, cols TO).

    Q = real(logm(P)). logm can give negative off-diagonal rates (invalid generator); fix:
      "DA" (diagonal adjustment, default; Israel-Rosenthal-Wei 2001): set negative off-diagonals
           to 0, then q_ii = -sum_{j!=i} q_ij;
      "WA" (weighted adjustment): zero negative off-diagonals and remove the row excess B_i from
           the remaining entries (diagonal included) in proportion to |q_ij|;
      None: raw real(logm(P)).
    Rows of the result sum to 0; an absorbing default row stays 0. expm(Q) is close to, not
    exactly, P after a fix. percent="auto" divides by 100 if rows sum to ~100.
    DataFrame in -> DataFrame out with the same labels.
    """
    A, labels = _as_matrix(P, percent)
    Q = np.real(logm(A))
    n = Q.shape[0]
    off = ~np.eye(n, dtype=bool)
    if fix is None:
        pass
    elif str(fix).upper() == "DA":
        Q = np.where(off & (Q < 0), 0.0, Q)
        np.fill_diagonal(Q, 0.0)
        np.fill_diagonal(Q, -Q.sum(axis=1))
    elif str(fix).upper() == "WA":
        for i in range(n):
            row = Q[i].copy()
            neg = off[i] & (row < 0)
            B = -row[neg].sum()
            keep = ~neg
            G = np.abs(row[keep]).sum()
            row[neg] = 0.0
            if G > 0:
                row[keep] = row[keep] - B * np.abs(row[keep]) / G
            Q[i] = row
            # clean residual rounding so the row sums to exactly 0
            Q[i, i] = -np.delete(Q[i], i).sum()
    else:
        raise ValueError("fix must be 'DA', 'WA' or None")
    Q[np.abs(Q) < 1e-15] = 0.0
    return _wrap(Q, labels)


def matrix_power_migration(P, years, *, percent="auto", fractional_fix="DA"):
    """h-period transition matrix under a time-homogeneous Markov chain.

    Integer years -> P^years (np.linalg.matrix_power). Non-integer years ->
    expm(years * generator_from_matrix(P, fix=fractional_fix)). Rows FROM, cols TO; rows sum to 1.
    DataFrame in -> DataFrame out.
    """
    A, labels = _as_matrix(P, percent)
    y = float(years)
    if abs(y - round(y)) < 1e-12:
        out = np.linalg.matrix_power(A, int(round(y)))
    else:
        Q = np.asarray(generator_from_matrix(A, fix=fractional_fix, percent=False))
        out = expm(y * Q)
    return _wrap(out, labels)


def cohort_matrix(transitions, *, states=None, from_col="from", to_col="to", weight_col=None,
                  censored=("NR", "WR", "WD", "N.R.", "WITHDRAWN"), empty_row="identity",
                  return_counts=False):
    """Cohort (discrete-time) estimator P_ij = N_ij / N_i.

    transitions: DataFrame with columns from_col, to_col (optional weight_col = count per row),
      or a list of (from, to) pairs, or a square count matrix (array / DataFrame, rows FROM).
    censored: destination (and origin) labels dropped from numerator AND denominator
      (NR-adjusted, S&P convention). Pass censored=() to keep NR as a state.
    states: output order; default = rating-scale order (AAA..CCC, D) when all labels are known
      rating letters, else sorted. Pass it explicitly when the scale is anything else.
    empty_row: "identity" (state with no issuers stays put; default state becomes absorbing)
      or "nan". Returns DataFrame P (or (P, counts) if return_counts).
    Pooled multi-year: concatenate all years' transitions (issuer-weighted), do not average P's.
    """
    import pandas as pd

    if isinstance(transitions, pd.DataFrame) and from_col in transitions.columns \
            and to_col in transitions.columns:
        df = transitions[[from_col, to_col]].copy()
        df.columns = ["f", "t"]
        df["w"] = transitions[weight_col].to_numpy(float) if weight_col else 1.0
    else:
        arr = transitions
        if isinstance(arr, pd.DataFrame) or (np.ndim(arr) == 2 and np.shape(arr)[0] == np.shape(arr)[1]
                                             and np.issubdtype(np.asarray(arr).dtype, np.number)):
            if isinstance(arr, pd.DataFrame):
                C = arr.astype(float)
            else:
                C = pd.DataFrame(np.asarray(arr, float))
            C = C.loc[[i for i in C.index if str(i).strip().upper() not in {c.upper() for c in censored}],
                      [j for j in C.columns if str(j).strip().upper() not in {c.upper() for c in censored}]]
            df = C.stack().reset_index()
            df.columns = ["f", "t", "w"]
        else:
            df = pd.DataFrame(list(arr), columns=["f", "t"])
            df["w"] = 1.0
    df["f"] = df["f"].astype(str).str.strip()
    df["t"] = df["t"].astype(str).str.strip()
    cen = {c.upper() for c in censored}
    df = df[~df["f"].str.upper().isin(cen) & ~df["t"].str.upper().isin(cen)]
    if states is None:
        labs = pd.unique(pd.concat([df["f"], df["t"]]))
        if all(str(x).upper() in _SCALE for x in labs):
            states = sorted(labs, key=lambda x: _SCALE.index(str(x).upper()))
        else:
            states = sorted(labs)
    states = [str(s) for s in states]
    N = df.pivot_table(index="f", columns="t", values="w", aggfunc="sum", fill_value=0.0)
    N = N.reindex(index=states, columns=states, fill_value=0.0).astype(float)
    tot = N.sum(axis=1)
    P = N.div(tot.replace(0.0, np.nan), axis=0)
    for s in states:
        if tot[s] == 0:
            if empty_row == "identity":
                P.loc[s] = 0.0
                P.loc[s, s] = 1.0
            elif empty_row != "nan":
                raise ValueError("empty_row must be 'identity' or 'nan'")
    return (P, N) if return_counts else P


def cumulative_default_probs(P, horizons, *, default_state=-1, percent="auto", fractional_fix="DA"):
    """Cumulative PD_i(h) = [P^h]_{i, D} for each origin i and horizon h (years).

    default_state: column index or label of the default state (default: last column).
    Returns pandas DataFrame, index = origin states, columns = horizons.
    Marginal PD = diff along columns; forward PD = (PD(h)-PD(h-1))/(1-PD(h-1));
    average annual PD = 1 - (1 - PD(h))**(1/h).
    """
    import pandas as pd

    A, labels = _as_matrix(P, percent)
    idx = labels[0] if labels else list(range(A.shape[0]))
    cols = labels[1] if labels else list(range(A.shape[1]))
    j = cols.index(default_state) if (labels and default_state in cols) else int(default_state)
    hs = list(np.atleast_1d(horizons))
    data = {h: np.asarray(matrix_power_migration(A, h, percent=False,
                                                 fractional_fix=fractional_fix))[:, j] for h in hs}
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------------------------
# Short-rate models
# ---------------------------------------------------------------------------------------------

def _vas_AB(kappa, theta, sigma, tau):
    tau = np.asarray(tau, float)
    B = (1 - np.exp(-kappa * tau)) / kappa
    lnA = (B - tau) * (kappa ** 2 * theta - 0.5 * sigma ** 2) / kappa ** 2 - sigma ** 2 * B ** 2 / (4 * kappa)
    return lnA, B


def vasicek_zcb(r0, kappa, theta, sigma, T):
    """Vasicek zero-coupon bond P(0,T) = A(T) exp(-B(T) r0), dr = kappa(theta - r)dt + sigma dW
    (risk-neutral parameters). B = (1-e^{-kappa T})/kappa,
    ln A = (B - T)(kappa^2 theta - sigma^2/2)/kappa^2 - sigma^2 B^2/(4 kappa).
    T = time to maturity (also P(t, t+T) given r_t = r0). Vectorised in T."""
    lnA, B = _vas_AB(kappa, theta, sigma, T)
    return _out(np.exp(lnA - B * r0))


def cir_zcb(r0, kappa, theta, sigma, T):
    """CIR zero-coupon bond, dr = kappa(theta - r)dt + sigma sqrt(r) dW. g = sqrt(kappa^2+2 sigma^2),
    den = (g+kappa)(e^{gT}-1) + 2g, B = 2(e^{gT}-1)/den, A = [2g e^{(kappa+g)T/2}/den]^{2 kappa theta/sigma^2}.
    Feller condition 2 kappa theta >= sigma^2 is not required for the price."""
    T = np.asarray(T, float)
    g = math.sqrt(kappa ** 2 + 2 * sigma ** 2)
    eg = np.expm1(g * T)
    den = (g + kappa) * eg + 2 * g
    B = 2 * eg / den
    lnA = (2 * kappa * theta / sigma ** 2) * (np.log(2 * g) + 0.5 * (kappa + g) * T - np.log(den))
    return _out(np.exp(lnA - B * r0))


def _inst_fwd(curve_fn, t, h):
    if t < h:
        return -(math.log(curve_fn(t + h)) - math.log(curve_fn(t) if t > 0 else 1.0)) / h
    return -(math.log(curve_fn(t + h)) - math.log(curve_fn(t - h))) / (2 * h)


def hull_white_zcb(a, sigma, T, curve_fn, *, t=0.0, r_t=None, fwd_fn=None, h=1e-5):
    """Hull-White P(t,T) given the short rate r_t, fitted to today's curve curve_fn(s) = P(0,s).

    dr = (theta(t) - a r)dt + sigma dW. B = (1-e^{-a(T-t)})/a;
    ln A = ln(P(0,T)/P(0,t)) + B f(0,t) - sigma^2 (1-e^{-2at}) B^2/(4a).
    f(0,t) from fwd_fn if given, else numerical -d ln P(0,t)/dt (step h).
    t=0 and r_t=None -> r_0 = f(0,0), which returns P(0,T) exactly (the curve is reproduced)."""
    t = float(t)
    f0t = float(fwd_fn(t)) if fwd_fn is not None else _inst_fwd(curve_fn, t, h)
    if r_t is None:
        if t != 0.0:
            raise ValueError("r_t required for t > 0")
        r_t = f0t
    T = np.asarray(T, float)
    B = (1 - np.exp(-a * (T - t))) / a
    P0t = curve_fn(t) if t > 0 else 1.0
    P0T = np.vectorize(lambda x: float(curve_fn(x)))(T)
    lnA = np.log(P0T / P0t) + B * f0t - sigma ** 2 * (1 - math.exp(-2 * a * t)) * B ** 2 / (4 * a)
    return _out(np.exp(lnA - B * r_t))


def _zcb_option(P0T, P0S, sp, K, kind):
    h = math.log(P0S / (P0T * K)) / sp + 0.5 * sp
    if kind == "call":
        return P0S * norm.cdf(h) - K * P0T * norm.cdf(h - sp)
    if kind == "put":
        return K * P0T * norm.cdf(-h + sp) - P0S * norm.cdf(-h)
    raise ValueError("kind must be 'call' or 'put'")


def vasicek_zcb_option(r0, kappa, theta, sigma, T, S, K, kind="call"):
    """Jamshidian (1989) closed form: European option expiring T on a zero-coupon bond maturing
    S > T, strike K per unit face. sp = sigma sqrt((1-e^{-2 kappa T})/(2 kappa)) B(S-T),
    h = ln(P(0,S)/(P(0,T) K))/sp + sp/2; call = P(0,S)N(h) - K P(0,T)N(h-sp);
    put = K P(0,T)N(-h+sp) - P(0,S)N(-h). Multiply by face for notional."""
    P0T = vasicek_zcb(r0, kappa, theta, sigma, T)
    P0S = vasicek_zcb(r0, kappa, theta, sigma, S)
    B = (1 - math.exp(-kappa * (S - T))) / kappa
    sp = sigma * math.sqrt((1 - math.exp(-2 * kappa * T)) / (2 * kappa)) * B
    return float(_zcb_option(P0T, P0S, sp, K, kind))


def hull_white_zcb_option(a, sigma, T, S, K, curve_fn, kind="call"):
    """Hull-White ZCB option (expiry T, bond maturity S), same formula as vasicek_zcb_option with
    P(0,.) = curve_fn. Caplet (T -> S, accrual d, strike Kc) = (1 + Kc d) * put(K = 1/(1 + Kc d))."""
    B = (1 - math.exp(-a * (S - T))) / a
    sp = sigma * math.sqrt((1 - math.exp(-2 * a * T)) / (2 * a)) * B
    return float(_zcb_option(float(curve_fn(T)), float(curve_fn(S)), sp, K, kind))


def vasicek_swaption(r0, kappa, theta, sigma, T0, pay_times, K, *, accruals=None, kind="payer",
                     notional=1.0):
    """European swaption in Vasicek by Jamshidian decomposition. Option expiry = swap start T0;
    fixed payments at pay_times (> T0) with accruals (default: consecutive differences starting
    from T0); fixed rate K. Coupon bond c_i = K d_i, c_n += 1; find r* with
    sum c_i P(T0,T_i; r*) = 1; payer = sum c_i ZBP(T0, T_i, X_i), receiver uses ZBC."""
    Ti = np.asarray(pay_times, float)
    d = np.diff(np.concatenate([[T0], Ti])) if accruals is None else np.asarray(accruals, float)
    c = K * d
    c[-1] += 1.0
    f = lambda r: float(np.sum(c * np.asarray(vasicek_zcb(r, kappa, theta, sigma, Ti - T0)))) - 1.0  # noqa: E731
    lo, hi = -1.0, 1.0
    while f(lo) * f(hi) > 0 and hi < 1e3:
        lo, hi = lo * 2, hi * 2
    r_star = brentq(f, lo, hi, xtol=1e-14)
    X = np.asarray(vasicek_zcb(r_star, kappa, theta, sigma, Ti - T0))
    opt = "put" if kind == "payer" else "call" if kind == "receiver" else None
    if opt is None:
        raise ValueError("kind must be 'payer' or 'receiver'")
    return notional * float(sum(ci * vasicek_zcb_option(r0, kappa, theta, sigma, T0, t, x, opt)
                                for ci, t, x in zip(c, Ti, X)))


# ---------------------------------------------------------------------------------------------
# Black / Bachelier rate options
# ---------------------------------------------------------------------------------------------

def black_caplet(F, K, T, sigma, df, accrual, notional=1.0, *, kind="caplet"):
    """Black-76 caplet/floorlet: notional * accrual * df * [F N(d1) - K N(d2)] (floorlet:
    K N(-d2) - F N(-d1)). T = FIXING time (accrual start) in d1; df = DF to the PAYMENT date;
    F = simple forward (DF(T1)/DF(T2) - 1)/accrual; sigma lognormal decimal.
    T <= 0 -> intrinsic. kind: 'caplet' | 'floorlet'."""
    k = {"caplet": "call", "floorlet": "put", "cap": "call", "floor": "put"}.get(kind)
    if k is None:
        raise ValueError("kind must be 'caplet' or 'floorlet'")
    if np.all(np.asarray(T) <= 0):
        undisc = np.maximum(np.asarray(F) - K, 0.0) if k == "call" else np.maximum(K - np.asarray(F), 0.0)
    else:
        undisc = bs_price(F, K, T, 0.0, sigma, k)
    return _out(notional * accrual * np.asarray(df) * undisc)


def black_swaption(F_swap, K, T, sigma, annuity, kind="payer", notional=1.0):
    """Black swaption: payer = notional * annuity [S N(d1) - K N(d2)], receiver = annuity
    [K N(-d2) - S N(-d1)]; annuity = sum tau_i DF(T_i) over fixed-leg dates (see
    swap_rate_and_annuity); T = option expiry; payer - receiver = annuity (S - K)."""
    k = {"payer": "call", "receiver": "put"}.get(kind)
    if k is None:
        raise ValueError("kind must be 'payer' or 'receiver'")
    return _out(notional * np.asarray(annuity) * bs_price(F_swap, K, T, 0.0, sigma, k))


def bachelier_price(F, K, T, sigma_n, kind="call", df=1.0):
    """Bachelier (normal) option on a forward: df * [(F-K)N(x) + sigma_n sqrt(T) n(x)] for a call,
    df * [(K-F)N(-x) + sigma_n sqrt(T) n(x)] for a put, x = (F-K)/(sigma_n sqrt T).
    sigma_n in forward units per sqrt(year) (85 bp -> 0.0085). Works for negative rates."""
    F, K, T = np.asarray(F, float), np.asarray(K, float), np.asarray(T, float)
    sv = sigma_n * np.sqrt(T)
    x = (F - K) / sv
    if kind == "call":
        v = (F - K) * norm.cdf(x) + sv * norm.pdf(x)
    elif kind == "put":
        v = (K - F) * norm.cdf(-x) + sv * norm.pdf(x)
    else:
        raise ValueError("kind must be 'call' or 'put'")
    return _out(df * v)


def bachelier_caplet(F, K, T, sigma_n, df, accrual, notional=1.0, *, kind="caplet"):
    """Normal-vol caplet/floorlet: notional * accrual * bachelier_price(F, K, T_fix, sigma_n) * df."""
    k = {"caplet": "call", "floorlet": "put"}.get(kind)
    if k is None:
        raise ValueError("kind must be 'caplet' or 'floorlet'")
    return _out(notional * accrual * np.asarray(bachelier_price(F, K, T, sigma_n, k, df)))


def bachelier_swaption(F_swap, K, T, sigma_n, annuity, kind="payer", notional=1.0):
    """Normal-vol swaption: notional * annuity * bachelier_price(S, K, T, sigma_n)."""
    k = {"payer": "call", "receiver": "put"}.get(kind)
    if k is None:
        raise ValueError("kind must be 'payer' or 'receiver'")
    return _out(notional * np.asarray(annuity) * np.asarray(bachelier_price(F_swap, K, T, sigma_n, k)))


def swap_rate_and_annuity(dfs, times, freq=None, *, df_start=1.0, t_start=0.0) -> dict:
    """Par swap rate and annuity from discount factors at the FIXED-leg payment times.

    Accruals tau_i = 1/freq if freq is given, else times[i] - times[i-1] with times[-1]:=t_start.
    annuity = sum tau_i DF(T_i); swap_rate = (df_start - DF(T_n)) / annuity.
    Spot-starting swap: df_start=1, t_start=0. Forward swap starting T0 (swaption underlying):
    df_start = DF(T0), t_start = T0, times = payment dates after T0.
    Returns dict{swap_rate, annuity, accruals}."""
    D = np.asarray(dfs, float)
    t = np.asarray(times, float)
    tau = np.full(t.shape, 1.0 / freq) if freq else np.diff(np.concatenate([[t_start], t]))
    A = float(np.sum(tau * D))
    return {"swap_rate": (float(df_start) - float(D[-1])) / A, "annuity": A, "accruals": tau}
