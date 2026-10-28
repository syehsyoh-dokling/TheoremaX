"""know77 group-4 helpers: task-PATTERN functions for the steps that failed or swung in the know76
full run - tolerant FX convention / snapshot-date readers, money-market day-count defaults, a generic
sequential discount-curve bootstrap (deposits, FRAs with adjustments, par OIS swaps; log-linear DF),
stationary moments of an OU-with-jumps process and a log-level (exponential OU) jump-diffusion study,
a multi-factor copula credit-portfolio risk report, a CreditMetrics migration report with flat keys,
VaR-backtest coverage tests with chi-square survival p-values, and an EMA-crossover CTA + Basel IMA
capital report.

Every function is general (no task names), pure numpy/pandas/scipy (arch for GARCH), states its
formula and conventions in the docstring, and calls the existing tested helpers underneath. Numbers
a task fixes (seeds, windows, levels, horizons) are REQUIRED arguments. Self-tests:
proven_work/fail72/selftest77_4.py.
"""
from __future__ import annotations

# know82 _rd82: report results resolve short / nested keys
def _rd82(d):
    from ._rdict import ResultDict
    return ResultDict({k: (ResultDict(v) if isinstance(v, dict) and not isinstance(v, ResultDict) else v) for k, v in d.items()})

import math
import re

import numpy as np
import pandas as pd
from scipy import optimize, stats

# =============================================================================================
# 1. FX conventions: pair info, snapshot dates, money-market day counts
# =============================================================================================

_T1_PAIRS = {"USDCAD", "CADUSD", "USDTRY", "TRYUSD", "USDRUB", "RUBUSD", "USDPHP", "PHPUSD"}
_ACT365_CCYS = {"GBP", "AUD", "NZD", "CAD", "HKD", "SGD", "ZAR", "INR", "MYR", "THB", "PLN", "TWD", "KRW", "ILS"}


def _norm_pair(p) -> str:
    return re.sub(r"[^A-Z]", "", str(p).upper())


def fx_pair_info(pair_conventions, pair) -> dict:
    """-> dict{pair, base, terms, spot_lag, point_multiplier} from a conventions dict, nested ('pairs') or flat, any key aliases.

    Pair lookup ignores '/', '_', case ('EUR/USD' finds 'EURUSD'). Key aliases: base | base_currency |
    base_ccy | ccy1; terms | terms_currency | quote | quote_currency | quote_ccy | ccy2; spot lag |
    spot_lag_business_days | spot_lag | spot_lag_days | spot_days | lag; point_multiplier |
    pip_multiplier | points_multiplier | pip_factor | multiplier. Missing fields take the textbook
    market defaults: base/terms from the pair letters, spot lag T+1 for USD/CAD, USD/TRY, USD/RUB,
    USD/PHP and T+2 otherwise, point multiplier 100 for JPY pairs and 10000 otherwise."""
    conv = pair_conventions or {}
    if isinstance(conv, dict) and isinstance(conv.get("pairs"), (dict, list)):
        conv = conv["pairs"]
    if isinstance(conv, list):
        conv = {str(r.get("pair", r.get("symbol", ""))): r for r in conv if isinstance(r, dict)}
    key = _norm_pair(pair)
    rec = {}
    for k, v in conv.items():
        if _norm_pair(k) == key and isinstance(v, dict):
            rec = {str(a).lower(): b for a, b in v.items()}
            break

    def pick(names, default):
        for n in names:
            if n in rec and rec[n] is not None:
                return rec[n]
        return default

    base = str(pick(("base", "base_currency", "base_ccy", "ccy1"), key[:3])).upper()
    terms = str(pick(("terms", "terms_currency", "quote", "quote_currency", "quote_ccy", "ccy2"), key[3:6])).upper()
    lag = int(pick(("spot_lag_business_days", "spot_lag", "spot_lag_days", "spot_days", "lag"),
                   1 if key in _T1_PAIRS else 2))
    mult = float(pick(("point_multiplier", "pip_multiplier", "points_multiplier", "pip_factor", "multiplier"),
                      100.0 if "JPY" in key else 10000.0))
    return {"pair": key, "base": base, "terms": terms, "spot_lag": lag, "point_multiplier": mult}


def first_iso_date(*sources) -> str:
    """-> 'YYYY-MM-DD': the first ISO date found in dicts (keys snapshot_date / trade_date / valuation_date / as_of / date first) or strings.

    Free-text fields such as a '_comment' "Snapshot date: 2023-09-01. Bid/offer ..." are searched with
    a regex instead of being parsed whole (pd.to_datetime on the whole comment raises DateParseError).
    Raises ValueError when no source holds a date."""
    pat = re.compile(r"(\d{4})-(\d{2})-(\d{2})")

    def scan(s):
        m = pat.search(str(s))
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None

    for src in sources:
        if isinstance(src, dict):
            for k in ("snapshot_date", "trade_date", "valuation_date", "as_of", "as_of_date", "date"):
                if k in src and scan(src[k]):
                    return scan(src[k])
            for v in src.values():
                if isinstance(v, str) and scan(v):
                    return scan(v)
        elif src is not None and scan(src):
            return scan(src)
    raise ValueError("no YYYY-MM-DD date in the given sources")


def money_market_basis(ccy, day_counts=None) -> float:
    """-> 360.0 or 365.0: money-market year basis of a currency (a given day_counts map wins).

    Textbook market convention when nothing is given: ACT/365 for GBP, AUD, NZD, CAD, HKD, SGD, ZAR,
    INR, MYR, THB, PLN, TWD, KRW, ILS; ACT/360 for USD, EUR, JPY, CHF, SEK, NOK, DKK and the rest.
    day_counts values like 'ACT/360', 'ACT/365F', 'A365', 360 are read by their denominator."""
    c = str(ccy).upper()
    if day_counts:
        dc = {str(k).upper(): v for k, v in dict(day_counts).items()}
        if c in dc and dc[c] is not None:
            v = dc[c]
            if isinstance(v, (int, float)):
                return float(v)
            m = re.search(r"(360|365|366)", str(v))
            if m:
                return float(m.group(1))
    return 365.0 if c in _ACT365_CCYS else 360.0


# =============================================================================================
# 2. Sequential discount-curve bootstrap (deposits, FRAs, par OIS swaps), log-linear DFs
# =============================================================================================

def _basis_of(day_count) -> float:
    if isinstance(day_count, (int, float)):
        return float(day_count)
    m = re.search(r"(360|365|366)", str(day_count))
    if not m:
        raise ValueError(f"day count {day_count!r} has no 360/365 denominator")
    return float(m.group(1))


def _instrument_rows(instruments):
    if isinstance(instruments, pd.DataFrame):
        recs = instruments.to_dict("records")
    else:
        recs = [dict(r) for r in instruments]
    out = []
    for i, r in enumerate(recs):
        lo = {str(k).lower(): v for k, v in r.items()}
        typ = str(lo.get("instrument_type", lo.get("type", "deposit"))).lower()
        rate = lo.get("rate", lo.get("normalized_quote", lo.get("quote")))
        adj = lo.get("adj_bps", lo.get("quote_adjustment_bps", 0.0))
        adj = 0.0 if adj is None or (isinstance(adj, float) and math.isnan(adj)) else float(adj)
        out.append({"type": typ, "start": pd.Timestamp(lo["start_date"]).normalize(),
                    "end": pd.Timestamp(lo["end_date"]).normalize(), "rate": float(rate), "adj_bps": adj,
                    "tenor": lo.get("tenor"), "order": i})
    return out


def bootstrap_df_curve(anchor, instruments, day_count="ACT/360", pay_months=12, extrapolate="flat_forward"):
    """-> dict{nodes DataFrame[tenor, pillar_date, days, df], df(date), table(curve_type)}: sequential DF bootstrap of deposits / FRAs / par OIS swaps, linear in -ln DF.

    anchor: curve date with DF = 1 (the valuation date). instruments: DataFrame or records with
    instrument_type ('deposit' | 'fra' | 'swap'), start_date, end_date, rate (DECIMAL; or
    normalized_quote) and optional adj_bps / quote_adjustment_bps (added to an FRA rate, in bp).
    Accrual tau(a, b) = days(a, b) / basis(day_count). Instruments are solved one at a time in
    end-date order; each adds the node -ln DF(end) chosen (brentq) so that it reprices exactly given
    the nodes already known (interpolation is LINEAR IN -ln DF ON CALENDAR DAYS from the anchor node
    (0, 0), i.e. piecewise-flat forwards; beyond the last node the last forward is continued):
      deposit / fra:  DF(start) / DF(end) = 1 + (rate + adj_bps 1e-4) tau(start, end)
      swap (par OIS, fixed annuity vs compounded overnight = DF(start) - DF(end)):
          rate sum_i tau_i DF(d_i) = DF(start) - DF(end), fixed dates rolled BACK from end every
          pay_months (front stub from start); a swap no longer than pay_months is one period.
    A start date after the anchor (spot-starting quotes) is handled through the same interpolation.
    table(curve_type) gives DataFrame[curve_type, tenor, pillar_date ('YYYY-MM-DD'), year_frac =
    days(anchor, pillar)/basis, discount_factor, zero_rate = -ln DF / year_frac (continuous),
    forward_rate = simple forward between consecutive pillars (DF_prev/DF - 1)/tau]."""
    a = pd.Timestamp(anchor).normalize()
    B = _basis_of(day_count)
    rows = sorted(_instrument_rows(instruments), key=lambda r: (r["end"], r["order"]))
    xs, gs = [0.0], [0.0]

    def g_at(x, X, G):
        X, G = np.asarray(X), np.asarray(G)
        if x <= X[-1] or len(X) < 2:
            return float(np.interp(x, X, G)) if len(X) > 1 else float(G[-1])
        slope = (G[-1] - G[-2]) / (X[-1] - X[-2]) if extrapolate == "flat_forward" else G[-1] / X[-1]
        return float(G[-1] + slope * (x - X[-1]))

    def days(d):
        return float((pd.Timestamp(d).normalize() - a).days)

    def tau(d1, d2):
        return (pd.Timestamp(d2) - pd.Timestamp(d1)).days / B

    labels = []
    for r in rows:
        xe = days(r["end"])
        if xe <= 0:
            continue
        rate = r["rate"] + r["adj_bps"] * 1e-4
        if r["type"] == "swap":
            dates = []
            d = r["end"]
            while d > r["start"]:
                dates.append(d)
                d = d - pd.DateOffset(months=int(pay_months))
            dates = sorted(dates)
            prev = [r["start"]] + dates[:-1]
        else:
            dates, prev = None, None

        def err(ge, r=r, xe=xe, rate=rate, dates=dates, prev=prev):
            X = xs + [xe]
            G = gs + [ge]
            o = np.argsort(X)
            X, G = list(np.asarray(X)[o]), list(np.asarray(G)[o])
            dfs = math.exp(-g_at(days(r["start"]), X, G))
            dfe = math.exp(-ge)
            if r["type"] in ("deposit", "fra", "depo", "cash"):
                return dfs / dfe - (1.0 + rate * tau(r["start"], r["end"]))
            ann = sum(tau(p, d) * math.exp(-g_at(days(d), X, G)) for p, d in zip(prev, dates))
            return rate * ann - (dfs - dfe)

        ge = optimize.brentq(err, -2.0, 10.0, xtol=1e-15, maxiter=500)
        if xe in xs:
            i = xs.index(xe)
            gs[i] = ge
        else:
            xs.append(xe)
            gs.append(ge)
            o = np.argsort(xs)
            xs[:] = list(np.asarray(xs)[o])
            gs[:] = list(np.asarray(gs)[o])
        labels.append((xe, r["tenor"], r["end"]))

    def df(d):
        x = days(d) if not isinstance(d, (int, float, np.floating)) else float(d)
        return math.exp(-g_at(x, xs, gs))

    lab = {}
    for xe, t, e in labels:
        lab[xe] = (t, e)
    nodes = pd.DataFrame({"tenor": [lab[x][0] for x in xs[1:]], "pillar_date": [lab[x][1] for x in xs[1:]],
                          "days": xs[1:], "df": [math.exp(-g) for g in gs[1:]]})

    def table(curve_type=""):
        out, prev_d, prev_df = [], a, 1.0
        for _, n in nodes.iterrows():
            yf = n["days"] / B
            dfv = float(n["df"])
            fwd = (prev_df / dfv - 1.0) / tau(prev_d, n["pillar_date"])
            out.append({"curve_type": curve_type, "tenor": n["tenor"],
                        "pillar_date": pd.Timestamp(n["pillar_date"]).strftime("%Y-%m-%d"), "year_frac": yf,
                        "discount_factor": dfv, "zero_rate": -math.log(dfv) / yf, "forward_rate": fwd})
            prev_d, prev_df = n["pillar_date"], dfv
        return pd.DataFrame(out)

    return {"nodes": nodes, "df": df, "table": table, "anchor": a, "basis": B}


# =============================================================================================
# 3. OU with jumps: stationary moments; log-level (exponential OU) jump-diffusion study
# =============================================================================================

def ou_jump_stationary(kappa, theta, sigma, lam=0.0, mu_J=0.0, sigma_J=0.0) -> dict:
    """-> dict{mean, var, std, mean_ou, var_ou}: STATIONARY (tau -> infinity) moments of dX = kappa(theta - X)dt + sigma dW + J dN.

    mean = theta + lam mu_J / kappa; var = (sigma^2 + lam (mu_J^2 + sigma_J^2)) / (2 kappa);
    mean_ou = theta, var_ou = sigma^2/(2 kappa) (no jumps). Never take stationary moments from
    ou_jump_moments(..., tau=0) - that is the starting point (variance 0)."""
    k = float(kappa)
    q2 = float(sigma) ** 2 + float(lam) * (float(mu_J) ** 2 + float(sigma_J) ** 2)
    v = q2 / (2.0 * k)
    return {"mean": float(theta) + float(lam) * float(mu_J) / k, "var": v, "std": math.sqrt(v),
            "mean_ou": float(theta), "var_ou": float(sigma) ** 2 / (2.0 * k)}


def log_ou_jump_report(levels, dt, taus, n_paths, seed, jump_threshold=3.0, level_scale=0.01, max_iter=5,
                       steps_per_year=None, half_life_cap=20.0) -> dict:
    """-> dict{calibration, conditional_moments DataFrame, forward_curve DataFrame, summary}: exponential-OU (log-level) jump-diffusion study.

    rate = levels * level_scale (percent -> decimal by default), NaN and non-positive values dropped;
    X = ln(rate); fit = fit_ou_jump(X, dt, jump_threshold, max_iter) (AR(1) on X, kappa = -ln b/dt,
    theta = a/(1-b), exact-discretisation sigma, residual jump filter); x0 = X[-1], S0 = e^x0.
    calibration: n_obs, n_changes, rate_mean/rate_std, log_rate_mean/log_rate_std (ddof=1), kappa,
    theta, sigma, lambda, mu_J, sigma_J, n_jumps, x0, S0, stationary_mean_X / stationary_var_X
    (ou_jump_stationary - NOT tau = 0). Per tau: ou_jump_moments -> E_X, Var_X (jump variance
    included), E_S = exp(E + V/2), Var_S = (e^V - 1) e^{2E + V}; MC: simulate_ou_jump(x0, ..., tau,
    round(tau steps_per_year), n_paths, seed) with steps_per_year = 1/dt by default, E_X_mc / Var_X_mc
    (ddof=1) of the last column. forward_rate_jd = exp(E_X + Var_X/2), forward_rate_ou the same with
    lam = 0. summary: kappa, theta, sigma, lambda_annual, half_life_years = ln2/kappa,
    stationary_vol_X = sqrt(stationary var), feller_ratio = 2 kappa theta / sigma^2,
    max_mc_mean_relerr = max |E - E_mc|/|E|, max_mc_var_relerr = max |V - V_mc|/V,
    jump_impact_on_forward = |F_jd - F_ou|/F_ou at tau = 1 (nearest tau), mean_reversion_confirmed =
    kappa > 0 and half-life < half_life_cap, jumps_detected = n_jumps > 0."""
    from .recipes import fit_ou_jump, simulate_ou_jump
    from .helpers75_3 import ou_jump_moments
    v = pd.to_numeric(pd.Series(np.asarray(levels).ravel()), errors="coerce").to_numpy(float)
    fin = v[np.isfinite(v)]
    if len(fin) > 0 and (fin <= 0).mean() > 0.5:
        X = fin.astype(float)
        rate = np.exp(X)
    else:
        if len(fin) > 0 and np.nanmax(fin) <= 1.0 and float(level_scale) == 0.01:
            rate = fin
        else:
            rate = fin * float(level_scale)
        rate = rate[rate > 0]
        X = np.log(rate)
    fit = fit_ou_jump(X, dt, jump_threshold=jump_threshold, max_iter=max_iter)
    kap, th, sg = float(fit["kappa"]), float(fit["theta"]), float(fit["sigma"])
    lam, mJ, sJ = float(fit["jump_intensity"]), float(fit["jump_mean"]), float(fit["jump_std"])
    if not np.isfinite(mJ):
        mJ = 0.0
    nj = int(fit["n_jumps"])
    x0 = float(X[-1])
    stn = ou_jump_stationary(kap, th, sg, lam, mJ, sJ)
    cal = {"n_obs": int(len(rate)), "n_changes": int(len(X) - 1), "rate_mean": float(rate.mean()),
           "rate_std": float(rate.std(ddof=1)), "log_rate_mean": float(X.mean()), "log_rate_std": float(X.std(ddof=1)),
           "kappa": kap, "theta": th, "sigma": sg, "lambda": lam, "mu_J": mJ, "sigma_J": sJ, "n_jumps": nj,
           "x0": x0, "S0": float(math.exp(x0)), "stationary_mean_X": stn["mean"], "stationary_var_X": stn["var"]}
    spy = (1.0 / float(dt)) if steps_per_year is None else float(steps_per_year)
    rows, fwd = [], []
    for tau in taus:
        tau = float(tau)
        m = ou_jump_moments(x0, kap, th, sg, tau, lam, mJ, sJ)
        m0 = ou_jump_moments(x0, kap, th, sg, tau, 0.0, 0.0, 0.0)
        E, V = float(m["mean"]), float(m["var"])
        paths = simulate_ou_jump(x0, kap, th, sg, lam, mJ, sJ, tau, int(round(tau * spy)), int(n_paths), seed=seed)
        last = paths[:, -1]
        rows.append({"tau": tau, "E_X": E, "Var_X": V, "E_S": math.exp(E + 0.5 * V),
                     "Var_S": (math.exp(V) - 1.0) * math.exp(2 * E + V),
                     "E_X_mc": float(last.mean()), "Var_X_mc": float(last.var(ddof=1))})
        fwd.append({"tau": tau, "forward_rate_jd": math.exp(E + 0.5 * V),
                    "forward_rate_ou": math.exp(float(m0["mean"]) + 0.5 * float(m0["var"]))})
    cm = pd.DataFrame(rows)
    fc = pd.DataFrame(fwd)
    i1 = int(np.argmin(np.abs(fc["tau"].to_numpy() - 1.0)))
    fj, fo = float(fc["forward_rate_jd"].iloc[i1]), float(fc["forward_rate_ou"].iloc[i1])
    hl = math.log(2) / kap
    summ = {"kappa": kap, "theta": th, "sigma": sg, "lambda_annual": lam, "half_life_years": hl,
            "stationary_vol_X": stn["std"], "feller_ratio": 2 * kap * th / sg ** 2,
            "max_mc_mean_relerr": float(np.max(np.abs(cm["E_X"] - cm["E_X_mc"]) / np.abs(cm["E_X"]))),
            "max_mc_var_relerr": float(np.max(np.abs(cm["Var_X"] - cm["Var_X_mc"]) / cm["Var_X"])),
            "jump_impact_on_forward": abs(fj - fo) / fo,
            "mean_reversion_confirmed": bool(kap > 0 and hl < half_life_cap), "jumps_detected": bool(nj > 0)}
    return _rd82({**cal, **summ, "calibration": cal, "conditional_moments": cm, "forward_curve": fc, "summary": summ, "fit": fit})


# =============================================================================================
# 4. Multi-factor copula credit portfolio risk report
# =============================================================================================

def copula_credit_portfolio_report(portfolio, sector_corr, config, *, t_seed_offset=1000, euler_level=0.99,
                                   quantile_method="linear", es_tail="geq", wrong_way_cap=0.999, cols=None,
                                   chunk_rows=10000) -> dict:
    """-> {var_cvar_results.csv, wrong_way_results.csv, marginal_contributions.csv, loss_statistics.csv, concentration_risk.json, summary.json, analytical_benchmark.csv, sector_risk_decomposition.csv, granularity_adjustment.csv}.

    Multi-factor Merton default model A_i = sqrt(rho_i) M_s(i) + sqrt(1-rho_i) e_i, M ~ N(0, sector
    correlation) by Cholesky (copula_default_mc, chunked). portfolio columns (rename via cols):
    obligor_id, issuer, pd_1y, exposure_usd, recovery_rate, recovery_stressed, factor_sector,
    basel_correlation; LGD = 1 - recovery. sector_corr: square matrix DataFrame (first column may be
    the row names); factor order = its column order. config keys: n_simulations, seed,
    confidence_levels, t_copula_df, wrong_way_sectors, wrong_way_correlation_boost.
    Gaussian copula with seed; Student-t copula (W = chi2(nu)/nu scales BOTH M and e, threshold
    t_nu^-1(PD)) with seed + t_seed_offset; wrong-way: rho + boost capped at wrong_way_cap and LGD =
    1 - recovery_stressed for the wrong-way sectors, Gaussian with the SAME seed as the base.
    VaR = np.quantile(L, level, quantile_method); CVaR = mean(L >= VaR) (es_tail 'geq') via loss_var_es.
    Euler: contribution_i = EAD_i LGD_i mean(D_i | L >= VaR_euler_level) (euler_es_contributions), sums
    to CVaR; pct_contribution = 100 c_i / CVaR. Loss statistics: mean, std (ddof=0), skewness and
    EXCESS kurtosis (scipy.stats, bias=True), max, pct_zero_loss = 100 mean(L == 0), analytical EL =
    sum PD EAD LGD. Concentration: hhi of obligor and sector exposures, top-5 by weight and by
    marginal CVaR, sector weights. Analytics per level: vasicek_portfolio_var, gordy granularity
    adjustment GA (gordy_granularity_adjustment), gordy_var = vasicek + GA. Sector decomposition per
    factor_sector present: n_obligors, total_exposure, exposure_weight, marginal_cvar_99, pct_of_total_cvar,
    risk_weight_ratio = pct_of_total_cvar / (100 exposure_weight). Summary keys as the usual report:
    n_obligors ... granularity_adjustment_99 (99 = level_key(euler_level))."""
    from .helpers75_2 import (copula_default_mc, loss_var_es, euler_es_contributions, vasicek_portfolio_var,
                              gordy_granularity_adjustment, hhi, level_key)
    c = {"id": "obligor_id", "issuer": "issuer", "pd": "pd_1y", "ead": "exposure_usd", "rec": "recovery_rate",
         "rec_s": "recovery_stressed", "sector": "factor_sector", "rho": "basel_correlation"}
    c.update(cols or {})
    pf = portfolio.copy()
    cm = sector_corr.copy()
    if not pd.api.types.is_numeric_dtype(cm.iloc[:, 0]):
        cm = cm.set_index(cm.columns[0])
    names = [str(x) for x in cm.columns]
    C = cm.loc[names, names].to_numpy(float) if set(names) <= set(map(str, cm.index)) else cm.to_numpy(float)
    C = 0.5 * (C + C.T)
    cfg = dict(config)
    n, seed = int(cfg["n_simulations"]), int(cfg["seed"])
    levels = [float(x) for x in cfg["confidence_levels"]]
    nu = float(cfg.get("t_copula_df", 5))
    ww = set(map(str, cfg.get("wrong_way_sectors", [])))
    boost = float(cfg.get("wrong_way_correlation_boost", 0.0))
    p = pf[c["pd"]].to_numpy(float)
    ead = pf[c["ead"]].to_numpy(float)
    lgd = 1.0 - pf[c["rec"]].to_numpy(float)
    rho = pf[c["rho"]].to_numpy(float)
    sec = pf[c["sector"]].astype(str).to_numpy()
    fidx = np.array([names.index(s) for s in sec], int)
    is_w = np.isin(sec, list(ww))
    rho_w = np.where(is_w, np.minimum(rho + boost, wrong_way_cap), rho)
    lgd_w = np.where(is_w, 1.0 - pf[c["rec_s"]].to_numpy(float), lgd) if c["rec_s"] in pf.columns else lgd
    g = copula_default_mc(p, ead, lgd, rho, fidx, C, n, seed, "gaussian", return_defaults=True, chunk_rows=chunk_rows)
    t = copula_default_mc(p, ead, lgd, rho, fidx, C, n, seed + int(t_seed_offset), "t", nu=nu, chunk_rows=chunk_rows)
    w = copula_default_mc(p, ead, lgd_w, rho_w, fidx, C, n, seed, "gaussian", chunk_rows=chunk_rows)
    Lg, Lt, Lw = (np.asarray(x["losses"], float) for x in (g, t, w))

    def ve(L, a):
        return loss_var_es(L, a, method=quantile_method, tail=es_tail)

    vc = [{"copula": nm, "confidence_level": a, "VaR": ve(L, a)[0], "CVaR": ve(L, a)[1]}
          for nm, L in (("gaussian", Lg), ("student_t", Lt)) for a in levels]
    wr = [{"scenario": nm, "confidence_level": a, "VaR": ve(L, a)[0], "CVaR": ve(L, a)[1]}
          for nm, L in (("base", Lg), ("wrong_way", Lw)) for a in levels]
    contrib, v_e, es_e = euler_es_contributions(g["defaults"], ead * lgd, Lg, euler_level, method=quantile_method)
    del g["defaults"]
    wts = ead / ead.sum()
    mc = pd.DataFrame({"obligor_id": pf[c["id"]].to_numpy() if c["id"] in pf.columns else np.arange(1, len(p) + 1),
                       "issuer": pf[c["issuer"]].to_numpy() if c["issuer"] in pf.columns else "",
                       "sector": sec, "exposure_usd": ead, "exposure_weight": wts,
                       "marginal_cvar_99": contrib, "pct_contribution": 100.0 * contrib / es_e})
    mc = mc.sort_values("marginal_cvar_99", ascending=False, kind="mergesort").reset_index(drop=True)
    el = float(np.sum(p * ead * lgd))

    def lstats(nm, L):
        return {"copula": nm, "mean_loss": float(L.mean()), "std_loss": float(L.std()),
                "skewness": float(stats.skew(L)), "kurtosis": float(stats.kurtosis(L)), "max_loss": float(L.max()),
                "pct_zero_loss": float(100.0 * np.mean(L == 0)), "expected_loss_analytical": el}

    ls = pd.DataFrame([lstats("gaussian", Lg), lstats("student_t", Lt)])
    sec_exp = pd.Series(ead, index=sec).groupby(level=0, sort=False).sum()
    order = [s for s in names if s in sec_exp.index] + [s for s in sec_exp.index if s not in names]
    sec_exp = sec_exp.reindex(order)
    sec_w = sec_exp / ead.sum()
    top_e = mc.sort_values("exposure_weight", ascending=False, kind="mergesort").head(5)
    top_m = mc.head(5)
    conc = {"obligor_hhi": hhi(ead), "sector_hhi": hhi(sec_exp.to_numpy(float)),
            "top_5_exposures": [{"obligor_id": int(r.obligor_id), "issuer": str(r.issuer), "weight": float(r.exposure_weight)}
                                for r in top_e.itertuples()],
            "top_5_marginal_cvar": [{"obligor_id": int(r.obligor_id), "issuer": str(r.issuer),
                                     "marginal_cvar_99": float(r.marginal_cvar_99)} for r in top_m.itertuples()],
            "sector_weights": {str(k): float(v) for k, v in sec_w.items()}}
    ab, ga = [], []
    for a in levels:
        mv = ve(Lg, a)[0]
        vv = vasicek_portfolio_var(p, ead, lgd, rho, a)
        gg = gordy_granularity_adjustment(p, ead, lgd, rho, a)
        gv = float(gg) if np.isscalar(gg) else float(np.asarray(gg).ravel()[0])
        ab.append({"confidence_level": a, "mc_var_gaussian": mv, "vasicek_var": vv, "analytical_el": el,
                   "mc_mean_loss": float(Lg.mean()), "el_ratio": float(Lg.mean()) / el})
        ga.append({"confidence_level": a, "vasicek_var": vv, "granularity_adjustment": gv, "gordy_var": vv + gv,
                   "mc_var_gaussian": mv, "gordy_to_mc_ratio": (vv + gv) / mv if mv else float("nan")})
    srd = []
    for s in order:
        m = mc["sector"] == s
        e_s = float(mc.loc[m, "exposure_usd"].sum())
        mcv = float(mc.loc[m, "marginal_cvar_99"].sum())
        ew = e_s / ead.sum()
        pc = 100.0 * mcv / es_e
        srd.append({"sector": s, "n_obligors": int(m.sum()), "total_exposure": e_s, "exposure_weight": ew,
                    "marginal_cvar_99": mcv, "pct_of_total_cvar": pc, "risk_weight_ratio": pc / (100.0 * ew) if ew else float("nan")})
    k = level_key(euler_level)
    gv99, tv99, wv99 = ve(Lg, euler_level), ve(Lt, euler_level), ve(Lw, euler_level)
    v99 = vasicek_portfolio_var(p, ead, lgd, rho, euler_level)
    gg99 = gordy_granularity_adjustment(p, ead, lgd, rho, euler_level)
    gg99 = float(gg99) if np.isscalar(gg99) else float(np.asarray(gg99).ravel()[0])
    summ = {"n_obligors": int(len(p)), "total_exposure_usd": float(ead.sum()), "n_simulations": n,
            f"gaussian_var_{k}": gv99[0], f"gaussian_cvar_{k}": gv99[1], f"student_t_var_{k}": tv99[0],
            f"student_t_cvar_{k}": tv99[1], f"wrong_way_var_{k}": wv99[0], f"wrong_way_cvar_{k}": wv99[1],
            f"var_ratio_t_vs_gaussian_{k}": tv99[0] / gv99[0] if gv99[0] else float("nan"),
            "wrong_way_cvar_increase_pct": 100.0 * (wv99[1] - gv99[1]) / gv99[1] if gv99[1] else float("nan"),
            "obligor_hhi": conc["obligor_hhi"], "max_exposure_weight": float(wts.max()),
            "mean_portfolio_pd": float(np.sum(wts * p)), "analytical_el": el, "mc_el_ratio": float(Lg.mean()) / el,
            f"vasicek_var_{k}": v99, f"vasicek_to_mc_ratio_{k}": v99 / gv99[0] if gv99[0] else float("nan"),
            "marginal_cvar_sum_check": float(contrib.sum()) / es_e, f"gordy_var_{k}": v99 + gg99,
            f"granularity_adjustment_{k}": gg99}
    return {"var_cvar_results.csv": pd.DataFrame(vc), "wrong_way_results.csv": pd.DataFrame(wr),
            "marginal_contributions.csv": mc, "loss_statistics.csv": ls, "concentration_risk.json": conc,
            "summary.json": summ, "analytical_benchmark.csv": pd.DataFrame(ab),
            "sector_risk_decomposition.csv": pd.DataFrame(srd), "granularity_adjustment.csv": pd.DataFrame(ga)}


# =============================================================================================
# 5. CreditMetrics migration report with flat keys
# =============================================================================================

def creditmetrics_report(P, initial_counts, rho, n_sims, seed, lgd, face=1.0, ratings=None,
                         var_levels=(0.95, 0.99, 0.999), es_levels=(0.95, 0.99), rng="legacy",
                         es_tail="geq") -> dict:
    """-> dict{portfolio_loss {mean_loss, std_loss, var_95, var_99, var_999, es_95, es_99}, loss_by_rating DataFrame, migration_stats, summary_core} (flat keys, never tuples).

    Wraps creditmetrics_default_sim (one-factor Gaussian CreditMetrics, thresholds cumulated from the
    worst state, M drawn first; rng 'legacy' = np.random.seed stream, 'default' = default_rng).
    P in PROBABILITIES (a percent table: divide by 100 first; rows beyond len(initial_counts) are
    ignored), last column = Default. Keys: var_<level_key>, es_<level_key> (0.95 -> 95, 0.999 -> 999).
    loss_by_rating: initial_rating (ratings labels, default R0, R1, ...), n_bonds, mean_loss (mean over
    simulations of the category's total loss), prob_default (bond-simulation default fraction).
    migration_stats: upgrade_rate, downgrade_rate (worse but not default), default_rate.
    summary_core: n_bonds, n_simulations, correlation, mean_loss, var_99, es_99 (add task / recovery)."""
    from .helpers76_4 import creditmetrics_default_sim
    from .helpers75_2 import level_key
    Pm = np.asarray(P, float)
    if Pm.max() > 1.0 + 1e-9:
        raise ValueError("P looks like percent: pass np.asarray(table)/100")
    lv = sorted(set(map(float, var_levels)) | set(map(float, es_levels)))
    r = creditmetrics_default_sim(Pm, initial_counts, rho, n_sims, seed, lgd, face, rng=rng, levels=tuple(lv),
                                  es_tail=es_tail)
    pl = {"mean_loss": r["mean_loss"], "std_loss": r["std_loss"]}
    for a in var_levels:
        pl[f"var_{level_key(a)}"] = float(r["var"][float(a)])
    for a in es_levels:
        pl[f"es_{level_key(a)}"] = float(r["es"][float(a)])
    _std_r = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C"]
    labs = list(ratings) if ratings is not None else (_std_r[:len(initial_counts)] if len(initial_counts) <= len(_std_r) else [f"R{i}" for i in range(len(initial_counts))])
    by = r["by_rating"].copy()
    by.insert(0, "initial_rating", labs[:len(by)])
    ms = {"upgrade_rate": r["upgrade_rate"], "downgrade_rate": r["downgrade_rate"], "default_rate": r["default_rate"]}
    core = {"n_bonds": int(np.sum(initial_counts)), "n_simulations": int(n_sims), "correlation": float(rho),
            "mean_loss": r["mean_loss"]}
    for k in ("var_99", "es_99"):
        if k in pl:
            core[k] = pl[k]
    return {"portfolio_loss": pl, "loss_by_rating": by, "migration_stats": ms, "summary_core": core, "sim": r}


# =============================================================================================
# 6. VaR backtest coverage tests (chi-square SURVIVAL p-values)
# =============================================================================================

def var_coverage_tests(hits, level, prefix="") -> dict:
    """-> flat dict{<prefix>violations, kupiec_lr, kupiec_pvalue, christoffersen_ind_lr, christoffersen_ind_pvalue, cc_lr, cc_pvalue}; p = chi2.sf(LR, df).

    hits: 0/1 (or bool) violation indicators in time order; level: VaR confidence (0.99 -> expected
    rate 0.01). Kupiec POF (all T days, df 1), Christoffersen independence on the T-1 consecutive
    pairs (df 1), conditional coverage LR_cc = LR_uc + LR_ind (df 2); 0 ln 0 = 0 (kupiec_pof,
    christoffersen_independence). The p-value is ALWAYS scipy.stats.chi2.sf(LR, df) - exp(-LR/2) is the
    df=2 survival only and is wrong for the df=1 tests."""
    from .risk import kupiec_pof
    from .riskx import christoffersen_independence
    h = np.asarray(hits).astype(int).ravel()
    lv = float(level) if float(level) < 1 else float(level) / 100.0
    x, T = int(h.sum()), int(h.size)
    lr_uc, p_uc = kupiec_pof(x, T, lv)
    lr_ind, p_ind = christoffersen_independence(h)
    lr_cc = lr_uc + lr_ind
    pre = str(prefix)
    return {f"{pre}violations": x, f"{pre}kupiec_lr": float(lr_uc), f"{pre}kupiec_pvalue": float(stats.chi2.sf(lr_uc, 1)),
            f"{pre}christoffersen_ind_lr": float(lr_ind),
            f"{pre}christoffersen_ind_pvalue": float(stats.chi2.sf(lr_ind, 1)),
            f"{pre}cc_lr": float(lr_cc), f"{pre}cc_pvalue": float(stats.chi2.sf(lr_cc, 2))}


# =============================================================================================
# 7. EMA-crossover CTA + GARCH Basel IMA capital report
# =============================================================================================

def cta_ima_capital_report(prices, fast_ema, slow_ema, vol_lookback, vol_target, n_assets, max_leverage,
                           rf_annual, level, horizon, multiplier, window=250, warmup="zero", ddof=1,
                           periods_per_year=252, fit_scale="mle", text_conventions=False) -> dict:
    """-> dict{results {capital_charge, stressed_var, var_component, svar_component}, solution {intermediates}, portfolio_returns}: EMA-crossover vol-targeted CTA + GARCH IMA capital.

    prices: [T+1 x n] matrix of ALL asset columns. Strategy = cta_ema_vol_target_returns (SMA-seeded
    EMAs on prices, EWMA variance a = 2/(L+1) seeded with r0^2, w = clip(signal (target/n)/vol),
    portfolio log return sum w_{t-1} r_t). warmup="zero" (default, know78): an EMA that is still
    undefined ('treat as NaN') gives a NaN signal -> weight 0 (the sibling CTA spec: 'Replace NaN values
    with 0'), and the zero-return warm-up rows STAY in the series (r_p[t] for every t >= 1) that feeds the
    metrics and the GARCH fits; know76 tried "mask" (rows dropped) + warm-started refits and know77
    "short" (-1 during warm-up) + fresh refits - both WRONG. warmup="short": -1 while undefined;
    warmup="mask": 0 and the series starts at the first traded row.
    Performance: log_return_performance(r_p, A, rf_annual, ddof). Capital: basel_ima_garch_capital(r_p,
    level, horizon, multiplier, window, mean='zero', reported in native units; fit_scale="mle" (know79
    default) fits on the scale (1 or 100) with the higher native-unit log-likelihood - the MLE is
    scale-equivariant, but arch stalls at its starting values on raw decimal returns; fit_scale=1.0 is the
    raw fit, 100.0 the percent fit descaled): VaR = z
    sqrt(sigma2_next) sqrt(horizon); stressed VaR = max over every window-long GARCH refit; components
    = multiplier x each; capital = sum. intermediates: annualized_return, annualized_volatility,
    sharpe_ratio, max_drawdown, calmar_ratio, avg_abs_leverage, garch_omega, garch_alpha, garch_beta,
    garch_persistence, long_run_variance, var_<level_key> (each {"value": x}).
    know81 text_conventions=True: when the task text is available (qf_toolkit.instruction_text()), the
    conventions reference table decides what the task's words settle (helpers81_1.cta_conventions_from_text):
    warmup ('-1 otherwise' / "no 'no position' state" -> "short"), fit_scale ('native units' / 'do not
    rescale' -> 1.0, the arch fit on the raw series), and the metrics follow the table's textbook definitions
    (log_performance_table: annualized_return = mean x A, ddof=1, Sharpe on the excess return, drawdown sign
    from the table). Silent text / no text -> the arguments above stand. result['conventions'] lists them."""
    from .helpers76_1 import cta_ema_vol_target_returns
    from .helpers75_3 import log_return_performance
    from .helpers75_1 import basel_ima_garch_capital
    from .helpers75_2 import level_key
    cv = {}
    if text_conventions:
        from .helpers81_1 import cta_conventions_from_text
        cv = cta_conventions_from_text()
        warmup = cv.get("warmup", warmup)
        fit_scale = cv.get("fit_scale", fit_scale)
    P = np.asarray(prices, float)
    c = cta_ema_vol_target_returns(P, fast_ema, slow_ema, vol_lookback, vol_target, n_assets=n_assets,
                                   max_leverage=max_leverage, periods_per_year=periods_per_year, warmup=warmup, lag=1)
    rp = np.asarray(c["portfolio_returns"], float)
    if cv.get("evidence"):
        from .helpers81_1 import log_performance_table
        perf = log_performance_table(rp, periods_per_year=periods_per_year, rf_annual=rf_annual,
                                     ddof=cv.get("ddof", 1), mdd_sign=cv.get("mdd_sign", "negative"))
    else:
        perf = log_return_performance(rp, periods_per_year=periods_per_year, rf_annual=rf_annual, ddof=ddof)
    rp_garch = rp[1:] if (warmup == "zero" and len(rp) > 1 and rp[0] == 0.0) else rp
    sc = 100.0 if fit_scale == "mle" else float(fit_scale)
    cap = basel_ima_garch_capital(rp_garch, level=level, horizon=horizon, multiplier=multiplier, window=window, scale=sc,
                                  mean="zero", periods_per_year=periods_per_year)
    from .helpers75_1 import garch11_forecast
    g_const = garch11_forecast(rp, mean="constant", scale=100.0, periods_per_year=periods_per_year)
    res = {"capital_charge": float(cap["capital_charge"]), "stressed_var": float(cap["stressed_var"]),
           "var_component": float(cap["var_component"]), "svar_component": float(cap["svar_component"])}
    vals = {"annualized_return": perf["annualized_return"], "annualized_volatility": perf["annualized_volatility"],
            "sharpe_ratio": perf["sharpe_ratio"], "max_drawdown": perf["max_drawdown"],
            "calmar_ratio": perf["calmar_ratio"], "avg_abs_leverage": c["avg_abs_leverage"],
            "garch_omega": float(g_const["omega"]), "garch_alpha": float(cap["garch_alpha"]), "garch_beta": float(cap["garch_beta"]),
            "garch_persistence": cap["garch_persistence"], "long_run_variance": cap["long_run_variance"],
            f"var_{level_key(level)}": cap["var"]}
    sol = {"intermediates": {k: {"value": float(v)} for k, v in vals.items()}}
    return {"results": res, "solution": sol, "portfolio_returns": rp, "cta": c, "capital": cap,
            "conventions": {"warmup": warmup, "fit_scale": fit_scale, **cv}}
