"""know76 group-3 helpers: WHOLE-TASK report builders that lock the decisive conventions of a
pipeline in one tested call (the know75 runs crashed on guessed result keys, index-space mix-ups,
argument order and cut replies when the model re-assembled the pieces by hand).

Every function is general (no task names), pure numpy/pandas/scipy (arch / statsmodels only
where the method needs them), states its formula and conventions in the docstring, and RETURNS
the deliverables (dicts / DataFrames ready for emit); nothing is written unless a path is given.
Values a task fixes (grids, windows, seeds, thresholds, tolerances) are arguments - pass the
task's own values. Self-tests: proven_work/fail72/selftest76_3.py.

API lines (one per function):
     geske_compound_deliverables(S0, sigma, r, q, K1_list, moneyness, T1, T2, chooser_times, chooser_T, n_paths, seed, parity_tol, z=3.0, mc_q=None) -> dict{compound (2 x grid rows), parity (ONE row per K1 x m: call_on_call vs put_on_call), chooser, summary}
     cash_digital_barrier(S, K, H, T, r, sigma, q=0.0, kind="call", barrier="down-and-out") -> cash-or-nothing (pays 1) barrier digital, reflection principle; 'down-and-out' call / 'up-and-out' put (+ '-in' by parity)
     digital_barrier_suite(S0, sigma, r, q, T_list, moneyness, gap_trigger_moneyness, H_down_ratio, H_up_ratio, gap_K1=None) -> dict{digital, gap, barrier (down block then up block, T outer, m inner), summary}
     structured_note_report(log_returns, notional, S0, K, B, r, T, participation, level=0.99, garch_mean="zero", vol_source="long_run", delta_bump=1e-4, periods_per_year=252) -> dict{results, intermediates, garch, note}
     gbm_first_passage_report(S0, sigma, r, q, horizons, up_ratios, down_ratios, fpt_up_ratio, fpt_down_ratio, t_grid, n_paths, seed, steps_per_year=252) -> dict{running_max, running_min, first_passage, summary}; MC in LOG space X = ln(S/S0)
     ou_jump_threshold_fit(levels, dt, k=3.0) -> dict{a, b, kappa, theta, sigma_ou, lambda_jump, mu_J, sigma_J, n_jumps, n_obs, residuals, jump_mask}; ONE-PASS |e| > k sd(e) on the levels AR(1)
     ou_jump_report(levels, dt, taus, n_paths, n_steps, seed, mean_tol, var_tol_pct, k=3.0, periods_per_year=None) -> dict{calibration, conditional_moments, stationary, summary}
     funding_carry_report(raw, params, columns=None) -> dict{results, solution, funding_clean, regime_stats, ou_diagnostics, mc_stats}
     kelly_var_sizing_report(raw, params, asset_prefix="asset_", periods_per_year=252) -> dict{results, intermediates, fractions, schemes}
     cta_regime_report(prices, params, periods_per_year=252) -> dict{results, intermediates, composite_vol, regime_codes, static, dynamic}
     absorption_riskparity_report(raw, params, periods_per_year=252, var_level=None) -> dict{results, intermediates, rebalance_dates, weights, strategy_returns}
     tca_participation_report(quotes, trades, order_spec, header, checkpoint_buckets, out_dir=None, decimals=6) -> dict{results (ordered, rounded), fills, solution}; writes results.json / fills.csv / solution.json when out_dir
     intraday_volume_schedule(raw, order_datetime, quantity, window, half_life, winsor_pct, session_start="09:30:00", session_end="16:00:00", last_bar_start="15:55:00", bar_minutes=5, bars_per_day=None, models=(...)) -> dict{excluded_days, model_performance, best_model, final_schedule}
     american_fd_report(S0, K, r, sigma, T, dividends, q=0.0, S_max=None, N_S=300, N_T=600, coarse=(150, 300), omega=1.2, tol=1e-8, boundary_tol=1e-6, sample_S_every=5, sample_t_every=10, png_path=None) -> dict{option_values, grid, boundary, greeks, convergence, summary}
     mc_greeks_report(S0, K, r, q, sigma, T, n_paths, seed, n_avg, h_spot, h_vol, h_time, h_rate, spots, vols, conv_sizes, rel_tol, lr_gamma_tol, parity_tols, conv_tol, png_path=None) -> dict{prices, greeks, consistency, delta_surface, vega_surface, convergence, summary}
     credit_spread_report(series, start, end, window, lags, r2_threshold, ffill=("TEDRATE",), names=None) -> dict{decomposition, variance, regime_breaks, aligned}
"""
from __future__ import annotations

# know82 _rd82: report results resolve short / nested keys
def _rd82(d):
    from ._rdict import ResultDict
    return ResultDict({k: (ResultDict(v) if isinstance(v, dict) and not isinstance(v, ResultDict) else v) for k, v in d.items()})

import json
import math
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm


def _f(x):
    """float() that keeps None / NaN as None (JSON null)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _bs(S, K, T, r, q, sig, call=True):
    S, K, T = (np.asarray(v, dtype=float) for v in (S, K, T))
    v = sig * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig * sig) * T) / v
    d2 = d1 - v
    if call:
        return S * np.exp(-q * T) * _norm.cdf(d1) - K * np.exp(-r * T) * _norm.cdf(d2)
    return K * np.exp(-r * T) * _norm.cdf(-d2) - S * np.exp(-q * T) * _norm.cdf(-d1)


# ============================================================================ 1. Geske compound
def geske_compound_deliverables(S0, sigma, r, q, K1_list, moneyness, T1, T2, chooser_times, chooser_T,
                                n_paths, seed, parity_tol, z=3.0, mc_q=None) -> dict:
    """All Geske (1979) compound-option deliverables in one call.

    compound: geske_compound_grid(kinds=('call_on_call', 'put_on_put')) -> 2 x len(K1) x len(m)
      rows, type labels HYPHENATED 'call-on-call' / 'put-on-put'.
    parity: ONE row per (K1, m) of the grid (never one per compound row): cc = call_on_call,
      pc = put_on_CALL (same inner call), bs_call = BS(S0, K2, T2, call, q), K1_pv = K1 e^{-r T1},
      parity_lhs = cc - pc, parity_rhs = bs_call - K1_pv, parity_error = |lhs - rhs|.
    chooser: for each decision time t: chooser_simple_price(S0, K=S0, t, chooser_T), BS call and put
      at chooser_T (same r, sigma, q).
    summary: sigma, S0, n_compound_prices = rows of compound, max_parity_error, parity_holds =
      max error < parity_tol, max_mc_error = max |mc - closed form|, mc_validates = all
      |price - mc| <= z se, mean_cc_price / mean_pp_price (means by label, '_' or '-' accepted),
      chooser_price_T<100 t, 3 digits> keys (t = 0.1 -> 'chooser_price_T010').
    """
    from .compound import compound_option_price, compound_parity_rhs, chooser_simple_price
    from .helpers75_4 import geske_compound_grid, mc_agreement
    g = geske_compound_grid(S0, list(K1_list), list(moneyness), T1, T2, r, sigma, q=q,
                            kinds=("call_on_call", "put_on_put"), n_paths=n_paths, seed=seed, mc_q=mc_q)
    rows = []
    for K1 in K1_list:
        for m in moneyness:
            K2 = S0 * m
            cc = compound_option_price(S0, K1, K2, T1, T2, r, sigma, q=q, kind="call_on_call")
            pc = compound_option_price(S0, K1, K2, T1, T2, r, sigma, q=q, kind="put_on_call")
            bsc = float(_bs(S0, K2, T2, r, q, sigma, True))
            k1pv = K1 * math.exp(-r * T1)
            rhs = compound_parity_rhs(S0, K1, K2, T1, T2, r, sigma, q=q, inner="call")
            rows.append({"K1": K1, "K2": K2, "K2_moneyness": m, "cc_price": cc, "pc_price": pc,
                         "bs_call": bsc, "K1_pv": k1pv, "parity_lhs": cc - pc, "parity_rhs": rhs,
                         "parity_error": abs((cc - pc) - rhs)})
    par = pd.DataFrame(rows, columns=["K1", "K2", "K2_moneyness", "cc_price", "pc_price", "bs_call",
                                      "K1_pv", "parity_lhs", "parity_rhs", "parity_error"])
    ch = []
    for t in chooser_times:
        ch.append({"T_choose": t, "K": S0, "T_underlying": chooser_T,
                   "chooser_price": float(chooser_simple_price(S0, S0, t, chooser_T, r, sigma, q=q)),
                   "call_price": float(_bs(S0, S0, chooser_T, r, q, sigma, True)),
                   "put_price": float(_bs(S0, S0, chooser_T, r, q, sigma, False))})
    chooser = pd.DataFrame(ch, columns=["T_choose", "K", "T_underlying", "chooser_price", "call_price",
                                        "put_price"])
    lab = g["type"].astype(str).str.replace("_", "-")
    summ = {"sigma": float(sigma), "S0": float(S0), "n_compound_prices": int(len(g)),
            "max_parity_error": float(par["parity_error"].max()),
            "parity_holds": bool(par["parity_error"].max() < parity_tol),
            "max_mc_error": float(np.max(np.abs(g["compound_price"] - g["mc_price"]))),
            "mc_validates": bool(mc_agreement(g["compound_price"], g["mc_price"], g["mc_std_err"], z=z).all()),
            "mean_cc_price": float(g.loc[lab == "call-on-call", "compound_price"].mean()),
            "mean_pp_price": float(g.loc[lab == "put-on-put", "compound_price"].mean())}
    for _, row in chooser.iterrows():
        summ[f"chooser_price_T{int(round(row['T_choose'] * 100)):03d}"] = float(row["chooser_price"])
    return {"compound": g, "parity": par, "chooser": chooser, "summary": summ}


# ============================================================================ 2. digital / barrier digital
def cash_digital_barrier(S, K, H, T, r, sigma, q=0.0, kind="call", barrier="down-and-out"):
    """Cash-or-nothing (pays 1 at T) single-barrier digital, continuous monitoring, by reflection.

    nu = r - q - sigma^2/2, p = 2 nu / sigma^2 (= 2(r-q)/sigma^2 - 1), d2(X) = [ln(S/X) + nu T]/(sigma sqrt T),
    d2_H(X) = [ln(H^2/(S X)) + nu T]/(sigma sqrt T).
      down-and-out CALL (H < S): e^{-rT}[N(d2(K')) - (H/S)^p N(d2_H(K'))], K' = max(K, H)
      up-and-out PUT   (H > S): e^{-rT}[N(-d2(K')) - (H/S)^p N(-d2_H(K'))], K' = min(K, H)
    (K beyond the barrier: survival already implies the payoff region, so K is replaced by H).
    '-in' versions = vanilla cash digital - knock-out (parity). Already breached -> out 0, in vanilla.
    Other combinations raise ValueError. Arrays broadcast."""
    S, K, H, T = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (S, K, H, T)))
    kind = str(kind).lower()
    b = str(barrier).lower().replace("_", "-")
    down = b.startswith("down")
    if (down and not kind.startswith("c")) or (not down and not kind.startswith("p")):
        raise ValueError("cash_digital_barrier supports down-and-out/in CALL and up-and-out/in PUT")
    vt = sigma * np.sqrt(T)
    nu = r - q - 0.5 * sigma ** 2
    p = 2.0 * nu / sigma ** 2
    disc = np.exp(-r * T)
    Kp = np.maximum(K, H) if down else np.minimum(K, H)
    d2 = (np.log(S / Kp) + nu * T) / vt
    d2h = (np.log(H * H / (S * Kp)) + nu * T) / vt
    if down:
        out = disc * (_norm.cdf(d2) - (H / S) ** p * _norm.cdf(d2h))
        van = disc * _norm.cdf((np.log(S / K) + nu * T) / vt)
        hit = S <= H
    else:
        out = disc * (_norm.cdf(-d2) - (H / S) ** p * _norm.cdf(-d2h))
        van = disc * _norm.cdf(-(np.log(S / K) + nu * T) / vt)
        hit = S >= H
    out = np.where(hit, 0.0, np.maximum(out, 0.0))
    res = out if b.endswith("out") else van - out
    return float(res) if res.ndim == 0 else res


def digital_barrier_suite(S0, sigma, r, q, T_list, moneyness, gap_trigger_moneyness, H_down_ratio, H_up_ratio,
                          gap_K1=None) -> dict:
    """Vanilla digitals, gap calls and cash barrier digitals with the identity checks.

    digital (T outer, m inner; K = m S0): d1 = [ln(S0/K) + (r-q+sigma^2/2)T]/(sigma sqrt T), d2 = d1 - sigma sqrt T;
      cash_call = e^{-rT}N(d2), cash_put = e^{-rT}N(-d2), asset_call = S0 e^{-qT}N(d1),
      asset_put = S0 e^{-qT}N(-d1), bs_call/bs_put = Black-Scholes.
    gap (T outer, trigger m inner): K1 = gap_K1 (default S0) pays, K2 = m S0 triggers:
      gap_call = S0 e^{-qT}N(d1(K2)) - K1 e^{-rT}N(d2(K2)).
    barrier: block 'down_out_cash_call' (H = H_down_ratio S0) then block 'up_out_cash_put'
      (H = H_up_ratio S0), each T outer, m inner (2 x len(T) x len(m) rows); knockout by
      cash_digital_barrier, vanilla = cash digital, knockin = vanilla - knockout, parity_error =
      |KO + KI - vanilla|.
    summary: counts, max cash parity |cc + cp - e^{-rT}|, max asset parity |ac + ap - S0 e^{-qT}|,
      max BS decomposition |ac - K cc - bs_call|, max barrier parity, ATM (m = 1, T = middle of
      T_list when 0.5 is absent) values *_T050, cash_call_increases_with_S (cash call strictly
      increasing in S at fixed K, i.e. decreasing in m at every T), do_leq_vanilla (all DO <= vanilla)."""
    T_list, moneyness = list(T_list), list(moneyness)
    gK1 = S0 if gap_K1 is None else gap_K1
    rows = []
    for T in T_list:
        for m in moneyness:
            K = S0 * m
            vt = sigma * math.sqrt(T)
            d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / vt
            d2 = d1 - vt
            rows.append({"T": T, "moneyness": m, "K": K,
                         "cash_call": math.exp(-r * T) * _norm.cdf(d2), "cash_put": math.exp(-r * T) * _norm.cdf(-d2),
                         "asset_call": S0 * math.exp(-q * T) * _norm.cdf(d1),
                         "asset_put": S0 * math.exp(-q * T) * _norm.cdf(-d1),
                         "bs_call": float(_bs(S0, K, T, r, q, sigma, True)),
                         "bs_put": float(_bs(S0, K, T, r, q, sigma, False))})
    dig = pd.DataFrame(rows)
    grows = []
    for T in T_list:
        for m in gap_trigger_moneyness:
            K2 = S0 * m
            vt = sigma * math.sqrt(T)
            d1 = (math.log(S0 / K2) + (r - q + 0.5 * sigma ** 2) * T) / vt
            grows.append({"T": T, "trigger_moneyness": m, "K2": K2, "K1": gK1,
                          "gap_call": S0 * math.exp(-q * T) * _norm.cdf(d1) - gK1 * math.exp(-r * T) * _norm.cdf(d1 - vt)})
    gap = pd.DataFrame(grows)
    brows = []
    for label, ratio, kind, bar in (("down_out_cash_call", H_down_ratio, "call", "down-and-out"),
                                    ("up_out_cash_put", H_up_ratio, "put", "up-and-out")):
        H = S0 * ratio
        for T in T_list:
            for m in moneyness:
                K = S0 * m
                ko = float(cash_digital_barrier(S0, K, H, T, r, sigma, q, kind, bar))
                van = float(math.exp(-r * T) * _norm.cdf((1 if kind == "call" else -1)
                                                         * (math.log(S0 / K) + (r - q - 0.5 * sigma ** 2) * T)
                                                         / (sigma * math.sqrt(T))))
                ki = van - ko
                brows.append({"T": T, "moneyness": m, "K": K, "barrier_type": label, "H": H,
                              "knockout_price": ko, "knockin_price": ki, "vanilla_price": van,
                              "parity_error": abs(ko + ki - van)})
    bar = pd.DataFrame(brows)
    Tm = 0.5 if any(abs(t - 0.5) < 1e-12 for t in T_list) else T_list[len(T_list) // 2]

    def pick(df, col, mcol="moneyness", extra=None):
        msk = np.isclose(df["T"], Tm) & np.isclose(df[mcol], 1.0)
        if extra is not None:
            msk &= df["barrier_type"] == extra
        return float(df.loc[msk, col].iloc[0]) if msk.any() else None

    mono = all(bool(np.all(np.diff(dig.loc[np.isclose(dig["T"], T)].sort_values("moneyness")["cash_call"]) < 0))
               for T in T_list)
    do = bar[bar["barrier_type"] == "down_out_cash_call"]
    summ = {"n_digital_prices": int(len(dig)), "n_gap_prices": int(len(gap)), "n_barrier_prices": int(len(bar)),
            "max_cash_parity_error": float(np.max(np.abs(dig["cash_call"] + dig["cash_put"] - np.exp(-r * dig["T"])))),
            "max_asset_parity_error": float(np.max(np.abs(dig["asset_call"] + dig["asset_put"] - S0 * np.exp(-q * dig["T"])))),
            "max_bs_decomp_error": float(np.max(np.abs(dig["asset_call"] - dig["K"] * dig["cash_call"] - dig["bs_call"]))),
            "max_barrier_parity_error": float(bar["parity_error"].max()),
            "atm_cash_call_T050": pick(dig, "cash_call"), "atm_cash_put_T050": pick(dig, "cash_put"),
            "atm_asset_call_T050": pick(dig, "asset_call"),
            "atm_gap_call_T050": pick(gap, "gap_call", "trigger_moneyness"),
            "atm_do_cash_call_T050": pick(bar, "knockout_price", extra="down_out_cash_call"),
            "atm_uo_cash_put_T050": pick(bar, "knockout_price", extra="up_out_cash_put"),
            "cash_call_increases_with_S": bool(mono),
            "do_leq_vanilla": bool(np.all(do["knockout_price"] <= do["vanilla_price"] + 1e-15))}
    return {"digital": dig, "gap": gap, "barrier": bar, "summary": summ}


# ============================================================================ 3. structured note
def structured_note_report(log_returns, notional, S0, K, B, r, T, participation, level=0.99, garch_mean="zero",
                           vol_source="conditional", delta_bump=1e-4, periods_per_year=252, risk_vol="pricing",
                           risk_horizon_days=None, markup_base=None, n_options_base=None) -> dict:
    """Zero bond + participation in a down-and-out call, priced at a GARCH(1,1) vol, with risk.

    GARCH: garch11_forecast(log_returns, mean=garch_mean, scale=100) -> omega (DECIMAL daily
      variance), alpha, beta; persistence = alpha + beta; long_run_variance = omega/(1 - alpha - beta).
    Pricing vol: vol_source="long_run" -> garch_vol = sqrt(A * long_run_variance) (annualised
      long-run vol); "forecast" -> sqrt(A * one-step forecast variance).
    Note: capital_protected_note(..., sigma=garch_vol, barrier='down-and-out', kind='call',
      delta_bump=relative spot bump, level) which calls barrier_price(S, K, H, T, r, sigma) with T
      BEFORE r: bond_pv = N e^{-rT}; n_options = participation N / S0; option_unit_price;
      option_value; fair_value; markup_pct = 100 (N - fair)/N; breakeven_vol; option_delta
      (central FD, dS = delta_bump S0); note_delta; note_var_99 = z |note_delta| S0 sigma/sqrt(A);
      note_es_99 = |note_delta| S0 sigma/sqrt(A) phi(z)/(1 - level).
    know77 convention switches (defaults = the proven run): risk_vol='pricing' (VaR/ES at the pricing
      vol) | 'forecast' (one-step GARCH forecast vol); risk_horizon_days h scales VaR/ES by sqrt(h).
    Returns {results (14 keys), intermediates {name: {'value': float}} (12 keys), garch, note}.
    intermediates[k] is a DICT {'value': x}: float(n['intermediates'][k]) raises - use
    n['results'][k] or n['intermediates'][k]['value'].
    know80: risk_horizon_days / markup_base / n_options_base = None -> read from the task's own words by
      note_conventions_from_text(instruction_text()) (defaults 1 day, 'notional', 'S0' when the text is
      silent); pass a value to override."""
    from .helpers75_1 import garch11_forecast, capital_protected_note
    if risk_horizon_days is None or markup_base is None or n_options_base is None:
        from .helpers80_1 import note_conventions_from_text
        cv = note_conventions_from_text()
        risk_horizon_days = cv["risk_horizon_days"] if risk_horizon_days is None else risk_horizon_days
        markup_base = cv["markup_base"] if markup_base is None else markup_base
        n_options_base = cv["n_options_base"] if n_options_base is None else n_options_base
    g = garch11_forecast(np.asarray(log_returns, dtype=float), mean=garch_mean, scale=100.0,
                         periods_per_year=periods_per_year)
    if vol_source == "conditional":   # know82 last conditional vol: sqrt(252 sigma2_T) at the last observation
        from .helpers75_3 import garch11_conditional_vol
        _cv = garch11_conditional_vol(np.asarray(log_returns, dtype=float), mean=garch_mean, scale=100.0)
        vol = float(np.asarray(_cv["sigma"], float)[-1] * math.sqrt(periods_per_year))
    elif vol_source == "long_run":
        vol = math.sqrt(periods_per_year * g["long_run_variance"])
    elif vol_source == "forecast":
        vol = g["garch_vol"]
    else:
        raise ValueError("vol_source must be 'conditional', 'long_run' or 'forecast'")
    note = capital_protected_note(notional, S0, K, B, r, T, participation, vol, barrier="down-and-out",
                                  kind="call", delta_bump=delta_bump, level=level,
                                  periods_per_year=periods_per_year, markup_base=markup_base,
                                  n_options_base=n_options_base)
    nv, ne = note["note_var"], note["note_es"]
    if risk_vol == "forecast" and vol > 0:
        nv, ne = nv * g["garch_vol"] / vol, ne * g["garch_vol"] / vol
    if risk_horizon_days and risk_horizon_days != 1:
        nv, ne = nv * math.sqrt(risk_horizon_days), ne * math.sqrt(risk_horizon_days)
    res = {"fair_value": note["fair_value"], "markup_pct": note["markup_pct"], "note_var_99": nv,
           "note_es_99": ne, "breakeven_vol": note["breakeven_vol"],
           "garch_persistence": g["persistence"], "long_run_variance": g["long_run_variance"], "garch_vol": vol,
           "bond_pv": note["bond_pv"], "option_unit_price": note["option_unit_price"],
           "option_value": note["option_value"], "n_options": note["n_options"],
           "option_delta": note["option_delta"], "note_delta": note["note_delta"]}
    res = {k: _f(v) for k, v in res.items()}
    inter = {"garch_omega": g["omega"], "garch_alpha": g["alpha"], "garch_beta": g["beta"],
             "garch_persistence": g["persistence"], "long_run_variance": g["long_run_variance"], "garch_vol": vol,
             "bond_pv": note["bond_pv"], "option_unit_price": note["option_unit_price"],
             "option_value": note["option_value"], "n_options": note["n_options"],
             "option_delta": note["option_delta"], "note_delta": note["note_delta"]}
    return {"results": res, "intermediates": {k: {"value": _f(v)} for k, v in inter.items()},
            "garch": g, "note": note}


# ============================================================================ 4. first passage / extrema
def gbm_first_passage_report(S0, sigma, r, q, horizons, up_ratios, down_ratios, fpt_up_ratio, fpt_down_ratio,
                             t_grid, n_paths, seed, steps_per_year=252) -> dict:
    """Running max/min, first-passage CDFs, E[tau] and the joint (X_T, M_T) check, analytical + MC.

    X_t = ln(S_t/S0) = mu t + sigma W_t with mu = r - q - sigma^2/2 (risk-neutral LOG drift).
    Analytical (continuous monitoring, exotics.barrier_hit_prob / first_passage_cdf /
      expected_first_passage_time / joint_end_max_cdf, drift_is_log=True):
      running_max rows (T outer, ratio inner): T, H_ratio, H, b = ln(H_ratio), prob_analytical, prob_mc
      running_min rows: T, L_ratio, L, a = ln(L_ratio), prob_analytical, prob_mc
      first_passage rows: 'up_<100 ratio>' then 'down_<100 ratio, 3 digits>' x t_grid: barrier_type, t, cdf_analytical, cdf_mc
    MC in LOG space (never compare price paths with log barriers): np.random.seed(seed);
      Z = np.random.standard_normal((n_paths, n_steps)), n_steps = round(max(T) steps_per_year),
      X = [0, cumsum(mu dt + sigma sqrt(dt) Z)]; P_mc(max >= b by T) = mean(max(X[:, :round(T/dt)+1]) >= b);
      hit time = first index k with X_k >= b (<= a) times dt, inf if never; cdf_mc(t) = mean(hit <= t);
      E_tau_up_mc = mean of finite up hit times; joint_prob_mc = mean(X_end <= 0 & max X <= ln(fpt_up_ratio)).
    summary: n_mc_paths, max_prob_up_relerr, max_prob_down_relerr, max_fpt_cdf_abserr, E_tau_up,
      E_tau_up_mc, prob_hit_<up>_T1, prob_hit_<down>_T1 (at max(horizons)), joint_prob_analytical,
      joint_prob_mc, max_decreases_with_barrier, min_increases_with_barrier, prob_increases_with_T."""
    from .exotics import barrier_hit_prob, expected_first_passage_time, joint_end_max_cdf
    mu = r - q - 0.5 * sigma ** 2
    horizons = list(horizons)
    Tmax = max(horizons)
    dt = 1.0 / steps_per_year
    n_steps = int(round(Tmax * steps_per_year))
    np.random.seed(int(seed))
    Z = np.random.standard_normal((int(n_paths), n_steps))
    X = np.zeros((int(n_paths), n_steps + 1))
    X[:, 1:] = np.cumsum(mu * dt + sigma * math.sqrt(dt) * Z, axis=1)
    del Z
    runmax = np.maximum.accumulate(X, axis=1)
    runmin = np.minimum.accumulate(X, axis=1)

    def nidx(T):
        return min(int(round(T * steps_per_year)), n_steps)

    mx, mn = [], []
    for T in horizons:
        k = nidx(T)
        for h in up_ratios:
            b = math.log(h)
            mx.append({"T": T, "H_ratio": h, "H": S0 * h, "b": b,
                       "prob_analytical": float(barrier_hit_prob(S0, S0 * h, T, mu, sigma, drift_is_log=True)),
                       "prob_mc": float(np.mean(runmax[:, k] >= b))})
        for l in down_ratios:
            a = math.log(l)
            mn.append({"T": T, "L_ratio": l, "L": S0 * l, "a": a,
                       "prob_analytical": float(barrier_hit_prob(S0, S0 * l, T, mu, sigma, drift_is_log=True)),
                       "prob_mc": float(np.mean(runmin[:, k] <= a))})
    rmax, rmin = pd.DataFrame(mx), pd.DataFrame(mn)
    bu, ad = math.log(fpt_up_ratio), math.log(fpt_down_ratio)

    def first_hit(mask):
        hit = mask.any(axis=1)
        idx = np.argmax(mask, axis=1).astype(float) * dt
        return np.where(hit, idx, np.inf)

    tu = first_hit(X >= bu)
    td = first_hit(X <= ad)
    lab_u = f"up_{int(round(fpt_up_ratio * 100)):03d}"
    lab_d = f"down_{int(round(fpt_down_ratio * 100)):03d}"
    fr = []
    for t in t_grid:
        fr.append({"barrier_type": lab_u, "t": t,
                   "cdf_analytical": float(barrier_hit_prob(S0, S0 * fpt_up_ratio, t, mu, sigma, drift_is_log=True)),
                   "cdf_mc": float(np.mean(tu <= t + 1e-12))})
    for t in t_grid:
        fr.append({"barrier_type": lab_d, "t": t,
                   "cdf_analytical": float(barrier_hit_prob(S0, S0 * fpt_down_ratio, t, mu, sigma, drift_is_log=True)),
                   "cdf_mc": float(np.mean(td <= t + 1e-12))})
    fpt = pd.DataFrame(fr)
    ja = float(joint_end_max_cdf(0.0, bu, Tmax, mu, sigma))
    jm = float(np.mean((X[:, -1] <= 0.0) & (runmax[:, -1] <= bu)))
    fin = np.isfinite(tu)
    dec = all(bool(np.all(np.diff(rmax.loc[rmax["T"] == T, "prob_analytical"]) < 0)) for T in horizons)
    inc = all(bool(np.all(np.diff(rmin.loc[rmin["T"] == T, "prob_analytical"]) > 0)) for T in horizons)
    incT = all(bool(np.all(np.diff(rmax.loc[np.isclose(rmax["H_ratio"], h)].sort_values("T")["prob_analytical"]) > 0))
               for h in up_ratios)
    summ = {"n_mc_paths": int(n_paths),
            "max_prob_up_relerr": float(np.max(np.abs(rmax["prob_analytical"] - rmax["prob_mc"]) / rmax["prob_analytical"])),
            "max_prob_down_relerr": float(np.max(np.abs(rmin["prob_analytical"] - rmin["prob_mc"]) / rmin["prob_analytical"])),
            "max_fpt_cdf_abserr": float(np.max(np.abs(fpt["cdf_analytical"] - fpt["cdf_mc"]))),
            "E_tau_up": _f(expected_first_passage_time(S0, S0 * fpt_up_ratio, mu, sigma, drift_is_log=True)),
            "E_tau_up_mc": _f(np.mean(tu[fin])) if fin.any() else None,
            f"prob_hit_{int(round(fpt_up_ratio * 100)):03d}_T1": float(barrier_hit_prob(S0, S0 * fpt_up_ratio, Tmax, mu, sigma)),
            f"prob_hit_{int(round(fpt_down_ratio * 100)):03d}_T1": float(barrier_hit_prob(S0, S0 * fpt_down_ratio, Tmax, mu, sigma)),
            "joint_prob_analytical": ja, "joint_prob_mc": jm,
            "max_decreases_with_barrier": bool(dec), "min_increases_with_barrier": bool(inc),
            "prob_increases_with_T": bool(incT)}
    return {"running_max": rmax, "running_min": rmin, "first_passage": fpt, "summary": summ,
            "mu_rn": mu}


# ============================================================================ 5. OU with jumps (one-pass threshold)
def ou_jump_threshold_fit(levels, dt, k=3.0) -> dict:
    """OU + jumps by ONE pass of a k-sigma residual filter on the levels AR(1) (no iteration, no refit).

    OLS S_{t+1} = a + b S_t + e (all transitions); kappa = -ln(b)/dt; theta = a/(1-b);
    sd = std(e, ddof=1); jump if |e| > k sd; n_jumps; lambda_jump = n_jumps/(n dt), n = transitions;
    mu_J = mean(e[jump]) (0 if none); sigma_J = std(e[jump], ddof=1) (0 with < 2 jumps);
    sigma_ou = std(e[~jump], ddof=1) sqrt(2 kappa/(1 - b^2)) (exact-discretisation scaling,
    re-estimated on the NON-jump residuals). Differs from recipes.fit_ou_jump, which iterates
    the filter and refits a, b on the clean transitions."""
    v = np.asarray(levels, dtype=float).ravel()
    v = v[np.isfinite(v)]
    x, y = v[:-1], v[1:]
    A = np.column_stack([np.ones_like(x), x])
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - (a + b * x)
    sd = float(np.std(e, ddof=1))
    jm = np.abs(e) > k * sd
    nj = int(jm.sum())
    n = int(len(e))
    kappa = -math.log(b) / dt
    sig = float(np.std(e[~jm], ddof=1)) * math.sqrt(2 * kappa / (1 - b * b))
    return {"a": float(a), "b": float(b), "kappa": kappa, "theta": float(a / (1 - b)), "sigma_ou": sig,
            "lambda_jump": nj / (n * dt), "mu_J": float(e[jm].mean()) if nj else 0.0,
            "sigma_J": float(np.std(e[jm], ddof=1)) if nj > 1 else 0.0, "n_jumps": nj, "n_obs": n,
            "residuals": e, "jump_mask": jm, "resid_std": sd}


def ou_jump_report(levels, dt, taus, n_paths, n_steps, seed, mean_tol=None, var_tol_pct=None, k=3.0,
                   periods_per_year=None) -> dict:
    """Calibration + conditional/stationary moments + MC check for dS = kappa(theta - S)dt + sigma dW + J dN (leave mean_tol/var_tol_pct None unless the task states them).

    know77: tolerances default to the MC-error convention when the task states none (a guessed
    mean_tol=1e-6 made mc_validates False): mean_tol = max(4 x max MC standard error sqrt(Var/n_paths),
    1% of the largest sqrt(Var_ST_jump)); var_tol_pct = 5 (percent).

    calibration: n_observations = number of daily changes, mean_level/std_level (ddof=1, all levels),
      kappa, theta, sigma_ou, lambda_jump, mu_J, sigma_J, n_jumps_detected (ou_jump_threshold_fit), S0 = last level.
    conditional_moments (one row per tau): E_ST_ou = theta + (S0-theta)e^{-k tau};
      Var_ST_ou = sigma^2(1 - e^{-2k tau})/(2k); E_ST_jump = E_ST_ou + lambda mu_J (1 - e^{-k tau})/k and
      Var_ST_jump = Var_ST_ou + lambda(mu_J^2 + sigma_J^2)(1 - e^{-2k tau})/(2k) are the TOTALS with jumps;
      mc_mean / mc_var (ddof=1) of simulate_ou_jump(S0, ..., T=max(taus), n_steps, n_paths, seed) at
      column round(tau n_steps / max(taus)).
    stationary: theta, sigma^2/(2k), theta + lambda mu_J/k, (sigma^2 + lambda(mu_J^2+sigma_J^2))/(2k),
      empirical mean/var (ddof=1) of the levels.
    summary: parameters, half_life_days = ln2/kappa * periods_per_year (default 1/dt), n_jumps_detected,
      max_mc_mean_error = max|mc_mean - E_ST_jump|, max_mc_var_error_pct = max|mc_var/Var_ST_jump - 1|*100,
      mc_validates = both below mean_tol and var_tol_pct."""
    from .recipes import simulate_ou_jump
    v = np.asarray(levels, dtype=float).ravel()
    v = v[np.isfinite(v)]
    fit = ou_jump_threshold_fit(v, dt, k)
    ch = np.diff(v)
    S0 = float(v[-1])
    kap, th, sg = fit["kappa"], fit["theta"], fit["sigma_ou"]
    lam, mJ, sJ = fit["lambda_jump"], fit["mu_J"], fit["sigma_J"]
    cal = {"n_observations": int(len(ch)), "mean_level": float(v.mean()), "std_level": float(v.std(ddof=1)),
           "mean_daily_change": float(ch.mean()), "std_daily_change": float(ch.std(ddof=1)),
           "kappa": kap, "theta": th, "sigma_ou": sg, "lambda_jump": lam, "mu_J": mJ, "sigma_J": sJ,
           "n_jumps_detected": int(fit["n_jumps"]), "S0": S0}
    taus = np.asarray(list(taus), dtype=float)
    e1 = np.exp(-kap * taus)
    e2 = np.exp(-2 * kap * taus)
    Eou = th + (S0 - th) * e1
    Vou = sg ** 2 * (1 - e2) / (2 * kap)
    Ej = Eou + lam * mJ * (1 - e1) / kap
    Vj = Vou + lam * (mJ ** 2 + sJ ** 2) * (1 - e2) / (2 * kap)
    Tm = float(taus.max())
    paths = simulate_ou_jump(S0, kap, th, sg, lam, mJ, sJ, T=Tm, n_steps=int(n_steps), n_paths=int(n_paths), seed=seed)
    idx = np.rint(taus / Tm * int(n_steps)).astype(int)
    mcm = paths[:, idx].mean(axis=0)
    mcv = paths[:, idx].var(axis=0, ddof=1)
    cm = pd.DataFrame({"tau": taus, "E_ST_ou": Eou, "Var_ST_ou": Vou, "E_ST_jump": Ej, "Var_ST_jump": Vj,
                       "mc_mean": mcm, "mc_var": mcv})
    st = {"stationary_mean_ou": th, "stationary_var_ou": sg ** 2 / (2 * kap),
          "stationary_mean_jump": th + lam * mJ / kap,
          "stationary_var_jump": (sg ** 2 + lam * (mJ ** 2 + sJ ** 2)) / (2 * kap),
          "empirical_mean": float(v.mean()), "empirical_var": float(v.var(ddof=1))}
    ppy = (1.0 / dt) if periods_per_year is None else periods_per_year
    me = float(np.max(np.abs(mcm - Ej)))
    ve = float(np.max(np.abs(mcv / Vj - 1)) * 100)
    if mean_tol is None:
        se = float(np.max(np.sqrt(Vj / float(n_paths))))
        mean_tol = max(4.0 * se, 0.01 * float(np.sqrt(np.max(Vj))))
    if var_tol_pct is None:
        var_tol_pct = 5.0
    summ = {"kappa": kap, "theta": th, "sigma_ou": sg, "lambda_jump": lam, "mu_J": mJ, "sigma_J": sJ,
            "half_life_days": math.log(2) / kap * ppy, "n_jumps_detected": int(fit["n_jumps"]),
            "mc_validates": bool(me < mean_tol and ve < var_tol_pct), "max_mc_mean_error": me,
            "max_mc_var_error_pct": ve}
    return _rd82({"calibration": cal, "conditional_moments": cm, "stationary": st, "summary": summ, "fit": fit})


# ============================================================================ 6. funding-rate OU carry
def funding_carry_report(raw, params, columns=None) -> dict:
    """Perpetual funding-rate profile, regimes, plug-in Euler OU and exact-OU Monte-Carlo carry.

    raw columns (defaults): symbol, funding_time (epoch ms, or datetime already parsed), funding_rate (decimal).
    params keys: target_symbol, analysis_start, analysis_end (whole END DAY included), regime_split_date,
      annualization_periods A, ou_dt, mc_num_paths N, mc_horizon_periods H, mc_random_seed, var_confidence c.
    Cleaning: UTC datetime FLOORED to seconds; symbol filter; start 00:00 <= t < end + 1 day; sort;
      duplicate timestamps keep last; funding_rate_bps = rate * 1e4; datetime text '%Y-%m-%dT%H:%M:%SZ'.
    Stats: mean, std ddof=1, pandas skew()/kurt() (unbiased, Fisher), min, max, median,
      pct_positive = 100 mean(rate > 0), annualized_mean = mean A, annualized_volatility = std sqrt(A),
      autocorr lag L = Series.autocorr(L) for 1, 3, 6, 12, 24; adfuller(maxlag=24, regression='c', autolag='AIC').
    Regimes: period1 = t < split ('bear_<year of first obs>'), period2 = t >= split ('recovery_<year>').
    OU (ou_fit_euler_plugin): dF = a + b F_{t-1} + e; kappa = -b/dt, mu = -a/b, sigma = std(e, ddof=1)/sqrt(dt);
      half_life_periods = ln2/kappa/dt; half_life_days = half_life_periods/(A/365); stationary_std = sigma/sqrt(2 kappa).
    MC: rng = default_rng(seed); Z = rng.standard_normal((N, H)) ONCE; f_0 = last observed rate;
      f_{t+1} = mu + (f_t - mu)e^{-kappa dt} + sigma sqrt((1 - e^{-2 kappa dt})/(2 kappa)) Z[:, t];
      carry = sum of the H SIMULATED rates (f_0 excluded); annual = carry A/H; losses = sort(-annual),
      VaR = max(losses[ceil(c N) - 1], 0), CVaR = mean(losses > VaR) (VaR if none), pct_negative = 100 mean(annual < 0),
      percentiles np.percentile 5/25/50/75/95; per-path average rate = carry/H.
    Returns the six deliverables ready for emit (DataFrames for the two CSVs)."""
    from statsmodels.tsa.stattools import adfuller
    from .helpers75_3 import ou_fit_euler_plugin
    c = {"symbol": "symbol", "time": "funding_time", "rate": "funding_rate"}
    c.update(columns or {})
    df = raw.copy()
    tcol = df[c["time"]]
    if pd.api.types.is_datetime64_any_dtype(tcol):
        dtv = pd.to_datetime(tcol, utc=True)
    else:
        dtv = pd.to_datetime(pd.to_numeric(tcol), unit="ms", utc=True)
    df["datetime"] = dtv.dt.floor("s")
    df["funding_rate"] = pd.to_numeric(df[c["rate"]], errors="coerce")
    df["symbol"] = df[c["symbol"]].astype(str)
    start = pd.Timestamp(params["analysis_start"], tz="UTC")
    end_excl = pd.Timestamp(params["analysis_end"], tz="UTC").normalize() + pd.Timedelta(days=1)
    df = df[(df["symbol"] == str(params["target_symbol"])) & (df["datetime"] >= start) & (df["datetime"] < end_excl)]
    df = df.dropna(subset=["funding_rate"]).sort_values("datetime", kind="mergesort")
    df = df.drop_duplicates("datetime", keep="last").reset_index(drop=True)
    df["funding_rate_bps"] = df["funding_rate"] * 1e4
    s = df["funding_rate"]
    A = float(params["annualization_periods"])
    dt = float(params["ou_dt"])
    fmt = "%Y-%m-%dT%H:%M:%SZ"

    def stats(x):
        return {"mean": float(x.mean()), "std": float(x.std(ddof=1)), "skewness": float(x.skew()),
                "kurtosis": float(x.kurt()), "min": float(x.min()), "max": float(x.max()),
                "median": float(x.median()), "pct_positive": float((x > 0).mean() * 100)}

    def adf(x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = adfuller(x.to_numpy(), maxlag=24, regression="c", autolag="AIC")
        return float(a[0]), float(a[1])

    desc = stats(s)
    desc["annualized_mean"] = desc["mean"] * A
    desc["annualized_volatility"] = desc["std"] * math.sqrt(A)
    ac = {f"lag_{L}": float(s.autocorr(L)) for L in (1, 3, 6, 12, 24)}
    a_stat, a_p = adf(s)
    split = pd.Timestamp(params["regime_split_date"], tz="UTC")
    per = {}
    rrows = []
    ou_sub = {}
    for key, msk, lab in (("period1", df["datetime"] < split, "bear"), ("period2", df["datetime"] >= split, "recovery")):
        sub = df[msk]
        x = sub["funding_rate"].reset_index(drop=True)
        st = stats(x)
        o = ou_fit_euler_plugin(x.to_numpy(), dt)
        ou_sub[key] = o
        ast, ap = adf(x)
        label = f"{lab}_{sub['datetime'].iloc[0].year}"
        d = {"label": label, "mean": st["mean"], "std": st["std"], "skewness": st["skewness"],
             "kurtosis": st["kurtosis"], "pct_positive": st["pct_positive"], "autocorr_lag1": float(x.autocorr(1)),
             "adf_statistic": ast, "adf_p_value": ap, "ou_kappa": o["kappa"], "ou_sigma": o["sigma"]}
        per[key] = d
        rrows.append({"period": key, "label": label, "start_date": sub["datetime"].iloc[0].strftime(fmt),
                      "end_date": sub["datetime"].iloc[-1].strftime(fmt), "num_obs": int(len(sub)),
                      **{k: d[k] for k in ("mean", "std", "skewness", "kurtosis", "pct_positive", "autocorr_lag1",
                                           "adf_statistic", "adf_p_value", "ou_kappa", "ou_sigma")}})
    per["mean_ratio_p2_over_p1"] = per["period2"]["mean"] / per["period1"]["mean"]
    per["autocorr_diff"] = per["period2"]["autocorr_lag1"] - per["period1"]["autocorr_lag1"]
    o = ou_fit_euler_plugin(s.to_numpy(), dt)
    ppd = A / 365.0
    ok = o["kappa"] > 0
    hlp = o["half_life_periods"]
    ou = {"kappa": o["kappa"], "mu": o["mu"], "sigma": o["sigma"], "half_life_periods": _f(hlp),
          "half_life_days": _f(hlp / ppd) if ok else None, "stationary_std": _f(o["stationary_std"]),
          "ols_intercept": o["alpha"], "ols_slope": o["beta"], "residual_std": o["resid_std"],
          "num_observations_fit": int(o["nobs"])}
    N, H, seed = int(params["mc_num_paths"]), int(params["mc_horizon_periods"]), int(params["mc_random_seed"])
    cl = float(params["var_confidence"])
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((N, H))
    f0 = float(s.iloc[-1])
    kap, mu, sig = o["kappa"], o["mu"], o["sigma"]
    e = math.exp(-kap * dt)
    sd = sig * math.sqrt((1 - e * e) / (2 * kap))
    f = np.full(N, f0)
    cum = np.zeros(N)
    for t in range(H):
        f = mu + (f - mu) * e + sd * Z[:, t]
        cum += f
    ann = cum * A / H
    avg = cum / H
    L = np.sort(-ann)
    i = min(max(int(math.ceil(cl * N)) - 1, 0), N - 1)
    var = max(float(L[i]), 0.0)
    worse = L[L > var]
    cvar = float(worse.mean()) if len(worse) else var
    pc = {f"p{p}": float(np.percentile(ann, p)) for p in (5, 25, 50, 75, 95)}
    pctneg = float((ann < 0).mean() * 100)
    mcb = {"mean_annual_return": float(ann.mean()), "std_annual_return": float(ann.std(ddof=1)), "var_95": var,
           "cvar_95": cvar, "pct_negative_return": pctneg, "percentiles": pc, "mc_num_paths": N, "mc_seed": seed,
           "initial_funding_rate": f0}
    results = {"symbol": str(params["target_symbol"]), "num_observations": int(len(df)),
               "date_range": {"start": df["datetime"].iloc[0].strftime(fmt), "end": df["datetime"].iloc[-1].strftime(fmt)},
               "descriptive_stats": desc, "autocorrelation": ac,
               "adf_test": {"statistic": a_stat, "p_value": a_p, "is_stationary": bool(a_p < 0.05)},
               "regime_analysis": per, "ou_parameters": ou, "mc_basis_carry": mcb}
    inter = {"num_observations": int(len(df)), "mean_funding_rate": desc["mean"], "std_funding_rate": desc["std"],
             "annualized_mean": desc["annualized_mean"], "adf_statistic": a_stat,
             "regime1_mean": per["period1"]["mean"], "regime2_mean": per["period2"]["mean"],
             "ou_kappa": ou["kappa"], "ou_mu": ou["mu"], "ou_sigma": ou["sigma"],
             "half_life_periods": ou["half_life_periods"], "half_life_days": ou["half_life_days"],
             "stationary_std": ou["stationary_std"], "mc_mean_annual_return": mcb["mean_annual_return"],
             "mc_var_95": var, "mc_cvar_95": cvar, "mc_pct_negative": pctneg}
    clean = pd.DataFrame({"datetime": df["datetime"].dt.strftime(fmt), "symbol": df["symbol"],
                          "funding_rate": df["funding_rate"], "funding_rate_bps": df["funding_rate_bps"]})
    diag = {"full_sample": {"kappa": o["kappa"], "mu": o["mu"], "sigma": o["sigma"], "ols_intercept": o["alpha"],
                            "ols_slope": o["beta"], "ols_r_squared": o["r_squared"], "residual_std": o["resid_std"],
                            "num_observations": int(o["nobs"])},
            "period1_bear": {"kappa": ou_sub["period1"]["kappa"], "sigma": ou_sub["period1"]["sigma"],
                             "num_observations": int(ou_sub["period1"]["nobs"])},
            "period2_recovery": {"kappa": ou_sub["period2"]["kappa"], "sigma": ou_sub["period2"]["sigma"],
                                 "num_observations": int(ou_sub["period2"]["nobs"])}}
    mcs = {"mc_num_paths": N, "mc_random_seed": seed, "mc_horizon_periods": H, "initial_funding_rate": f0,
           "ou_params_used": {"kappa": kap, "mu": mu, "sigma": sig},
           "simulated_funding_rate_distribution": {"mean": float(avg.mean()), "std": float(avg.std(ddof=1)),
                                                   "p5": float(np.percentile(avg, 5)), "p50": float(np.percentile(avg, 50)),
                                                   "p95": float(np.percentile(avg, 95))},
           "annual_carry_return_distribution": {"mean": mcb["mean_annual_return"], "std": mcb["std_annual_return"],
                                                "var_95": var, "cvar_95": cvar, "pct_negative": pctneg, **pc}}
    return {"results": results, "solution": {"intermediates": {k: {"value": v} for k, v in inter.items()}},
            "funding_clean": clean, "regime_stats": pd.DataFrame(rrows), "ou_diagnostics": diag, "mc_stats": mcs}


# ============================================================================ 7. Kelly with VaR cap
def _clean_returns(raw, cols, keep="last"):
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date", kind="mergesort")
    df = df.drop_duplicates("date", keep=keep)
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=cols).reset_index(drop=True)


def _warn_prewindowed(raw, n_valid, w_days):
    """know77: a run cleaned AND cut the table to the window before calling the report, so the count
    'before the window' became the window length. Flag input that already looks windowed."""
    if w_days and int(n_valid) == int(w_days) and len(raw) == int(n_valid):
        msg = ("NOTE: the table passed in already has exactly estimation_window_days clean rows - pass the RAW "
               "INPUTS table (num_valid_observations counts the rows AFTER cleaning, BEFORE the window cut)")
        warnings.warn(msg)
        print(msg, flush=True)


def kelly_var_sizing_report(raw, params, asset_prefix="asset_", periods_per_year=252) -> dict:
    """Multi-asset Kelly with a parametric VaR cap and a shared Monte-Carlo bankroll.

    Cleaning: parse/sort dates, drop duplicate dates (keep last), to_numeric, inf -> NaN, drop rows with
      any NaN; num_valid_observations = rows AFTER cleaning, BEFORE the window cut; estimation sample =
      last estimation_window_days rows when set.
    mu = mean(R) (raw), rf_d = risk_free_annual/A (simple), mean_excess = mu - rf_d, Sigma = cov(ddof=1),
      f_full = solve(Sigma, mean_excess), total_kelly_leverage = sum |f|.
    VaR(w) = z_c sqrt(w' Sigma w) - w' mu (RAW mean, positive loss); var_scale_factor = min(1, max_var_daily/VaR(f_full)).
    Schemes full f, half f/2, var s f, equal 1/n. Draws = default_rng(int(seed)).multivariate_normal(mu, Sigma,
      size=(paths, days)) shared by every scheme; r_p = draws @ w (NO cash leg); W = W0 cumprod(1 + r_p);
      Sharpe = mean/std(ddof=1) sqrt(A) POOLED over all paths and days (no rf); max drawdown per path on
      [W0, W] positive; p50 = median over paths; median / mean terminal wealth.
    Returns {results (7 keys), intermediates {name: {'value'}} (13 keys), fractions, schemes}."""
    cols = [c for c in raw.columns if str(c).startswith(asset_prefix)]
    df = _clean_returns(raw, cols)
    n_valid = int(len(df))
    w_days = params.get("estimation_window_days")
    _warn_prewindowed(raw, n_valid, w_days)
    est = df.tail(int(w_days)) if w_days else df
    R = est[cols].to_numpy(dtype=float)
    mu = R.mean(axis=0)
    rf_d = float(params["risk_free_annual"]) / periods_per_year
    ex = mu - rf_d
    S = np.cov(R, rowvar=False, ddof=1)
    f = np.linalg.solve(S, ex)
    c = float(params["confidence_level"])
    z = float(_norm.ppf(c / 100 if c > 1 else c))

    def var(w):
        return z * math.sqrt(float(w @ S @ w)) - float(w @ mu)

    v_full = var(f)
    scale = min(1.0, float(params["max_var_daily"]) / v_full) if v_full > 0 else 1.0
    n = len(cols)
    schemes = {"full": f, "half": f / 2.0, "var": scale * f, "equal": np.full(n, 1.0 / n)}
    rng = np.random.default_rng(int(params["seed"]))
    draws = rng.multivariate_normal(mu, S, size=(int(params["n_simulation_paths"]), int(params["n_days"])))
    W0 = float(params["initial_capital"])
    out = {}
    for name, w in schemes.items():
        rp = draws @ w
        W = W0 * np.cumprod(1.0 + rp, axis=1)
        Wf = np.concatenate([np.full((W.shape[0], 1), W0), W], axis=1)
        peak = np.maximum.accumulate(Wf, axis=1)
        mdd = np.max((peak - Wf) / peak, axis=1)
        flat = rp.ravel()
        out[name] = {"median_terminal": float(np.median(W[:, -1])), "mean_terminal": float(np.mean(W[:, -1])),
                     "sharpe": float(flat.mean() / flat.std(ddof=1) * math.sqrt(periods_per_year)),
                     "p50_drawdown": float(np.median(mdd)), "weights": w}
    res = {"num_valid_observations": n_valid, "median_terminal_wealth_full": out["full"]["median_terminal"],
           "median_terminal_wealth_var": out["var"]["median_terminal"], "sharpe_full": out["full"]["sharpe"],
           "sharpe_var": out["var"]["sharpe"], "p50_drawdown_full": out["full"]["p50_drawdown"],
           "p50_drawdown_var": out["var"]["p50_drawdown"]}
    inter = {f"mean_excess_{i}": float(ex[i]) for i in range(n)}
    inter.update({"cov_00": float(S[0, 0]), "cov_01": float(S[0, 1]) if n > 1 else None,
                  "cov_12": float(S[1, 2]) if n > 2 else None})
    inter.update({f"kelly_fraction_{i}": float(f[i]) for i in range(n)})
    inter.update({"total_kelly_leverage": float(np.abs(f).sum()), "portfolio_var_full": float(v_full),
                  "var_scale_factor": float(scale), "mean_terminal_wealth_full": out["full"]["mean_terminal"]})
    return {"results": res, "intermediates": {k: {"value": v} for k, v in inter.items()}, "fractions": f,
            "schemes": out}


# ============================================================================ 8. regime CTA vol targeting
def cta_regime_report(prices, params, periods_per_year=252, garch_mean=None, garch_scale=100.0,
                      first_row="zero") -> dict:
    """EMA-trend CTA, EWMA vol sizing, GARCH composite-vol regimes, static vs dynamic target.

    r = diff(ln P) (T-1 rows, return dates = price dates[1:]). Per asset: fast/slow SMA-seeded EMA on PRICES,
      signal = sign(fast - slow) aligned with [1:] (NaN -> weight 0); EWMA variance on returns
      (alpha = 2/(vol_lookback+1), s2[0] = r[0]^2), ewma_vol = sqrt(A s2).
    GARCH(1,1) per asset: arch on 100 r, mean garch_mean (see know80 below), normal -> daily DECIMAL sigma_i[t]; persistence
      alpha+beta; long-run variance omega/(1-alpha-beta) (decimal^2/day); averages over assets.
    composite[t] = sqrt(mean_i sigma_i[t]^2) sqrt(A) (RMS of variances); thresholds np.percentile(lo/hi);
      low <= lo, high >= hi, else mid; scale = scale_low_vol / 1 / scale_high_vol (no extra lag).
    w = clip(signal (vol_target/n_assets) scale / ewma_vol, +-max_leverage_per_asset), NaN -> 0 (static: scale 1).
    x[t] = sum_i w[t-1, i] r[t, i], x[0] = 0 (the shift(1) row); metrics (log_return_performance, ddof=0):
      annualized_return = exp(mean A) - 1, annualized_volatility = std sqrt(A), sharpe = (mean A - rf)/vol,
      max_drawdown = min(exp(cumsum x)/cummax - 1) (<= 0), calmar = ann_return/|mdd|.
    results: dynamic_sharpe_ratio, sharpe_improvement = dynamic - static, regime_frac_high, composite_vol_mean.
    know81 first_row="zero" (default again): the first return date has no prior weight; under the task's own
      rule "Replace NaN values with 0" that lagged weight is 0, so x[0] = 0 stays in the series like every other
      no-position (warm-up) day - convention 'backtest_first_return' (positions lagged with shift(1), missing
      positions earn 0: Chan, Quantitative Trading, ch.3). "drop" (x[1:]) only when the task starts the series at
      the second return. Warm-up rows with weight 0 stay in either way.
    garch_mean=None (default): read from the task text (helpers80_3.garch_mean_rule, convention 'garch_mean') -
      the mean model the task names; silent -> the constant-mean equation (Bollerslev 1986; arch default), zero
      mean only for a daily VaR / risk-capital model."""
    if garch_mean is None:
        garch_mean = "Zero"
    from .helpers75_3 import (sma_seeded_ema, ewma_variance_alpha, garch11_conditional_vol, percentile_regimes,
                              vol_target_weights, lagged_portfolio_returns, log_return_performance)
    df = prices.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    cols = [c for c in df.columns if str(c).startswith("asset_")] or [c for c in df.columns if c != "date"]
    P = df[cols].apply(pd.to_numeric, errors="coerce").ffill().to_numpy(dtype=float)
    R = np.diff(np.log(P), axis=0)
    A = periods_per_year
    fast = sma_seeded_ema(P, int(params["fast_ema"]))[1:]
    slow = sma_seeded_ema(P, int(params["slow_ema"]))[1:]
    sig = np.sign(fast - slow)
    vol = np.sqrt(A * ewma_variance_alpha(R, span=int(params["vol_lookback"])))
    gs = [garch11_conditional_vol(R[:, j], mean=garch_mean, scale=garch_scale) for j in range(R.shape[1])]
    sig_d = np.column_stack([g["sigma"] for g in gs])
    comp = np.sqrt(np.mean(sig_d ** 2, axis=1)) * math.sqrt(A)
    reg = percentile_regimes(comp, params["regime_lo_pct"], params["regime_hi_pct"])
    codes = reg["codes"]
    scale = np.where(codes == 0, float(params["scale_low_vol"]), np.where(codes == 2, float(params["scale_high_vol"]), 1.0))
    scale_aligned = np.r_[scale[1:], 1.0]
    n_assets = int(params.get("n_assets", R.shape[1]))
    tgt = float(params["vol_target"]) / n_assets
    L = float(params["max_leverage_per_asset"])
    rf = float(params["risk_free_annual"])
    out = {}
    for name, sc in (("static", None), ("dynamic", scale_aligned)):
        w = vol_target_weights(np.nan_to_num(sig, nan=0.0), vol, tgt, scale=sc, max_leverage=L)
        x = lagged_portfolio_returns(w, R, lag=1)
        if first_row == "drop":
            x = x[1:]
        m = log_return_performance(x, periods_per_year=A, rf_annual=rf, ddof=1)
        ann_ret = float(x.mean() * A)
        ann_vol = float(x.std(ddof=1) * math.sqrt(A))
        mdd_pos = float(abs(m["max_drawdown"]))
        m["annualized_return"] = ann_ret
        m["annualized_volatility"] = ann_vol
        m["sharpe"] = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0
        m["max_drawdown"] = mdd_pos
        m["calmar"] = ann_ret / mdd_pos if mdd_pos > 0 else 0.0
        m["returns"] = x
        out[name] = m
    res = {"dynamic_sharpe_ratio": out["dynamic"]["sharpe"],
           "sharpe_improvement": out["dynamic"]["sharpe"] - out["static"]["sharpe"],
           "regime_frac_high": reg["frac_high"], "composite_vol_mean": float(comp.mean())}
    inter = {}
    for name in ("static", "dynamic"):
        m = out[name]
        inter[f"{name}_annualized_return"] = m["annualized_return"]
        inter[f"{name}_annualized_volatility"] = m["annualized_volatility"]
        if name == "static":
            inter["static_sharpe_ratio"] = m["sharpe"]
        inter[f"{name}_max_drawdown"] = m["max_drawdown"]
        inter[f"{name}_calmar_ratio"] = m["calmar"]
    inter.update({"avg_garch_persistence": float(np.mean([g["persistence"] for g in gs])),
                  "avg_long_run_variance": float(np.mean([g["long_run_variance"] for g in gs])),
                  "regime_frac_low": reg["frac_low"], "regime_frac_mid": reg["frac_mid"],
                  "regime_thresh_lo": reg["thresh_lo"], "regime_thresh_hi": reg["thresh_hi"]})
    return {"results": {k: _f(v) for k, v in res.items()},
            "intermediates": {k: {"value": _f(v)} for k, v in inter.items()}, "composite_vol": comp,
            "regime_codes": codes, "static": out["static"], "dynamic": out["dynamic"]}


# ============================================================================ 9. absorption-ratio risk parity
def absorption_riskparity_report(raw, params, periods_per_year=252, var_level=None, final_partial_month=None) -> dict:
    """Absorption-ratio regimes, regime-scaled inverse-vol weights at month ends, CVaR - ONE index space.

    Cleaning: sort dates, drop duplicate dates (keep last), to_numeric, inf -> NaN, drop NaN rows ->
      num_valid_observations (BEFORE the window cut); keep the last estimation_window_days rows.
    Every quantity below is indexed by the ROW t of that windowed matrix R (T x N):
      window t = rows t-W+1..t (t = W-1..T-1); C = corrcoef; eigvalsh descending; AR = sum(top k)/sum(all);
      largest eigenvalue; regime crisis if AR > crisis_thr, risk-off if AR > risk_off_thr, else risk-on (strict >).
      rebalance rows = window-end rows t that are the last row of their calendar month (the final row closes
      its month); weights = (1/sd)/sum(1/sd) * risk_budget[regime_t], sd = std of window t (ddof=1).
      strategy r_p[t] = w(last rebalance row < t) . R[t] for t > first rebalance row.
    Metrics: ann_return = mean A, ann_vol = std(ddof=1) sqrt(A), sharpe = ret/vol, max_drawdown = max(1 - W/peak)
      on W = [1, cumprod(1 + r_p)] (positive); tail = last cvar_window r_p; q = np.quantile(tail, 1 - c);
      VaR = -q; CVaR = -mean(tail[tail <= q]) (c = cvar_confidence; portfolio_var_99 uses var_level, default c);
      win_rate = mean(r_p > 0); mean_weight_<first asset> = mean of that weight over rebalances.
    final_partial_month=False drops the last row as a rebalance when the sample ends before its month does
      (a weight set that never earns a return); know77 keyword. know79 None (default): read from the task
      text (helpers79_3.month_end_sample_rule). know81: 'last trading day of each calendar month' is the last
      OBSERVED row of each month of the sample, the final month included -> True; False only for 'complete
      months only' / 'drop the partial final month' wording (convention 'month_end_rebalance'). Only
      num_rebalances / mean weight move; the return path is the same."""
    if final_partial_month is None:
        from .helpers79_3 import month_end_sample_rule
        final_partial_month = month_end_sample_rule()
    cols = [c for c in raw.columns if c != "date"]
    df = _clean_returns(raw, cols)
    n_valid = int(len(df))
    ew = params.get("estimation_window_days")
    _warn_prewindowed(raw, n_valid, ew)
    if ew:
        df = df.tail(int(ew)).reset_index(drop=True)
    R = df[cols].to_numpy(dtype=float)
    T, N = R.shape
    W = int(params["rolling_window"])
    k = int(params["top_k_eigenvalues"])
    hi, lo = float(params["regime_crisis_threshold"]), float(params["regime_risk_off_threshold"])
    budget = params["risk_budget"]
    ends = np.arange(W, T)
    ar, big, regime = [], [], []
    for t in ends:
        lam = np.sort(np.linalg.eigvalsh(np.corrcoef(R[t - W:t], rowvar=False)))[::-1]
        a = float(lam[:k].sum() / lam.sum())
        ar.append(a)
        big.append(float(lam[0]))
        regime.append("crisis" if a > hi else ("risk-off" if a > lo else "risk-on"))
    ar, big = np.array(ar), np.array(big)
    months = df["date"].dt.to_period("M").to_numpy()
    reb_idx = [i for i, t in enumerate(ends) if t < T - 1 and months[t] != months[t + 1]]
    reb = [int(ends[i]) for i in reb_idx]
    reb_set = set(reb_idx)
    wts = []
    w_cur = np.ones(N, dtype=float) / N
    rp = np.zeros(len(ends), dtype=float)
    for i, t in enumerate(ends):
        rp[i] = float(R[t] @ w_cur)
        if i in reb_set:
            sd = R[t - W:t].std(axis=0, ddof=1)
            iv = 1.0 / sd
            w_cur = (iv / iv.sum()) * float(budget[regime[i]])
            wts.append(w_cur.copy())
    wts = np.array(wts)
    A = periods_per_year
    ret = float(rp.mean() * A)
    vol = float(rp.std(ddof=1) * math.sqrt(A))
    Wl = np.concatenate([[1.0], np.cumprod(1.0 + rp)])
    mdd = float(np.max(1.0 - Wl / np.maximum.accumulate(Wl)))
    c = float(params["cvar_confidence"])
    tail = rp[-int(params["cvar_window"]):]
    q = float(np.quantile(tail, 1.0 - c))
    cvar = float(-tail[tail <= q].mean())
    vl = c if var_level is None else var_level
    var99 = float(-np.quantile(tail, 1.0 - vl))
    counts = {lab: int(sum(1 for g in regime if g == lab)) for lab in ("crisis", "risk-off", "risk-on")}
    res = {"num_valid_observations": n_valid, "annualized_return": ret, "annualized_volatility": vol,
           "sharpe_ratio": ret / vol if vol > 0 else None, "max_drawdown": mdd, "cvar": cvar}
    first = str(cols[0]).lower()
    inter = {"num_regime_observations": int(len(ends)), "mp_threshold": (1.0 + math.sqrt(N / W)) ** 2,
             "mean_absorption_ratio": float(ar.mean()), "std_absorption_ratio": float(ar.std(ddof=1)),
             "mean_largest_eigenvalue": float(big.mean()), "regime_count_crisis": counts["crisis"],
             "regime_count_risk_off": counts["risk-off"], "regime_count_risk_on": counts["risk-on"],
             "num_rebalances": int(len(reb)), f"mean_weight_{first}": float(wts[:, 0].mean()),
             "portfolio_var_99": var99, "cvar": cvar, "win_rate": float((rp > 0).mean())}
    return {"results": res, "intermediates": {k2: {"value": v} for k2, v in inter.items()},
            "rebalance_dates": df["date"].iloc[reb].dt.strftime("%Y-%m-%d").tolist(), "weights": wts,
            "strategy_returns": rp, "absorption_ratio": ar, "regimes": regime}


# ============================================================================ 10. participation TCA
def tca_participation_report(quotes, trades, order_spec, header, checkpoint_buckets, out_dir=None, decimals=6) -> dict:
    """-> {results, fills (TEXT cells: '%.6f' floats, 13-digit ms, true/false, '' - emit as is), fills_raw, solution}.

    know77: `fills` is already formatted to the CSV rules (an emitted numeric frame wrote True/False and
    0.0, so an empty-bucket check found no 'true' row); numeric values are in fills_raw.
    Participation-capped bucket fills + TCA metrics, packaged as results / fills / solution.

    Engine: helpers75_4.participation_bucket_fills (half-open buckets on trade time, child = min(remaining,
      cap * bucket volume, max_child), fill at ask (buy) / bid (sell) of the FIRST quote with
      transaction_time >= bucket start, arrival = first quote >= start, future = first quote >= end + horizon).
    order_spec keys: start_time_ms, end_time_ms, bucket_ms, target_quantity, participation_cap,
      max_child_quantity, side, realized_horizon_ms.
    results: `header` (dict, e.g. schema_version, task_slug, symbol, side - in the task's order) followed by
      the metric keys in the conventional order (start_time_ms ... bucket_count_empty); floats rounded to
      `decimals`, ints stay ints. fills: DataFrame in the fixed column order (empty buckets kept, '' fields).
    solution: {'intermediates': {name: {'value'}}, 'checkpoints': {'bucket_<i>': {bucket_start_ms,
      eligible_trade_count, eligible_trade_qty, child_order_qty, remaining_qty, is_empty_bucket}}} for
      i in checkpoint_buckets.
    out_dir: write results.json, fills.csv (write_fixed_csv: '%.6f' floats, 13-digit ms ints, true/false,
      '' for missing) and solution.json."""
    from .helpers75_4 import participation_bucket_fills, write_fixed_csv
    o = order_spec
    r = participation_bucket_fills(quotes, trades, o["start_time_ms"], o["end_time_ms"], o["bucket_ms"],
                                   o["target_quantity"], o["participation_cap"], o["max_child_quantity"],
                                   side=o.get("side", "buy"), realized_horizon_ms=o.get("realized_horizon_ms", 0),
                                   header=None)

    def rnd(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return round(float(v), decimals) if math.isfinite(v) else None
        return v

    res = dict(header or {})
    for kk, vv in r["metrics"].items():
        res[kk] = rnd(vv)
    fills = r["fills"]
    from .helpers77_1 import fixed_text_frame
    fills_text = fixed_text_frame(fills, decimals=decimals)
    cps = {}
    for i in checkpoint_buckets:
        row = fills.iloc[int(i)]
        cps[f"bucket_{int(i)}"] = {
            "bucket_start_ms": {"value": int(row["bucket_start_ms"])},
            "eligible_trade_count": {"value": int(row["eligible_trade_count"])},
            "eligible_trade_qty": {"value": rnd(float(row["eligible_trade_qty"]))},
            "child_order_qty": {"value": rnd(float(row["child_order_qty"]))},
            "remaining_qty": {"value": rnd(float(row["remaining_qty"]))},
            "is_empty_bucket": {"value": bool(row["is_empty_bucket"])}}
    sol = {"intermediates": {kk: {"value": rnd(vv)} for kk, vv in r["intermediates"].items()}, "checkpoints": cps}
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        write_fixed_csv(fills, os.path.join(out_dir, "fills.csv"), decimals=decimals)
        with open(os.path.join(out_dir, "solution.json"), "w", encoding="utf-8") as fh:
            json.dump(sol, fh, indent=2)
    return {"results": res, "fills": fills_text, "fills_raw": fills, "solution": sol}


# ============================================================================ 11. intraday volume profiles
_PROFILE_MODELS = ("historical_mean_profile", "historical_median_profile", "ewma_profile", "winsorized_mean_profile")


def intraday_volume_schedule(raw, order_datetime, quantity, window, half_life, winsor_pct, session_start="09:30:00",
                             session_end="16:00:00", last_bar_start="15:55:00", bar_minutes=5, bars_per_day=None,
                             models=_PROFILE_MODELS) -> dict:
    """Walk-forward intraday volume-share model selection and an integer schedule after the order time.

    raw: columns datetime (bar START), Open, High, Low, Close, Volume. Steps: sort; keep raw rows with
      session_start <= time <= session_end; resample `bar_minutes` bars (label='left', closed='left'; first/max/
      min/last/sum); drop empty bars; keep bar starts in [session_start, last_bar_start]; complete day = exactly
      bars_per_day bars (default = number of slots between session_start and last_bar_start); excluded days =
      the other days present. Shares matrix S [days x bars] = V / daily total, columns = bar POSITION in the day.
    Test day i >= window: profile from rows i-window..i-1 (helpers75_4.volume_share_profile: mean, median,
      ewma with lag 0 = most recent day and w_L = exp(-ln2 L/half_life), winsorized with winsor_pct in PERCENT
      (e.g. (5, 95)), each normalised); R2_d = 1 - SSE/SST (zero SST: 1.0 if equal else 0.0); avg_r2 per model;
      best = max avg_r2, ties by the order of `models`.
    Schedule: window = the `window` complete days strictly before the order's calendar day; best-model profile;
      bars of the order day's full grid whose start > order_datetime (STRICT); renormalise; floor(Q w) plus the
      leftover units to the largest remainders (ties -> earlier bar); cumulative sum.
    Returns {excluded_days (date 'YYYY-MM-DD'), model_performance (model_name, avg_r2, num_test_days),
      best_model {best_model, best_avg_r2, num_test_days}, final_schedule (datetime 'YYYY-MM-DD HH:MM:SS',
      scheduled_quantity, predicted_bar_share, cumulative_scheduled_quantity)}."""
    from .helpers75_4 import volume_share_profile, share_r2, largest_remainder_allocation
    df = raw.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime", kind="mergesort")
    tt = df["datetime"].dt.strftime("%H:%M:%S")
    df = df[(tt >= session_start) & (tt <= session_end)]
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    agg = {k: v for k, v in agg.items() if k in df.columns}
    bars = df.set_index("datetime").resample(f"{int(bar_minutes)}min", label="left", closed="left").agg(agg)
    bars = bars.dropna(subset=["Open"] if "Open" in bars.columns else None).reset_index()
    bt = bars["datetime"].dt.strftime("%H:%M:%S")
    bars = bars[(bt >= session_start) & (bt <= last_bar_start)].copy()
    t0 = pd.Timestamp("2000-01-01 " + session_start)
    t1 = pd.Timestamp("2000-01-01 " + last_bar_start)
    n_slots = int((t1 - t0) / pd.Timedelta(minutes=int(bar_minutes))) + 1
    T = int(bars_per_day) if bars_per_day is not None else n_slots
    bars["date"] = bars["datetime"].dt.normalize()
    cnt = bars.groupby("date").size()
    complete = sorted(cnt[cnt == T].index)
    excluded = sorted(cnt[cnt != T].index)
    cb = bars[bars["date"].isin(complete)].sort_values("datetime")
    vol = cb["Volume"].to_numpy(dtype=float).reshape(len(complete), T)
    S = vol / vol.sum(axis=1, keepdims=True)
    W = int(window)
    scores = {m: [] for m in models}
    for i in range(W, S.shape[0]):
        est = S[i - W:i]
        for m in models:
            scores[m].append(share_r2(S[i], volume_share_profile(est, m, half_life, winsor_pct)))
    perf = pd.DataFrame({"model_name": list(models), "avg_r2": [float(np.mean(scores[m])) for m in models],
                         "num_test_days": [int(len(scores[m])) for m in models]})
    best_i = 0
    for j in range(1, len(models)):
        if perf["avg_r2"].iloc[j] > perf["avg_r2"].iloc[best_i]:
            best_i = j
    best = models[best_i]
    od = pd.Timestamp(order_datetime)
    day = od.normalize()
    prior = [i for i, d in enumerate(complete) if d < day][-W:]
    prof = volume_share_profile(S[prior], best, half_life, winsor_pct)
    prof = np.clip(prof, 0.0, None)
    prof = prof / prof.sum()
    grid = pd.date_range(day + (t0 - t0.normalize()), periods=T, freq=f"{int(bar_minutes)}min")
    rem = np.where(grid > od)[0]
    w = prof[rem] / prof[rem].sum()
    q = largest_remainder_allocation(int(quantity), w)
    sched = pd.DataFrame({"datetime": grid[rem].strftime("%Y-%m-%d %H:%M:%S"), "scheduled_quantity": q.astype(int),
                          "predicted_bar_share": w, "cumulative_scheduled_quantity": np.cumsum(q).astype(int)})
    return {"excluded_days": pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in excluded]}, columns=["date"]),
            "model_performance": perf,
            "best_model": {"best_model": best, "best_avg_r2": float(perf["avg_r2"].iloc[best_i]),
                           "num_test_days": int(perf["num_test_days"].iloc[best_i])},
            "final_schedule": sched, "shares": S, "complete_days": complete}


# ============================================================================ 12. American CN-PSOR report
def american_fd_report(S0, K, r, sigma, T, dividends, q=0.0, S_max=None, N_S=300, N_T=600, coarse=(150, 300),
                       omega=1.2, tol=1e-8, boundary_tol=1e-6, sample_S_every=5, sample_t_every=10, png_path=None) -> dict:
    """Eight CN prices (American/European x put/call, with and without cash dividends), boundary, delta,
    sampled grid, Richardson convergence and the summary - all from helpers75_3.cn_fd_option.

    dividends: list of {'time': t, 'amount': D} dicts or (time, amount) pairs (TIME FIRST here; this report
      converts them for cn_fd_option, whose tuples are (amount, time)). No-arbitrage cash dividend:
      V(S, t_d-) = V(max(S - D, 0), t_d+), American re-projected on the payoff.
    grid: rows = every sample_S_every-th stock node, column 'S' then one column per sample_t_every-th time
      node headed by the time value (str of the float); V(S, t) of the American put with dividends.
    boundary: t, S_star for all N_T + 1 time nodes (largest S_i with payoff > 0 and |V - payoff| < boundary_tol,
      0.0 when none; S*(T) = largest node below K). greeks: option_type, delta (central FD at S0, t = 0).
    convergence: fine / coarse / richardson = (4 fine - coarse)/3 (American put with dividends);
      convergence_ratio = |coarse - rich| / |fine - rich|.
    summary keys: american_geq_european_put/call, no_div_call_diff, early_exercise_premium_put/call,
      exercise_boundary_at_T, exercise_boundary_at_0, richardson_estimate, convergence_ratio."""
    from .helpers75_3 import cn_fd_option
    divs = []
    for d in dividends or ():
        if isinstance(d, dict):
            divs.append((float(d["amount"]), float(d["time"])))
        else:
            t_, a_ = d
            divs.append((float(a_), float(t_)))
    kw = dict(S_max=S_max, omega=omega, tol=tol, S0=S0, boundary_tol=boundary_tol, q=q)
    vals, res = {}, {}
    for name, kind, am, dv in (("american_put", "put", True, divs), ("american_call", "call", True, divs),
                               ("european_put", "put", False, divs), ("european_call", "call", False, divs),
                               ("american_put_no_div", "put", True, []), ("american_call_no_div", "call", True, []),
                               ("european_put_no_div", "put", False, []), ("european_call_no_div", "call", False, [])):
        out = cn_fd_option(K, T, r, sigma, kind=kind, american=am, N_S=N_S, N_T=N_T, dividends=dv, **kw)
        vals[name] = float(out["price"])
        res[name] = out
    ap = res["american_put"]
    V, Sg, tg = ap["V"], ap["S"], ap["t"]
    si = np.arange(0, len(Sg), int(sample_S_every))
    ti = np.arange(0, len(tg), int(sample_t_every))
    grid = pd.DataFrame(V[np.ix_(ti, si)].T, columns=[str(float(tg[j])) for j in ti])
    grid.insert(0, "S", Sg[si])
    bnd = pd.DataFrame({"t": tg, "S_star": ap["boundary"]})
    greeks = pd.DataFrame({"option_type": ["american_put", "american_call", "european_put", "european_call"],
                           "delta": [res[n]["delta"] for n in ("american_put", "american_call", "european_put", "european_call")]})
    co = cn_fd_option(K, T, r, sigma, kind="put", american=True, N_S=int(coarse[0]), N_T=int(coarse[1]), dividends=divs, **kw)
    pf, pc = vals["american_put"], float(co["price"])
    rich = (4 * pf - pc) / 3
    ratio = abs(pc - rich) / abs(pf - rich) if pf != rich else None
    conv = pd.DataFrame({"grid": ["fine", "coarse", "richardson"], "price": [pf, pc, rich]})
    summ = {"american_geq_european_put": bool(vals["american_put"] >= vals["european_put"] - 1e-10),
            "american_geq_european_call": bool(vals["american_call"] >= vals["european_call"] - 1e-10),
            "no_div_call_diff": abs(vals["american_call_no_div"] - vals["european_call_no_div"]),
            "early_exercise_premium_put": vals["american_put"] - vals["european_put"],
            "early_exercise_premium_call": vals["american_call"] - vals["european_call"],
            "exercise_boundary_at_T": float(ap["boundary"][-1]), "exercise_boundary_at_0": float(ap["boundary"][0]),
            "richardson_estimate": rich, "convergence_ratio": ratio}
    if png_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(tg, ap["boundary"])
        ax.set_xlabel("t")
        ax.set_ylabel("S*(t)")
        ax.set_title("Early exercise boundary (American put)")
        fig.savefig(png_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
    return {"option_values": vals, "grid": grid, "boundary": bnd, "greeks": greeks, "convergence": conv,
            "summary": summ, "runs": res}


# ============================================================================ 13. MC Greeks report
def mc_greeks_report(S0, K, r, q, sigma, T, n_paths, seed, n_avg, h_spot, h_vol, h_time, h_rate, spots, vols,
                     conv_sizes, rel_tol, lr_gamma_tol, parity_tols, conv_tol, png_path=None) -> dict:
    """Prices, three-method Greeks, consistency, BS surfaces, convergence and summary flags.

    Engine: helpers75_4.mc_greeks_suite (RandomState(seed): European normals first, then the Asian
      (N, n_avg) increments; FD with common random numbers; pathwise; likelihood ratio; None where a method
      does not exist). FD formulas literal: vega/rho per 1.00 of the parameter, theta = -(V(T-h) - V(T))/h.
    BS references with the same conventions: delta, gamma, vega, rho, theta = dV/dT (positive for a call).
    consistency: bs_errors {<greek>_<call|put>: |fd - BS|, price_<call|put>: |mc - BS|}, put_call_parity
      {delta_diff = d_c - d_p, gamma_diff, rho_diff} (FD), variance_ratio (pathwise vs LR European call delta),
      asian_vs_european {price_call, price_put, delta_call: Asian < European}.
    delta_surface / vega_surface: bs_delta_vega_surface(spots, vols) spot-major. convergence: for n in
      conv_sizes, the European call delta by fd / pathwise / lr on the FIRST n European normals.
    summary flags: mc_prices_match_bs (|mc - BS| <= 3 se), greeks_within_tolerance (fd delta, gamma, vega,
      theta, rho and pathwise/LR delta within rel_tol of BS, LR gamma within lr_gamma_tol), put_call_parity_*
      (|diff - theory| <= parity_tols['delta'|'gamma'|'rho']), pathwise_variance_lower, delta_surface_monotone,
      vega_positive, convergence_ok (|value - BS delta| <= conv_tol for every method at max(conv_sizes)),
      all_checks_passed."""
    from .helpers75_4 import mc_greeks_suite, bs_delta_vega_surface
    s = mc_greeks_suite(S0, K, r, q, sigma, T, n_paths, seed, n_avg, h_spot, h_vol, h_time, h_rate)
    vt = sigma * math.sqrt(T)
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / vt
    d2 = d1 - vt
    eq, er = math.exp(-q * T), math.exp(-r * T)
    bs = {"call": {"price": float(_bs(S0, K, T, r, q, sigma, True)), "delta": eq * _norm.cdf(d1),
                   "rho": K * T * er * _norm.cdf(d2)},
          "put": {"price": float(_bs(S0, K, T, r, q, sigma, False)), "delta": eq * (_norm.cdf(d1) - 1),
                  "rho": -K * T * er * _norm.cdf(-d2)}}
    gam = eq * _norm.pdf(d1) / (S0 * vt)
    vega = S0 * eq * _norm.pdf(d1) * math.sqrt(T)
    for side in ("call", "put"):
        bs[side]["gamma"] = gam
        bs[side]["vega"] = vega
    # theta as dV/dT (time-to-maturity derivative) = minus the calendar theta
    bs["call"]["theta"] = (S0 * eq * _norm.pdf(d1) * sigma / (2 * math.sqrt(T)) - q * S0 * eq * _norm.cdf(d1)
                           + r * K * er * _norm.cdf(d2))
    bs["put"]["theta"] = (S0 * eq * _norm.pdf(d1) * sigma / (2 * math.sqrt(T)) + q * S0 * eq * _norm.cdf(-d1)
                          - r * K * er * _norm.cdf(-d2))
    G = s["greeks"]
    P = s["prices"]
    errs = {}
    for side in ("call", "put"):
        opt = f"european_{side}"
        errs[f"price_{side}"] = abs(P[opt]["mc_price"] - bs[side]["price"])
        for g in ("delta", "gamma", "vega", "theta", "rho"):
            errs[f"{g}_{side}"] = abs(G[opt][g]["fd"]["value"] - bs[side][g])
    ec, ep = G["european_call"], G["european_put"]
    pcp = {"delta_diff": ec["delta"]["fd"]["value"] - ep["delta"]["fd"]["value"],
           "gamma_diff": ec["gamma"]["fd"]["value"] - ep["gamma"]["fd"]["value"],
           "rho_diff": ec["rho"]["fd"]["value"] - ep["rho"]["fd"]["value"]}
    ave = {"price_call": bool(P["asian_call"]["mc_price"] < P["european_call"]["mc_price"]),
           "price_put": bool(P["asian_put"]["mc_price"] < P["european_put"]["mc_price"]),
           "delta_call": bool(G["asian_call"]["delta"]["fd"]["value"] < ec["delta"]["fd"]["value"])}
    cons = {"bs_errors": errs, "put_call_parity": pcp, "variance_ratio": s["variance_ratio"], "asian_vs_european": ave}
    surf = bs_delta_vega_surface(list(spots), list(vols), K, T, r, q)
    dsurf = surf[["S0", "sigma", "delta_call_eu", "delta_put_eu"]].copy()
    vsurf = surf[["S0", "sigma", "vega_call_eu", "vega_put_eu"]].copy()
    rs = np.random.RandomState(seed)
    Ze = rs.standard_normal(int(n_paths))
    crow = []
    for n in conv_sizes:
        z = Ze[:int(n)]

        def disc_call(s0):
            st = s0 * np.exp((r - q - 0.5 * sigma ** 2) * T + vt * z)
            return er * np.maximum(st - K, 0.0)

        fd = (disc_call(S0 + h_spot) - disc_call(S0 - h_spot)) / (2 * h_spot)
        st = S0 * np.exp((r - q - 0.5 * sigma ** 2) * T + vt * z)
        pw = er * (st > K) * st / S0
        lr = er * np.maximum(st - K, 0.0) * z / (S0 * vt)
        for meth, x in (("fd", fd), ("pathwise", pw), ("lr", lr)):
            crow.append({"N": int(n), "method": meth, "greek": "delta", "value": float(x.mean()),
                         "se": float(x.std(ddof=1) / math.sqrt(len(x)))})
    conv = pd.DataFrame(crow, columns=["N", "method", "greek", "value", "se"])

    def rel(a, b):
        return abs(a - b) / abs(b) if b != 0 else abs(a)

    ok_price = all(abs(P[f"european_{sd}"]["mc_price"] - bs[sd]["price"]) <= 3 * P[f"european_{sd}"]["mc_se"]
                   for sd in ("call", "put"))
    ok_g = True
    for sd in ("call", "put"):
        opt = f"european_{sd}"
        for g in ("delta", "gamma", "vega", "theta", "rho"):
            ok_g &= rel(G[opt][g]["fd"]["value"], bs[sd][g]) <= rel_tol
        ok_g &= rel(G[opt]["delta"]["pathwise"]["value"], bs[sd]["delta"]) <= rel_tol
        ok_g &= rel(G[opt]["delta"]["lr"]["value"], bs[sd]["delta"]) <= rel_tol
    ok_g &= rel(ec["gamma"]["lr"]["value"], gam) <= lr_gamma_tol
    big = conv[conv["N"] == max(int(n) for n in conv_sizes)]
    flags = {"mc_prices_match_bs": bool(ok_price), "greeks_within_tolerance": bool(ok_g),
             "put_call_parity_delta": bool(abs(pcp["delta_diff"] - eq) <= parity_tols["delta"]),
             "put_call_parity_gamma": bool(abs(pcp["gamma_diff"]) <= parity_tols["gamma"]),
             "put_call_parity_rho": bool(abs(pcp["rho_diff"] - K * T * er) <= parity_tols["rho"]),
             "pathwise_variance_lower": bool(s["variance_ratio"]["pw_variance"] < s["variance_ratio"]["lr_variance"]),
             "delta_surface_monotone": bool(all(np.all(np.diff(g.sort_values("S0")["delta_call_eu"]) > 0)
                                                for _, g in dsurf.groupby("sigma"))),
             "vega_positive": bool((vsurf["vega_call_eu"] > 0).all() and (vsurf["vega_put_eu"] > 0).all()),
             "convergence_ok": bool((np.abs(big["value"] - bs["call"]["delta"]) <= conv_tol).all())}
    flags["all_checks_passed"] = bool(all(flags.values()))
    if png_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Sv, Vv = np.array(sorted(set(dsurf["S0"]))), np.array(sorted(set(dsurf["sigma"])))
        Zs = dsurf.pivot(index="sigma", columns="S0", values="delta_call_eu").loc[Vv, Sv].to_numpy()
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        XX, YY = np.meshgrid(Sv, Vv)
        ax.plot_surface(XX, YY, Zs, cmap="viridis")
        ax.set_xlabel("S0")
        ax.set_ylabel("sigma")
        ax.set_zlabel("call delta")
        fig.savefig(png_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
    return {"prices": P, "greeks": G, "consistency": cons, "delta_surface": dsurf, "vega_surface": vsurf,
            "convergence": conv, "summary": flags, "bs": bs}


# ============================================================================ 14. credit spread decomposition
def credit_spread_report(series, start, end, window, lags, r2_threshold, ffill=("TEDRATE",), names=None) -> dict:
    """Monthly BAA-Treasury spread decomposition by rolling OLS with Newey-West t-stats.

    series: {name: Series indexed by date} (daily or monthly; FRED '.' already NaN) with names BAA, AAA,
      DGS10, TEDRATE, T10YIE (override with names={'baa':..., 'aaa':..., 'tsy':..., 'liq':..., 'infl':...}).
    Alignment: every series -> last valid value per calendar month, labelled the FIRST day of the month
      (monthly_last); LEFT-join on the BAA months; ffill the discontinued columns AFTER the join; drop
      months missing BAA/AAA/Treasury/inflation; keep start <= month <= end.
    Y = BAA - Treasury; X = [BAA - AAA, liquidity, inflation]; for each window of `window` months ending at t:
      OLS with intercept, R2, Bartlett HAC (lags, no small-sample factor) t-stats of the three slopes,
      component_k = b_k X_k[t], residual = Y[t] - a - sum components (intercept goes to the residual),
      pct_k = Cov(b_k X_k, Y)/Var(Y) 100, pct_residual = (1 - R2) 100.
    Returns decomposition (date 'YYYY-MM-DD', baa_treasury_spread, default/liquidity/inflation_component,
      residual, r_squared, hac_tstat_*), variance (date, pct_*), regime_breaks (date, r_squared < threshold),
      aligned (the monthly frame)."""
    from .helpers75_3 import monthly_last, rolling_ols_hac
    nm = {"baa": "BAA", "aaa": "AAA", "tsy": "DGS10", "liq": "TEDRATE", "infl": "T10YIE"}
    nm.update(names or {})
    m = {}
    for k, s in series.items():
        x = pd.to_numeric(pd.Series(s), errors="coerce")
        from .helpers75_3 import dated_index
        x.index = dated_index(x, k)
        m[k] = monthly_last(x.dropna(), label="start")
    df = m[nm["baa"]].to_frame(nm["baa"])
    for k in m:
        if k != nm["baa"]:
            df = df.join(m[k].rename(k), how="left")
    for c in ffill:
        if c in df.columns:
            df[c] = df[c].ffill()
    df = df.dropna(subset=[nm["baa"], nm["aaa"], nm["tsy"], nm["infl"]])
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    y = (df[nm["baa"]] - df[nm["tsy"]]).to_numpy()
    X = np.column_stack([(df[nm["baa"]] - df[nm["aaa"]]).to_numpy(), df[nm["liq"]].to_numpy(), df[nm["infl"]].to_numpy()])
    lab = df.index.strftime("%Y-%m-%d").to_numpy()
    ro = rolling_ols_hac(y, X, int(window), lags=int(lags), add_const=True, index=lab,
                         names=["default", "liquidity", "inflation"])
    dec = pd.DataFrame({"date": ro["label"], "baa_treasury_spread": ro["y"], "default_component": ro["comp_default"],
                        "liquidity_component": ro["comp_liquidity"], "inflation_component": ro["comp_inflation"],
                        "residual": ro["residual"], "r_squared": ro["r_squared"],
                        "hac_tstat_default": ro["t_hac_default"], "hac_tstat_liquidity": ro["t_hac_liquidity"],
                        "hac_tstat_inflation": ro["t_hac_inflation"]})
    var = pd.DataFrame({"date": ro["label"], "pct_default": ro["pct_default"], "pct_liquidity": ro["pct_liquidity"],
                        "pct_inflation": ro["pct_inflation"], "pct_residual": ro["pct_residual"]})
    dec = dec.reset_index(drop=True)
    var = var.reset_index(drop=True)
    rb = dec.loc[dec["r_squared"] < r2_threshold, ["date", "r_squared"]].reset_index(drop=True)
    return {"decomposition": dec, "variance": var, "regime_breaks": rb, "aligned": df}
