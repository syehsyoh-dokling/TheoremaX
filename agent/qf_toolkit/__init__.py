"""qf_toolkit - tested building blocks the generated solution imports instead of rewriting.

Why it exists (crash census over 5,607 graded attempts, 16 Sep): the dominant failures were
programs that never ran - guessed column/key names (KeyError in 54 units), truncated long
programs (SyntaxError), missing imports, pandas-3 idioms, JSON that cannot hold numpy bools -
and re-derived finance with the wrong sign or unit (negative VaR, half-life in the wrong unit).
Every function here is covered by track1_agent/tests/test_qf_toolkit.py.

    from qf_toolkit import read_table, col, write_csv, bs_price, var_es_historical
"""
from .io import (col, find_input, key, output_path, read_csv, read_json, read_table,  # noqa: F401
                 to_native,
                 write_csv, write_json, write_parquet)
from .options import (baw_american, black76_price, bs_greeks, bs_price, crr_price,  # noqa: F401
                      garman_kohlhagen_price, implied_vol)
from .rates import (bond_price, bond_ytm, bootstrap_par, discount_factor,  # noqa: F401
                    duration_convexity, forward_rate, par_curve, zero_rate,
                    yield_curve_from_dfs)
from .risk import (cornish_fisher_var, kupiec_pof, var_es_historical, var_es_normal,  # noqa: F401
                   var_es_student_t, fit_garch11, dcc_correlation, simulate_jump_diffusion)
from .stats import (annualized_mean, annualized_vol, ar1_fit, cagr, drawdown_series,  # noqa: F401
                    ewma_variance, max_drawdown, newey_west_tstat, ols, ou_from_ar1, sharpe,
                    spearman_ic)
from .timeutil import (MONTH_END, QUARTER_END, YEAR_END, between_inclusive, month_end,  # noqa: F401
                       sorted_unique_dates, to_datetime, year_fraction)

# 26 Sep: textbook formula families the model got wrong when re-deriving them (barrier direction,
# probabilities > 1, forward points not in pips, P&L sign, negative VaR). Tests in the dev scratchpad
# (test_tk_exotics.py, test_tk_fx.py, test_tk_riskx.py, test_tk_credit_rates.py).
from . import exotics, fx, riskx, credit_rates  # noqa: F401,E402
from . import copulas, aggregate_loss, backtest, attribution  # noqa: F401,E402
from . import compound, event_study  # noqa: F401,E402
from . import recipes  # noqa: F401,E402
from .recipes import ewma_covariance, trade_ledger, cumulative_adjustment_factor  # noqa: F401,E402
from .recipes import calibrate_gbm, double_sort_portfolios, fit_ou_jump, simulate_ou_jump  # noqa: F401,E402

# know75: per-unit gap helpers (helpers75_<n>.py), imported when present
_H75 = []
for _hn in ("helpers75_1", "helpers75_2", "helpers75_3", "helpers75_4",
            "helpers76_1", "helpers76_2", "helpers76_3", "helpers76_4",
            "helpers77_1", "helpers77_2", "helpers77_3", "helpers77_4",
            "helpers78_1", "helpers78_2", "helpers78_3", "helpers78_4",
            "helpers79_1", "helpers79_2", "helpers79_3", "helpers79_4",
            "helpers80_1", "helpers80_2", "helpers80_3", "helpers80_4",
            "helpers81_1", "helpers81_2", "helpers81_3", "helpers81_4",
            "helpers86_1", "helpers86_2", "helpers87_1", "helpers87_A", "helpers87_B", "helpers87_C",
            "conventions_ref"):          # know81: the academic conventions reference table + its relation
    try:
        _H75.append(__import__(f"{__name__}.{_hn}", fromlist=["_"]))
    except Exception:  # noqa: BLE001 - a broken helper module must never break the toolkit
        pass

# export only the functions each module DEFINES (a star-import would also leak np/math/itertools)
for _mod in (exotics, fx, riskx, credit_rates, copulas, aggregate_loss, backtest, attribution,
             compound, event_study, *_H75):
    for _name in dir(_mod):
        _obj = getattr(_mod, _name)
        if (not _name.startswith("_") and callable(_obj)
                and getattr(_obj, "__module__", "") == _mod.__name__ and _name not in globals()
                and _name not in getattr(_mod, "_ENGINE_ONLY", ())):   # know80: engine aids stay out of the API
            globals()[_name] = _obj
del _mod, _name, _obj


# 27 Sep full run: the model imported `from qf_toolkit.options import asian_geometric_price` (it lives
# in exotics) and `qf_toolkit.reading` (does not exist) - 9 identical ImportErrors burned a unit's
# whole request budget. Any toolkit name is importable from ANY submodule, and an invented submodule
# name resolves to the package; a truly unknown name raises with the closest real names.
def _runtime_emit(name, obj):
    """know87 (round 2): `from qf_toolkit import emit` - emit is the RUNTIME's deliverable writer (predefined in
    every program and cell), not a toolkit function; the import used to resolve to an empty alias MODULE
    ('module' object is not callable, a whole monolith lost). This forwards to the runtime emit of the caller."""
    import sys
    g = sys._getframe(1).f_globals
    fn = g.get("emit")
    if callable(fn) and fn is not _runtime_emit and not isinstance(fn, type(sys)):
        return fn(name, obj)
    for mod in ("soft_run", "__main__"):
        fn = getattr(sys.modules.get(mod), "_emit", None)
        if callable(fn):
            return fn(name, obj)
    import os
    out = os.environ.get("T1_OUTPUT_DIR") or "/app/output"
    path = name if os.path.isabs(str(name)) else os.path.join(out, str(name))
    if hasattr(obj, "to_csv") and str(name).lower().endswith((".csv", ".tsv", ".txt")):
        return str(write_csv(obj, path, sep="\t" if str(name).lower().endswith(".tsv") else ","))
    if isinstance(obj, (dict, list)) or str(name).lower().endswith(".json"):
        return str(write_json(obj, path))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(obj if isinstance(obj, str) else str(obj))
    return path


_RUNTIME_NAMES = {"emit": _runtime_emit}


def __getattr__(name):
    """PEP 562: only the runtime names (emit) resolve here; anything else keeps the normal import path."""
    if name in _RUNTIME_NAMES:
        return _RUNTIME_NAMES[name]
    raise AttributeError(name)


def _delegate(name):
    if name.startswith("__"):
        raise AttributeError(name)
    if name in _RUNTIME_NAMES:
        return _RUNTIME_NAMES[name]
    if name in globals():
        return globals()[name]
    import difflib
    close = difflib.get_close_matches(name, [k for k in globals() if not k.startswith("_")], n=5)
    raise AttributeError(f"qf_toolkit has no {name!r}; closest: {close}; import from qf_toolkit directly")


from . import io as _io_m, options as _opt_m, rates as _rates_m, risk as _risk_m  # noqa: E402
from . import stats as _stats_m, timeutil as _time_m  # noqa: E402

for _m in (_io_m, _opt_m, _rates_m, _risk_m, _stats_m, _time_m, exotics, fx, riskx, credit_rates,
           copulas, aggregate_loss, backtest, attribution, compound, event_study):
    _m.__getattr__ = _delegate
del _m


def _install_alias_finder():
    import importlib.abc
    import importlib.util
    import sys
    import types

    class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        """Consulted only after the normal finders failed: qf_toolkit.<invented> -> the package."""
        def find_spec(self, fullname, path, target=None):
            if fullname.startswith(__name__ + ".") and fullname.count(".") == 1:
                return importlib.util.spec_from_loader(fullname, self)
            return None

        def create_module(self, spec):
            module = _AliasModule(spec.name)
            module.__getattr__ = _delegate
            return module

        def exec_module(self, module):
            pass

    class _AliasModule(types.ModuleType):
        """know87 (round 2): `from qf_toolkit import <invented name>` also lands here (the import system tries a
        submodule when the attribute is missing) and then gets CALLED; say what is wrong instead of
        TypeError: 'module' object is not callable."""
        def __call__(self, *args, **kwargs):
            import difflib
            short = self.__name__.rsplit(".", 1)[-1]
            pkg = sys.modules.get(__name__)
            real = [k for k, v in vars(pkg).items() if not k.startswith("_") and callable(v)
                    and not isinstance(v, types.ModuleType)] if pkg is not None else []
            close = difflib.get_close_matches(short, real, n=5)
            raise AttributeError(f"qf_toolkit has no {short!r} (an invented name imports as an empty module); "
                                 f"closest: {close}; import a listed toolkit name")

    if not any(type(f).__name__ == "_AliasFinder" for f in sys.meta_path):
        sys.meta_path.append(_AliasFinder())


_install_alias_finder()


# 27 Sep full run: TypeErrors on keyword names the model reaches for by habit - barrier_price(S0=...),
# annualized_vol(ppy=...) (the old API line itself said ppy). A keyword the function does not have is
# renamed to the ONE parameter of the same group it does have; anything else is left to raise.
_KW_GROUPS = (
    {"S", "S0", "spot", "spot_price", "s0"},
    {"K", "strike", "strike_price", "k"},
    {"sigma", "vol", "volatility", "sig"},
    {"T", "maturity", "tau", "time_to_maturity", "expiry"},
    {"r", "rate", "rf", "risk_free", "risk_free_rate", "r_f"},
    {"q", "div", "dividend", "dividend_yield", "div_yield"},
    {"kind", "option_type", "opt_type", "cp", "call_put", "type"},
    {"periods_per_year", "ppy", "periods", "annualization", "annualisation"},
    {"level", "confidence", "conf", "confidence_level"},
    {"n_paths", "paths", "num_paths", "n_sims", "n_simulations"},
    {"n_steps", "steps", "num_steps"},
)


def _with_kw_aliases(func):
    import functools
    import inspect
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return func
    params = set(sig.parameters)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return func
    mapping = {}
    for group in _KW_GROUPS:
        present = group & params
        if len(present) == 1:
            target = next(iter(present))
            mapping.update({alias: target for alias in group - params})
    if not mapping:
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for key in [k for k in kwargs if k in mapping]:
            if mapping[key] not in kwargs:
                kwargs[mapping[key]] = kwargs.pop(key)
        return func(*args, **kwargs)

    wrapper.__signature__ = sig
    return wrapper


for _name in [k for k in list(globals()) if not k.startswith("_")]:
    _obj = globals()[_name]
    if callable(_obj) and getattr(_obj, "__module__", "").startswith(__name__ + "."):
        _wrapped = _with_kw_aliases(_obj)
        globals()[_name] = _wrapped
        _home = __import__("sys").modules.get(_obj.__module__)
        if _home is not None and getattr(_home, _name, None) is _obj:
            setattr(_home, _name, _wrapped)   # `from qf_toolkit.exotics import x` gets it too
del _name, _obj, _wrapped, _home

# Alias for model hallucinating module names
from . import rates as rates_and_bonds  # noqa: F401
from . import stats as statistics  # noqa: F401

#: The prompt block: one line per function, conventions and RETURN TYPES included. v6.8
#: measured three misreadings of a terser block: `ols` unpacked as a 5-tuple, an imported
#: `forward_rate` that did not exist, and a case-sensitive day-count basis.
API = """\
from qf_toolkit import ...   ALWAYS import from the top level (from qf_toolkit import barrier_price), never from a submodule.
ONLY the function/constant names below exist; the [bracketed] lines are section titles, not names.
[reading inputs / writing outputs]
     (INPUTS only - never read a deliverable you are supposed to write)
     read_table(name, parse_dates=None) -> DataFrame   csv/parquet/pqt/json, bare filename ok
     read_json(name) -> dict/list | find_input(name) -> Path (error lists existing files)
     col(df, "Close") -> str real column name (case/space/_ tolerant; error lists columns)
     key(d, "n_obs") -> value from dict d, same tolerant lookup
     output_path(name, out_dir=None) -> Path | write_csv(df, path) full precision, index=False
     write_json(obj, path) numpy/bool/NaN-safe (use instead of json.dump) | write_parquet(df, path)
[dates]
     pandas 3 aliases: "s" "min" "h" "D" "ME" "QE" "YE" ("S","T","H","M" are invalid)
     MONTH_END="ME" | month_end(s) -> Series | to_datetime(x) raises on bad values
     between_inclusive(s, "2022-01-01", "2023-12-31") -> bool mask keeping the WHOLE end day
     sorted_unique_dates(x) -> DatetimeIndex
     year_fraction(d0, d1, basis) -> float; basis any case: ACT/365, ACT/360, ACT/ACT, 30/360, 30E/360
[options]
     bs_price(S,K,T,r,sigma,kind="call",q=0.0) -> float|array
     bs_greeks(S,K,T,r,sigma,kind,q) -> dict{delta,gamma,vega per 1.00 vol,theta per YEAR,rho per 1.00}
     black76_price(F,K,T,r,sigma,kind) | garman_kohlhagen_price(S,K,T,r_dom,r_for,sigma,kind)
     implied_vol(price,S,K,T,r,kind,q) -> float (nan if outside no-arbitrage bounds)
     crr_price(S,K,T,r,sigma,steps=500,kind,american=True,q) -> float
     baw_american(S,K,T,r,sigma,kind,q) -> tuple (price, critical_price)
[tail risk]
     var_es_historical(returns, level, method="linear") -> tuple (VaR, ES), POSITIVE losses
     var_es_normal(mu,sigma,level) -> (VaR, ES) | var_es_student_t(mu,scale,nu,level) -> (VaR, ES)
     cornish_fisher_var(mu,sigma,skew,excess_kurt,level) -> float | kupiec_pof(x,n,level) -> (LR, p)
[statistics]
     ols(y, X, add_const=True, hac_lags=None) -> dict{beta,se,t,resid,r2,nobs}  (NOT a tuple)
     newey_west_tstat(x, lags) -> tuple (mean, se, t)
     sharpe(r, periods_per_year, rf_per_period=0, ddof=1) -> float
     annualized_vol(r, periods_per_year, ddof=1) | annualized_mean(r, periods_per_year) | cagr(r, periods_per_year) -> float
     max_drawdown(r) -> NEGATIVE float | drawdown_series(r) -> Series
     ar1_fit(x, add_const=True) -> dict{a,phi,sigma_e,nobs,add_const}
     ou_from_ar1(a,phi,sigma_e,dt=1.0) -> dict{kappa,theta,sigma,half_life,mean_reverting}
       kappa per unit of dt; half_life in units of dt (dt=1 -> periods)
     ewma_variance(r, lam=0.94) -> array | spearman_ic(signal, fwd_return) -> float
[rates and bonds]
     discount_factor(rate,t,comp) / zero_rate(df,t,comp) -> float; comp "continuous"|"simple"|m
     forward_rate(df1,t1,df2,t2,comp) -> float
     par_curve(maturities, par_rates, freq) -> DataFrame[t,par_rate,df,zero_continuous,zero_compounded]
       (any maturities; interpolates) | bootstrap_par(grid, par_rates, freq) -> array (grid k/freq only)
     bond_price(ytm,coupon_rate,T,freq=2,face=100) | bond_ytm(price,coupon_rate,T,freq,face) -> float
     duration_convexity(ytm,coupon_rate,T,freq,face) -> dict{price,macaulay,modified,convexity}
[exotics]  continuous monitoring; H>S up, H<S down (automatic); scalar in -> float, array in -> ndarray
     barrier_hit_prob(S0,H,T,mu,sigma,drift_is_log=True) -> P(touch H by T) in [0,1]; mu=log-drift nu (drift_is_log=False: mu is the S-drift e.g. r-q)
     first_passage_cdf(S0,H,t,mu,sigma,drift_is_log=True) -> P(tau<=t) | first_passage_pdf(same) -> density per year
     expected_first_passage_time(S0,H,mu,sigma,drift_is_log=True) -> years (inf if drift points away)
     running_max_cdf(m,T,nu,sigma) = P(M_T<=m) = P(NO up-hit) | running_min_cdf(m,T,nu,sigma) = P(m_T>=m) = P(NO down-hit) (a SURVIVAL, not P(m_T<=m)) | joint_end_max_cdf(x,b,T,nu,sigma): LOG space m=ln(level/S0)
     for a HIT probability use barrier_hit_prob (either direction) = 1 - running_max_cdf (up) = 1 - running_min_cdf (down)
     bgk_shift(H,S0,sigma,dt) -> barrier moved away from S0 by exp(0.5826 sigma sqrt(dt)) (discrete monitoring)
     barrier_price(S,K,H,T,r,sigma,kind="call",barrier="down-and-out"|"down-and-in"|"up-and-out"|"up-and-in",q=0,rebate=0) -> float; in+out=vanilla
     digital_price(S,K,T,r,sigma,kind="call",payoff="cash"|"asset",q=0,cash=1.0) | one_touch_price(S,H,T,r,sigma,q=0,cash=1,pay_at="expiry"|"hit") | no_touch_price(S,H,T,r,sigma,q=0,cash=1)
     lookback_floating_price(S,T,r,sigma,kind="call",q=0,s_min=None,s_max=None) call S_T-min, put max-S_T
     lookback_fixed_price(S,K,T,r,sigma,kind="call",q=0,s_max=None,s_min=None) call (max-K)+, put (K-min)+
     asian_geometric_price(S,K,T,r,sigma,kind,q=0,n=None,include_t0=False) exact | asian_arithmetic_levy(same) approx | asian_curran(S,K,T,r,sigma,kind,q=0,n=12,include_t0=False)
     margrabe_price(S1,S2,T,sigma1,sigma2,rho,q1=0,q2=0) (S1-S2)+ | kirk_spread_price(F1,F2,K,T,r,sigma1,sigma2,rho,kind) on forwards | kirk_spread_price_spot(S1,S2,K,T,r,sigma1,sigma2,rho,q1=0,q2=0,kind)
     gbm_paths(S0,mu,sigma,T,n_steps,n_paths,seed=None,antithetic=False,rng=None,include_t0=True,drift_is_log=REQUIRED) -> (n_paths,n_steps+1); drift_is_log=True: mu = r-q-sigma^2/2 (as barrier_hit_prob), False: mu = r-q | running_extremes(paths) -> per-path (max,min) over the WHOLE path, 1-D
     running_max_path(paths) / running_min_path(paths) -> cumulative (n_paths,n_times) | first_hit_times(paths,H,dt,S0=None) -> years, inf if never
     mc_hit_probability(paths,H,dt,horizons,S0=None) -> (probs, se) P(touch H by t); discrete grid - use fine dt or bgk_shift to match barrier_hit_prob
[fx]  pair "BASE/QUOTE" = price of 1 BASE in QUOTE; rates DECIMAL (percent/100)
     pip_multiplier(pair,overrides=None) -> 1e4 (1e2 JPY quote) | forward_points(F,S,pair=None,multiplier=None) -> PIPS (F-S)*mult | outright_from_points(S,pips,pair)
     money_market_year_fraction(days,currency=None,basis=None) ACT/360, ACT/365 for GBP AUD NZD CAD HKD SGD ZAR...
     cip_forward(S,r_base,r_quote,t_base,t_quote=None,compounding="simple"|"continuous") -> F = S(1+r_q t_q)/(1+r_b t_b)
     cip_forward_bid_ask(S_bid,S_ask,rb_bid,rb_ask,rq_bid,rq_ask,t_base,t_quote=None) -> (F_bid,F_ask)
     cross_rate(pair1,bid1,ask1,pair2,bid2,ask2,target) -> (bid,ask,mid) any orientation | invert_quote(bid,ask) -> (1/ask,1/bid)
     cip_deviation_bps(F_mkt,F_cip) -> (F_mkt-F_cip)/F_cip*1e4 | implied_basis(F_mkt,S,r_base,r_quote,t_base,t_quote) | implied_quote_rate / implied_base_rate
     direction_sign(d) +1 buy/long, -1 sell/short | forward_mtm(N_base,K,F_now,direction="buy",discount_factor_quote=1) -> P&L in QUOTE ccy
     to_usd(amount,ccy,{pair:mid}) | forward_pnl_attribution(N,direction,S_then,S_now,F_then,F_now,total=None) -> {spot_delta,carry,total,residual}
     net_currency_exposure(trades) -> {ccy: signed amount} (buy BASE/QUOTE N at K: +N BASE, -N*K QUOTE) | triangular_arbitrage({pair:(bid,ask)})
[VaR / ES, backtests, sizing]  all VaR/ES POSITIVE loss fractions (value= scales to money); level 0.99 or 99; returns oldest->newest
     parametric_var_es(returns|mu=,sigma=, level, dist="normal"|"t", nu=None, include_mean=True, horizon=1, scaling="sqrt", ddof=1, t_scale="std"|"scale", value=1) -> (VaR,ES)
     ewma_var_es(returns,lam=0.94,level,include_mean=False,init="first"|"sample"|float,horizon=1,value=1,return_series=False) | ewma_sigma_series(returns,lam,init) -> (sigma2 VARIANCE array, sigma2_next) TUPLE | ewma_vol(returns,lam,init,as_of="last",periods_per_year=None) -> float current EWMA VOL (annualized when periods_per_year=252)
     historical_var_es(returns,level,method="linear"|"empirical"|"lower"|"higher"|"midpoint",es_tail="geq"|"gt"|"k_worst",value=1) -> (VaR,ES)
     age_weighted_var_es(returns,lam=0.98,level,interpolate=False,value=1) (BRW) | filtered_hs_var_es(returns,lam=0.94,level) | kernel_var_es(returns,level,bandwidth="silverman")
     scale_horizon(var_1d,horizon,method="sqrt") | portfolio_returns(prices_df,weights,kind="simple"|"log_avg"(avg of asset log-returns)|"log_portfolio"(ln of portfolio gross)) | var_from_portfolio(prices_df,weights,level,method=...)
     exceedances(returns,var_series,strict=True) | christoffersen_independence(hits) -> (LR,p) | christoffersen_cc(hits,level) | basel_traffic_light(exceptions,n=250,level=0.99) -> (zone,plus,multiplier)
     component_var(weights,cov,level,value=1) -> dict{total,marginal,component,pct} | kelly_fraction(mu,sigma,rf=0 | p=,b=) | fractional_kelly(f,fraction) | var_constrained_fraction(f_kelly,returns_or_sigma,var_limit,level)
[credit]  spreads DECIMAL (bp/1e4)
     merton_model(V,sigma_V,D,T,r) -> dict{equity,debt,pd_rn,dd,spread,...} | merton_calibrate(E,sigma_E,D,T,r) -> + {V,sigma_V} | black_cox_pd(V,barrier,sigma_V,T,mu)
     hazard_from_spread(spread,recovery=0.4) | survival_prob(t,hazard_tenors,hazards) | cds_bootstrap(tenors,spreads,recovery,discount_curve_fn,freq=4,accrual=True) -> DataFrame | cds_par_spread(...)
     vasicek_loss_quantile(pd,rho,level=0.999) | vasicek_loss_distribution(x,pd,rho,lgd=1,kind="cdf"|"pdf"|"quantile") | one_factor_mc_losses(pds,eads,lgds,rho,n_sims,seed)
     basel_irb_capital(pd,lgd=0.45,rho=None,maturity=2.5,asset_class="corporate",ead=1,pd_floor=0.0003) -> dict{K,risk_weight,rwa,capital,R,...}
     matrix_power_migration(P,years) | generator_from_matrix(P,fix="DA") | cohort_matrix(transitions,...) | cumulative_default_probs(P,horizons)
[short rate / rate options]
     vasicek_zcb(r0,kappa,theta,sigma,T) | cir_zcb(...) | hull_white_zcb(a,sigma,T,curve_fn) | vasicek_zcb_option(r0,kappa,theta,sigma,T,S,K,kind) | vasicek_swaption(...)
     black_caplet(F,K,T_fix,sigma,df_pay,accrual,notional=1,kind="caplet"|"floorlet") | black_swaption(F_swap,K,T,sigma,annuity,kind="payer") | bachelier_price/_caplet/_swaption (normal vol)
     swap_rate_and_annuity(dfs,times,freq=None) -> dict{swap_rate,annuity,accruals}
[copulas]  family gaussian|t|clayton|gumbel|frank; u = pseudo-observations in (0,1)
     copula_param_from_tau(family,tau) | copula_tau(family,theta) | spearman_rho(family,theta,nu=None) | tail_dependence(family,theta,nu=None) -> (lower,upper)
     copula_sample(family,theta,n,d=2,nu=None,seed=None) -> (n,d) uniforms (Marshall-Olkin for Archimedean: gumbel via positive stable)
     pseudo_observations(x) ranks/(n+1) | copula_logpdf(family,u,theta,nu=None) | fit_copula(family,u,method="mle"|"itau",nu_bounds=(2.01,100)) -> dict{theta,nu,loglik,aic,bic,...}
     select_copula(u,families,criterion="aic") -> ranked list | empirical_kendall_tau(u) | empirical_tail_dependence(u,q=0.95,tail="upper",denominator="first"|"nominal")
[aggregate loss]  lattice k*h
     severity_dist("gamma"|"lognormal"|"pareto"|"lomax"|"exponential"|"weibull", **params) -> scipy dist (gamma(shape,scale), lognormal(mu,sigma), pareto(alpha,xm)); every agg-loss fn also takes {"dist": name, **params}
     suggest_lattice(lam,severity_dist,level) -> (h,n) sizing the lattice for the tail (call it FIRST) | discretize_severity(dist,h,n,method="rounding",tail="drop"|"last") | compound_poisson_fft(lam,severity_pmf,n=None,pad=True) | panjer_poisson(lam,severity_pmf,n) | panjer_negbin | panjer_binomial
     aggregate_var_es(pmf,h,level,es_mode="geq"|"gt"|"tail_expectation") -> (VaR,ES) | aggregate_moments(pmf,h) | compound_poisson_moments(lam,m1,m2) | compound_poisson_mc(lam,sampler,n_sims,seed)
[recipes]  whole computations in one call (use these before hand-rolling)
     trade_ledger(prices,signal,initial_capital=100000.0,exec_lag=1,exec_price="open"|"close",whole_shares=True,cost_bps=0.0,open_prices=None) -> dict{trades DataFrame[type,signal_date,exec_date,price,shares,value,cost,pnl], daily_portfolio DataFrame[date,close,position_shares,cash,holdings_value,portfolio_value,daily_return,cumulative_return,drawdown], metrics{total_return,final_capital,annualized_return,annualized_volatility,sharpe_ratio,max_drawdown,calmar_ratio,num_trades,win_rate,total_pnl,buy_count,sell_count}}; signal = desired STATE after close t (+1 long, 0 flat)
     ewma_covariance(returns_matrix,lam=0.94,init="sample") -> Sigma [n,n] current EWMA covariance (per-asset recursion); portfolio variance = w @ Sigma @ w, annualise outside
     cumulative_adjustment_factor(factors) -> product of per-event price factors (<1 for forward splits and dividends)
     calibrate_gbm(prices,trading_days=252,log_returns=True,ddof=1) -> dict{S0 last price,n_prices,n_returns,return_mean,return_std,sigma=std*sqrt(td),log_mu=mean*td,mu=log_mu+0.5*sigma^2 (GBM drift) when log_returns}
     double_sort_portfolios(df,date_col,id_col,sort1,sort2,ret_col,n1=5,n2=5,dependent=True,pick=(b1,b2),weight_col=None,cost_bps=0.0,min_obs=n1*n2) -> dict{cells DataFrame[date,b1,b2,ret,n], series DataFrame[date,gross,turnover,cost,net] for pick}; rank buckets floor((rank-1)*n/count)+1 (ties by id), dependent = b2 within b1; turnover 0.5*sum|dw|, first period 1.0
     fit_ou_jump(x,dt,jump_threshold=3.0,max_iter=5) -> dict{kappa=-ln(phi)/dt,theta=a/(1-phi),sigma (exact discretisation),half_life,phi,a,jump_intensity per unit time,jump_mean,jump_std,n_jumps,n_obs}; AR(1) OLS with iterative |resid|>k*std jump filter; dt in years
     simulate_ou_jump(x0,kappa,theta,sigma,jump_intensity,jump_mean,jump_std,T,n_steps,n_paths,seed=None) -> ndarray [n_paths,n_steps+1]; exact OU step + Poisson(lam*dt) normal jumps
[backtest]  position[t] = held during bar t; a signal at t trades from t+1 (no look-ahead)
     signals_to_positions(signal,lag=1,long_only=False,hold="until_exit"|"signal") | bollinger_bands(price,window=20,k=2.0,ddof=0) | bollinger_signals(price,window,k,mode="mean_reversion"|"breakout",exit="mid"|"opposite")
     sma_crossover_signals(price,fast,slow) | momentum_signal(returns,lookback,skip=0)
     backtest_positions(price_or_returns,positions,initial_capital=100000.0,cost_bps=0.0,input="price"|"returns",position_is="weight"|"shares") -> DataFrame[position,strategy_return,cost,equity,turnover]
     trade_list(positions,price) -> round trips | performance_summary(strategy_returns,equity=None,periods_per_year=252,rf=0.0,trades=None,initial_capital=None) -> dict{total_return,cagr,annualized_volatility,sharpe,sortino,max_drawdown POSITIVE,calmar,win_rate,n_trades,final_capital}
[attribution]
     brinson_attribution(wp,wb,rp,rb,model="BHB"|"BF",interaction="separate"|"in_selection") -> DataFrame[allocation,selection,interaction,total]; sums to Rp-Rb
     link_attribution(period_frames,rp_series,rb_series,method="carino"|"menchero"|"geometric"|"arithmetic") | returns_based_attribution(port_returns,factor_returns)
     delta_hedge_pnl(S_path,option_values,deltas,r=0.0,dt=1/252,hedge_freq=1,position="short_option"|"long_option") -> dict{total_pnl,pnl_series,...}; short option + long delta
[compound / chooser]  Geske (1979), rho=sqrt(T1/T2); exact bivariate normal
     compound_option_price(S,K1,K2,T1,T2,r,sigma,q=0,kind="call_on_call"|"put_on_call"|"call_on_put"|"put_on_put") | geske_critical_price(K1,K2,T1,T2,r,sigma,q=0,inner="call")
     compound_parity_rhs(S,K1,K2,T1,T2,r,sigma,q=0,inner="call") -> BS(S,K2,T2) - K1 e^{-rT1} = CoX - PoX on the SAME inner X (call parity needs put_on_call, NOT put_on_put) | compound_option_mc(S,K1,K2,T1,T2,r,sigma,q=0,kind,n_paths=200000,seed=None) -> (price,se)
     chooser_simple_price(S,K,t_choose,T,r,sigma,q=0) | chooser_complex_price(S,Kc,Kp,t_choose,Tc,Tp,r,sigma,q=0) | bivariate_normal_cdf(a,b,rho)
[event study]  window=(a,b) inclusive
     market_model(stock_ret_est,market_ret_est) -> dict{alpha,beta,resid,sigma(ddof=2),L,mkt_mean,mkt_ss} | abnormal_returns(stock_ret,market_ret,alpha,beta) -> array | car(ar,window=None) -> FLOAT
     patell_standardized(ar_event,market_ret_event,mm,window=None,scar_var="sum_c") -> dict{sar,C,car,scar} (C = 1 + 1/L + (Rm-Rm_bar)^2/V_m) | patell_test(sars,L_list) -> float | bmp_test(scars) -> float
     kolari_pynnonen_adjust(t_bmp,rho_bar,n,form="kp2010"|"simple") kp2010: t*sqrt((1-rho)/(1+(n-1)rho)); simple: t/sqrt(1+(n-1)rho) - use the form the task writes | average_residual_correlation(resid_series_list,min_overlap=10) -> float; each item a Series indexed by CALENDAR DATE
     corrado_rank_test(est_resids_list,event_ars_list,event_day_index=None,window=None) -> dict{z_day0,z_window,s_K,...} | corrado_zivney_sign_test(...) | sign_test(cars,method="binomial"|"generalized")
     event_study_panel(events,returns,market,est_window=(-250,-11),event_window=(-5,5),...) -> dict{params,ar,aar,caar,stats}
Conventions not fixed by the task still need a decision: pass them explicitly."""


def _h75_api_lines() -> str:
    """know75: one API line per helper in helpers75_* (signature + first docstring line)."""
    import inspect
    lines = []
    for mod in _H75:
        for name in sorted(dir(mod)):
            obj = getattr(mod, name)
            if (name.startswith("_") or not callable(obj) or getattr(obj, "__module__", "") != mod.__name__
                    or name in getattr(mod, "_ENGINE_ONLY", ())):
                continue
            try:
                sig = str(inspect.signature(obj))
            except (TypeError, ValueError):
                sig = "(...)"
            doc = (inspect.getdoc(obj) or "").strip().splitlines()
            lines.append(f"     {name}{sig}" + (f" - {doc[0][:160]}" if doc else ""))
    return ("\n[task-pattern helpers (know75)]\n" + "\n".join(lines)) if lines else ""


API = API + _h75_api_lines()


# 29 Sep: 25 API lines above name parameters by a short documented name (zero_rate(df,t,comp),
# bootstrap_par(grid,...), forward_points(F,S,...)) that the function itself calls differently
# (compounding, maturities, forward). The model follows the text and got TypeErrors (zero-coupon-
# bootstrapping crashed on exactly this). Each documented name is mapped BY POSITION to the real
# parameter, so both the documented and the real keyword work; positional calls are unchanged.
def _doc_keyword_aliases():
    import inspect
    import re as _re
    for match in _re.finditer(r"([a-z_][a-z0-9_]{2,})\(([^()]*)\)", API):
        name, doc_args = match.group(1), match.group(2)
        fn = globals().get(name)
        if fn is None or not callable(fn):
            continue
        base = getattr(fn, "__wrapped__", fn)
        try:
            real = [p for p in inspect.signature(base).parameters.values()
                    if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
        except (TypeError, ValueError):
            continue
        names = [p.name for p in real]
        extra = {}
        for i, raw in enumerate(doc_args.split(",")):
            doc = raw.strip().split("=")[0].strip()
            if (doc and _re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", doc) and doc not in names
                    and i < len(names)):
                extra[doc] = names[i]
        if extra:
            _add_aliases(name, base, extra)


def _add_aliases(name, base, extra):
    import functools
    current = globals()[name]
    inner = current

    @functools.wraps(base)
    def wrapper(*args, **kwargs):
        for key in [k for k in kwargs if k in extra]:
            if extra[key] not in kwargs:
                kwargs[extra[key]] = kwargs.pop(key)
        return inner(*args, **kwargs)

    wrapper.__signature__ = getattr(current, "__signature__", None) or __import__("inspect").signature(base)
    globals()[name] = wrapper
    home = __import__("sys").modules.get(getattr(base, "__module__", ""))
    if home is not None and getattr(home, name, None) is current:
        setattr(home, name, wrapper)


_doc_keyword_aliases()


# know89 (flip census, guard A): the model guesses keys of helper results at random ('alloc', 'ewma_cov',
# 't_stat_alpha_ols') and a PLAIN dict answers with a bare KeyError - the repair round guesses again (one unit
# lost 15 attempts). Every task-pattern helper (helpers7x/8x modules) that returns a plain dict now returns a
# ResultDict (top level; values are the helper's own objects): exact keys are untouched, a missing key lists the
# real ones. (Whole-task bundles, helpers79_1._as_bundle, also wrap their nested plain dicts.)
def _internal_caller() -> bool:
    """True when the wrapped helper is called from INSIDE the toolkit (another helper): those calls keep the exact
    know88 behaviour (plain dicts, arguments as given); only the program's own calls get the know89 guards."""
    import sys as _sys
    try:
        return str(_sys._getframe(2).f_globals.get("__name__", "")).startswith(__name__ + ".")
    except ValueError:
        return False


def _wrap_plain_results():
    import functools
    import sys as _sys

    def wrap(fn):
        if getattr(fn, "_t1_rdict", False):
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            res = fn(*args, **kwargs)
            if type(res) is not dict or _internal_caller():
                return res
            # the top level only: every value stays the very object the helper returned (a mapping helper such
            # as bundle_files hands back the program's own objects; their identity is kept)
            from ._rdict import ResultDict
            return ResultDict(res)

        wrapper._t1_rdict = True
        for attr in ("_t1_memo", "__signature__"):
            if hasattr(fn, attr):
                try:
                    setattr(wrapper, attr, getattr(fn, attr))
                except (AttributeError, TypeError):
                    pass
        return wrapper

    g = globals()
    for name in [k for k in list(g) if not k.startswith("_")]:
        obj = g[name]
        mod = getattr(obj, "__module__", "") or ""
        if not (callable(obj) and mod.startswith(__name__ + ".helpers") and not isinstance(obj, type)):
            continue
        new = wrap(obj)
        g[name] = new
        home = _sys.modules.get(mod)          # the same object everywhere (`qf_toolkit.x is helpersNN.x`); a call
        if home is not None and getattr(home, name, None) is obj:   # from inside the toolkit keeps plain dicts
            setattr(home, name, new)


try:
    _wrap_plain_results()
except Exception:  # noqa: BLE001 - an aid, never a failure cause
    pass


# know89 (flip census, guard D): a helper that COUNTS the NaNs of its raw input (its docstring reports a
# nan_count) was handed the program's NaN-filled copy of an input (np.where(np.isnan(R), 0, R), fillna(0)) and
# counted none. When an array / frame argument equals the raw numeric columns of a loaded input everywhere except
# at that input's NaN cells (filled with ONE constant), the helper gets the raw columns instead, and says so.
def _raw_block_for(arg):
    import sys as _sys
    import numpy as _np
    sr = _sys.modules.get("soft_run")
    loaded = getattr(sr, "_LOADED", None) or []
    if not loaded:
        return None
    try:
        A = _np.asarray(arg, dtype=float)
    except (TypeError, ValueError):
        return None
    if A.ndim != 2 or _np.isnan(A).any() or A.size == 0:
        return None
    for name, num in loaded:
        if len(num) != A.shape[0] or num.shape[1] < A.shape[1]:
            continue
        cols, used, fill = [], set(), None
        M = num.to_numpy(float)
        for j in range(A.shape[1]):
            hit = None
            for c in range(M.shape[1]):
                if c in used:
                    continue
                raw = M[:, c]
                ok = ~_np.isnan(raw)
                if _np.array_equal(raw[ok], A[ok, j]):
                    hit = c
                    if (~ok).any():
                        vals = _np.unique(A[~ok, j])
                        if len(vals) != 1 or (fill is not None and vals[0] != fill):
                            hit = None
                            continue
                        fill = vals[0]
                    break
            if hit is None:
                break
            used.add(hit)
            cols.append(hit)
        if len(cols) == A.shape[1] and fill is not None:
            raw = M[:, cols]
            if hasattr(arg, "columns"):
                import pandas as _pd
                raw = _pd.DataFrame(raw, index=arg.index, columns=arg.columns)
            return name, raw, fill
    return None


def _wrap_raw_input_helpers():
    import functools
    import inspect
    g = globals()
    for name in [k for k in list(g) if not k.startswith("_")]:
        obj = g[name]
        if not callable(obj) or isinstance(obj, type):
            continue
        doc = inspect.getdoc(obj) or ""
        if "nan_count" not in doc:
            continue

        def make(fn, fname):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                if _internal_caller():
                    return fn(*args, **kwargs)
                try:
                    args = list(args)
                    for i, a in enumerate(args):
                        got = _raw_block_for(a) if hasattr(a, "shape") else None
                        if got:
                            args[i] = got[1]
                            print(f"[guard89 input] {fname}: argument {i} is input {got[0]!r} with its NaNs filled "
                                  f"by {got[2]!r} - this helper counts and handles the RAW NaNs itself; passed the raw "
                                  "columns", file=__import__("sys").stderr, flush=True)
                    for k, a in list(kwargs.items()):
                        got = _raw_block_for(a) if hasattr(a, "shape") else None
                        if got:
                            kwargs[k] = got[1]
                            print(f"[guard89 input] {fname}: {k}= is input {got[0]!r} with its NaNs filled by "
                                  f"{got[2]!r} - passed the raw columns", file=__import__("sys").stderr, flush=True)
                except Exception:  # noqa: BLE001 - an aid, never a failure cause
                    pass
                return fn(*args, **kwargs)
            return wrapper

        new = make(obj, name)
        g[name] = new
        home = __import__("sys").modules.get(getattr(obj, "__module__", "") or "")
        if home is not None and getattr(home, name, None) is obj:
            setattr(home, name, new)


try:
    _wrap_raw_input_helpers()
except Exception:  # noqa: BLE001
    pass


# know92 (enum drift): a string option the program EMBELLISHED - return_kind='log_avg' where the helper tests
# `return_kind == "log"` - fell into the helper's else-branch (simple returns) without a word, and every portfolio
# statistic was off. The literals each helper compares a string parameter with are read from its own source; a
# keyword value that is none of them but contains exactly ONE of them as a whole word ('log_avg' -> 'log',
# 'simple_returns' -> 'simple') is replaced by it, and the call says so. Anything else is passed on unchanged.
def _enum_literals(fn) -> dict:
    import ast as _ast
    import inspect as _inspect
    import textwrap as _tw
    try:
        src = _tw.dedent(_inspect.getsource(fn))
        params = set(_inspect.signature(fn).parameters)
        tree = _ast.parse(src)
    except (OSError, TypeError, ValueError, SyntaxError):
        return {}
    out: dict = {}
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Compare) or not isinstance(node.left, _ast.Name) or node.left.id not in params:
            continue
        for op, comp in zip(node.ops, node.comparators):
            lits = []
            if isinstance(op, (_ast.Eq, _ast.NotEq)) and isinstance(comp, _ast.Constant) and isinstance(comp.value, str):
                lits = [comp.value]
            elif isinstance(op, (_ast.In, _ast.NotIn)) and isinstance(comp, (_ast.Tuple, _ast.List, _ast.Set)):
                lits = [e.value for e in comp.elts if isinstance(e, _ast.Constant) and isinstance(e.value, str)]
            if lits:
                out.setdefault(node.left.id, set()).update(lits)
    return out


def _enum_fix(fname, lits, kwargs):
    import re as _re
    import sys as _sys
    for k, allowed in lits.items():
        v = kwargs.get(k)
        if not isinstance(v, str) or v in allowed or not allowed:
            continue
        words = set(_re.split(r"[^a-z0-9]+", v.lower())) - {""}
        hits = [a for a in allowed if a.lower() in words]
        if len(hits) == 1:
            kwargs[k] = hits[0]
            print(f"[toolkit] {fname}({k}={v!r}): read as {hits[0]!r} - the values this helper tests are "
                  f"{sorted(allowed)}", file=_sys.stderr, flush=True)
    return kwargs


def _wrap_enum_helpers():
    import functools
    import sys as _sys
    g = globals()
    for name in [k for k in list(g) if not k.startswith("_")]:
        obj = g[name]
        if not callable(obj) or isinstance(obj, type) or getattr(obj, "_t1_enum", False):
            continue
        base = obj
        while hasattr(base, "__wrapped__"):
            base = base.__wrapped__
        mod = getattr(base, "__module__", "") or ""
        if not mod.startswith(__name__ + "."):
            continue
        lits = _enum_literals(base)
        if not lits:
            continue

        def make(fn, fname, lits):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                if kwargs and not _internal_caller():
                    try:
                        kwargs = _enum_fix(fname, lits, dict(kwargs))
                    except Exception:  # noqa: BLE001 - an aid, never a failure cause
                        pass
                return fn(*args, **kwargs)
            wrapper._t1_enum = True
            for attr in ("_t1_memo", "_t1_rdict", "__signature__"):
                if hasattr(fn, attr):
                    try:
                        setattr(wrapper, attr, getattr(fn, attr))
                    except (AttributeError, TypeError):
                        pass
            return wrapper

        new = make(obj, name, lits)
        g[name] = new
        home = _sys.modules.get(mod)
        if home is not None and getattr(home, name, None) in (obj, base):
            setattr(home, name, new)


try:
    _wrap_enum_helpers()
except Exception:  # noqa: BLE001
    pass


# know92 (duplicate panel keys): a PANEL helper (one with date_col and id_col) handed a frame that still carries
# duplicate (date, id) rows, in a task whose text asks to de-duplicate, computed every cross-section on the raw
# rows - the numbers were all off with no error. Such a call raises a ValueError that says what to do; with a task
# silent about duplicates the call goes ahead and says how many there are.
def _panel_dups(fn, args, kwargs):
    import inspect as _inspect
    import pandas as _pd
    try:
        ba = _inspect.signature(fn).bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return None
    ba.apply_defaults()
    a = ba.arguments
    dc, ic = a.get("date_col"), a.get("id_col")
    if not isinstance(dc, str) or not isinstance(ic, str):
        return None
    for v in a.values():
        if isinstance(v, _pd.DataFrame) and dc in v.columns and ic in v.columns:
            n = int(v.duplicated([dc, ic]).sum())
            return (n, dc, ic) if n else None
    return None


def _wrap_panel_helpers():
    import functools
    import inspect as _inspect
    import os as _os
    import re as _re
    import sys as _sys
    g = globals()
    for name in [k for k in list(g) if not k.startswith("_")]:
        obj = g[name]
        if not callable(obj) or isinstance(obj, type) or getattr(obj, "_t1_panel", False):
            continue
        try:
            ps = _inspect.signature(obj).parameters
        except (TypeError, ValueError):
            continue
        if "date_col" not in ps or "id_col" not in ps:
            continue

        def make(fn, fname):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                if not _internal_caller():
                    try:
                        got = _panel_dups(fn, args, kwargs)
                    except Exception:  # noqa: BLE001
                        got = None
                    if got:
                        n, dc, ic = got
                        text = _os.environ.get("T1_INSTRUCTION", "")
                        if _re.search(r"de-?dup|duplicate", text, _re.I):
                            raise ValueError(
                                f"{fname}: the panel has {n} duplicate ({dc}, {ic}) rows and the task asks to "
                                f"de-duplicate - apply the task's dedup rule first (e.g. df.drop_duplicates([{dc!r}, "
                                f"{ic!r}], keep=...) as the task states) and pass the cleaned panel")
                        print(f"[toolkit] {fname}: {n} duplicate ({dc}, {ic}) rows in the panel", file=_sys.stderr,
                              flush=True)
                return fn(*args, **kwargs)
            wrapper._t1_panel = True
            for attr in ("_t1_memo", "_t1_rdict", "_t1_enum", "__signature__"):
                if hasattr(fn, attr):
                    try:
                        setattr(wrapper, attr, getattr(fn, attr))
                    except (AttributeError, TypeError):
                        pass
            return wrapper

        new = make(obj, name)
        g[name] = new
        base = obj
        while hasattr(base, "__wrapped__"):
            base = base.__wrapped__
        home = _sys.modules.get(getattr(base, "__module__", "") or "")
        if home is not None and getattr(home, name, None) in (obj, base):
            setattr(home, name, new)


# know92: not installed - the residual-momentum census found NO duplicate keys in its panel (the hypothesis this
# guard was written for was wrong); kept here, inactive, for a task that really carries duplicates.
