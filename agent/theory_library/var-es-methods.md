family: var-es-methods
category: risk-management
keywords: value at risk, var, expected shortfall, es, cvar, historical simulation, age-weighted, brw, filtered historical simulation, parametric, variance-covariance, delta-normal, student-t, monte carlo var, 10-day, sqrt of time, overlapping returns, kernel density, component var, marginal var, incremental var, euler, notional, confidence level, stationary bootstrap, fhs, kde var, kelly, var budget
---
## Worked frame
DIKETAHUI: prices [T+1 x N] or returns [T x N] (decimal), weights or positions [N], levels [C], horizon (days), notional. Check: pivot, inner-join dates, oldest -> newest, weights in column order.
DITANYAKAN: VaR, ES one row per (method, level), positive loss (fraction or currency); component VaR [N].
RUMUS: card Core 1 -> 2-7 (method) -> 9 (horizon); toolkit portfolio_returns, historical_var_es, parametric_var_es, component_var, scale_horizon.
JAWAB: 1) asset returns [T x N]. 2) portfolio returns [T]. 3) per method and level -> (VaR, ES). 4) horizon scaling. 5) Sigma [N x N] -> components [N]. 6) fraction x notional -> currency. 7) check: ES >= VaR > 0, VaR rises with level, components sum to VaR; then write.

## Core formulas
1. Loss L = -r (or -P&L). VaR_c = Q_c(L) = -Q_{1-c}(r); ES_c = E[L | L >= VaR_c] = (1/(1-c)) int_c^1 VaR_u du. Report both as POSITIVE numbers; dollar = notional * return-VaR.
2. HS: q = np.quantile(r, 1-c) (numpy default 'linear', type 7: h=(n-1)p); VaR = -q; ES = -mean(r[r <= q]).
3. Normal: VaR = z_c sigma - mu; ES = sigma phi(z_c)/(1-c) - mu, z_c = norm.ppf(c) (exact, never 2.33/1.645 literals).
4. Student-t loc-scale (scipy t.fit -> df, loc, scale): q = t_nu.ppf(c); VaR = scale q - loc; ES = scale f_nu(q)/(1-c) (nu+q^2)/(nu-1) - loc. Unit-variance t: scale = sigma sqrt((nu-2)/nu).
5. Age-weighted HS (Boudoukh-Richardson-Whitelaw 1998): w_i = lambda^{n-i}(1-lambda)/(1-lambda^n), i=1 oldest..n newest (newest weight largest). Sort returns ascending with their weights, cumsum; VaR at first index with cum >= 1-c; ES = sum(w r)/sum(w) over that tail.
6. Filtered HS (Barone-Adesi 1999; Hull-White 1998): z_t = r_t/sigma_t (GARCH/EWMA); scenarios r* = z sigma_{T+1}; apply HS. Multi-asset: resample WHOLE rows of z (keeps cross-correlation) and scale each column by its own sigma_{i,T+1}; do NOT also apply a Cholesky of R.
7. Delta-normal: sigma_p = sqrt(w' Sigma w) (w in currency -> currency vol). VaR = z sigma_p. Marginal MVaR_i = z (Sigma w)_i / sigma_p; component CVaR_i = w_i MVaR_i; sum_i CVaR_i = VaR (Euler, degree-1 homogeneity); pct_i = CVaR_i/VaR. Incremental VaR = VaR(w+a) - VaR(w) ~ a' MVaR. ES version: replace z by phi(z)/(1-c).
8. Monte Carlo: simulate factors (Cholesky of Sigma), full revaluation of P&L, then estimator 2 on simulated P&L.
9. Horizon h: iid parametric VaR_h = sqrt(h) VaR_1 (with drift: z sigma sqrt(h) - h mu). Simulation methods: h-day OVERLAPPING sums of log returns (rolling(h).sum(), n-h+1 values) unless "non-overlapping" is said. Simple returns compound: prod(1+r)-1.
10. Cornish-Fisher (modified) VaR: z = norm.ppf(1-c) (NEGATIVE), S, K = skew and EXCESS kurtosis of RETURNS; z_cf = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36; VaR = -(mu + z_cf sigma).
11. KDE (Gaussian, Silverman): CDF on a fine grid (data range +- several sd, 1e4-1e5 points), quantile by np.interp; ES = int_{VaR} x f(x) dx/(1-c).

## Conventions and defaults when the task is silent
- Confidence c = 0.95/0.99; "alpha = 0.01" or "1% VaR" means tail prob 0.01, same VaR. Detect: alpha < 0.5 is a tail probability.
- Quantile: numpy linear unless the text says lower / nearest / order statistic. Order-statistic cue ("k-th worst", Jorion, Basel): VaR = sorted losses descending[ceil(n(1-c))-1]; "smallest x with F(x) >= c" = method='inverted_cdf'.
- HS ES: mean of losses >= VaR (non-strict). Strict ">" only when stated.
- Parametric mean: include sample mean unless the text says zero-mean, RiskMetrics or EWMA (those assume mu = 0).
- Normal fit: "MLE" -> ddof=0; otherwise sample std ddof=1 (pandas default). Checkers are usually tolerant between the two; the stated word wins.
- Portfolio: fixed weights rebalanced daily, r_p = sum w_i r_i on simple returns. If the text says average of log returns, do exactly that (not ln sum w e^r).
- Log-return VaR in dollars: notional * (-q) by default; notional*(1-exp(q)) only if "exact loss" is requested.
- 10-day regulatory = sqrt(10) scaling of 1-day for parametric methods (Basel 1996).

## Completing missing inputs
- No lambda given: BRW 0.98 or 0.99; EWMA 0.94 daily (RiskMetrics 1996).
- No df for t: fit by MLE (scipy t.fit), or method of moments nu = 4 + 6/K (K excess kurtosis > 0).
- Dollar-delta vector = position value * beta/delta; covariance from returns.
- ES from VaR under normal: ES/VaR at 97.5% ES ~ 99% VaR (2.338 vs 2.326 sigma), the FRTB calibration.

## Invariants to self-check
- ES >= VaR at same c; VaR_99 >= VaR_95; VaR > 0 for normal-ish data.
- Sum of component VaR == total VaR (1e-10); sum pct == 1; negative component allowed (hedge).
- Parametric 10-day / 1-day == sqrt(10) exactly.
- BRW with lambda -> 1 reproduces an un-interpolated HS; HS VaR lies between adjacent order statistics.
- ES subadditive: ES(portfolio) <= sum ES(assets).
- Student-t VaR at 99% > normal VaR at equal variance; at 95% it can be SMALLER - not a bug.

## Variants to decide
- V1 return_type: returns. Options: (a) simple, weighted; (b) average of the asset log-returns sum w_i ln(P_t/P_t-1) (toolkit kind="log_avg"); (c) log of portfolio gross return ln(sum w_i P_t/P_t-1) (kind="log_portfolio"). Default when silent: (a) - Jorion 2007.
- V2 hs_quantile: HS quantile. Options: (a) np.quantile linear; (b) order statistic sorted losses[ceil(n(1-c))-1]; (c) inverted_cdf. Default when silent: (a) - Hyndman-Fan type 7.
- V3 es_tail: HS ES tail. Options: (a) losses >= VaR; (b) > VaR strict; (c) Acerbi-Tasche fractional. Default when silent: (a) - Dowd 2005.
- V4 param_mean: parametric mean. Options: (a) include sample mean; (b) mu=0. Default when silent: (a) - Jorion 2007; (b) RiskMetrics / EWMA.
- V5 horizon_scaling: h-day VaR. Options: (a) sqrt(h) x 1-day; (b) overlapping h-day returns; (c) non-overlapping. Default when silent: (a) - Basel 1996 for parametric; simulation uses (b).
- V6 sigma_ddof: std estimator. Options: (a) ddof=1; (b) ddof=0 (MLE). Default when silent: (a) - Jorion 2007.
- V7 dollar_loss: log-return VaR in currency. Options: (a) notional x (-q); (b) notional x (1-e^q). Default when silent: (a) - Jorion 2007.
- V8 fhs_resampling: residual draws in FHS. Options: (a) iid whole-row bootstrap; (b) stationary bootstrap (geometric blocks, circular); (c) all historical z rows without resampling. Default when silent: (c) - Barone-Adesi et al. 1999; (b) when Politis-Romano is named.
- V9 kde_bandwidth: KDE VaR smoothing. Options: (a) Scott (scipy gaussian_kde default); (b) Silverman; (c) cross-validation. Default when silent: (a) - scipy; (b) when "Silverman" is named.
- V10 var_constrained_size: sizing under a VaR limit. Options: (a) min(Kelly fraction, VaR budget / unit VaR, leverage cap); (b) VaR budget only. Default when silent: (a) - Thorp 2006 / Jorion 2007.

## Pitfalls
- Sign: reporting the left-tail quantile (negative) instead of the loss; double negation.
- np.percentile takes 5 (0-100 scale); np.quantile takes 0.05.
- Using ES = mean of all losses, or mean of r < q vs <= q when stated.
- Forgetting (nu+q^2)/(nu-1) in t ES, or mixing loc-scale t with unit-variance t.
- BRW: weights in time order but not re-sorted with the returns; newest given smallest weight; interpolating when the text says "first cumulative weight >= 1-c".
- 10-day HS by sqrt(10) when overlapping sums were requested (or vice versa).
- Component VaR with Sigma w * sigma_p instead of / sigma_p.
- Using portfolio weights that do not sum to 1 without noticing; long-format data not pivoted and aligned by date (inner join) before combining.
- Monte Carlo: legacy np.random vs default_rng stream mismatch when a seed is pinned; follow the stated generator and draw order.

- Student-t with scipy t.fit: the fitted scale is already the t scale; rescaling it by sqrt((nu-2)/nu) understates VaR (student_t_mle_var_es, or parametric_var_es(t_scale="scale")).
- Backtest hits compare the realised LOSS -r_t with VaR_t; a helper that takes returns must get returns, not losses (passing -r counts gains). quad integrands must return a 0-d float (kde(x)[0]).
- Shapes differ: VaR helpers return (VaR, ES); analytic_loss_var_es, student_t_mle_var_es and ewma_portfolio_vol return DICTS (var_es_one always returns a tuple). EWMA covariance: keep the seed the task or template uses (outer(r0, r0) -> seed='first'), it changes every element.
## Sources
Jorion, Value at Risk (3rd ed.); Hull, Risk Management and Financial Institutions; RiskMetrics Technical Document (1996); Boudoukh-Richardson-Whitelaw (1998); Barone-Adesi et al. (1999); Hull-White (1998); McNeil-Frey-Embrechts (2015) ch.2; Acerbi-Tasche (2002); Tasche (2000) Euler allocation; BCBS FRTB (2019).
