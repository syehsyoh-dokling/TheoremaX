method: garch-filtered-historical-simulation
family: var-es-methods
keywords: filtered historical simulation, fhs, garch-filtered, barone-adesi, stationary bootstrap, politis romano, standardised residuals, bootstrap residual rows, geometric block length, var_fhs, es_fhs, volatility-scaled historical simulation
---
## Formulas
1. z_t = r_t/sigma_t, r and sigma in the SAME units.
2. Scenarios for day t: r*_b = sum_i w_i z_{b,i} sigma_{i,t} (whole rows of Z keep dependence; no extra Cholesky).
3. VaR_t = -Q_{1-c}(r*); ES_t = -mean(r* | r* <= -VaR_t).
4. Stationary bootstrap (Politis-Romano 1994): I_1 ~ U{0..T-1}; then with prob p a new I_k ~ U{0..T-1}, else I_k = (I_{k-1}+1) mod T. Mean block 1/p.
5. Forecast form scales by sigma_{T+1}; per-day form by sigma_t.
## Variants to decide
- V1 bootstrap_scheme: residual draws. Options: (a) all historical z rows; (b) iid row bootstrap; (c) stationary bootstrap, circular. Default when silent: (a) - Barone-Adesi et al. 1999; (c) when Politis-Romano is named.
- V2 rng_order: bootstrap draws. Options: (a) uniform first index, then per step one uniform jump test and an integer only on a jump; (b) vectorised arrays of starts and flags. Default when silent: (a) - Politis-Romano 1994.
- V3 sample_set: rows across days. Options: (a) one set of B rows reused every day; (b) fresh rows per day. Default when silent: (a) - "construct B rows, scale per day" wording.
- V4 tail_quantile: percentile rule. Options: (a) np.percentile linear; (b) order statistic. Default when silent: (a) - Hyndman-Fan type 7.
## Invariants
- var_fhs > 0, es_fhs >= var_fhs daily; Z columns mean ~0, std ~1.
## Pitfalls
- Decimal r over percent sigma (z off by 100).
- Broadcasting (B x N) rows against (T x N) sigma; loop or einsum per day.
## Sources
- Barone-Adesi-Giannopoulos-Vosper 1999; Hull-White 1998; Politis-Romano 1994; Christoffersen 2012.
