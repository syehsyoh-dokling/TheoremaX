method: fama-macbeth-newey-west
family: factor-models-and-sorts
keywords: fama-macbeth, fama macbeth, cross-sectional regression, risk premium, factor premium, newey-west, newey west, hac standard error, bartlett kernel, two-pass regression, slope t-stat, ic t-stat, expanding window forecast
---
## Formulas
1. Per date t: OLS r_{i,t+1} = g_{0,t} + g_t' z_{i,t} + e on names with target and ALL regressors non-missing; skip t if n < minimum.
2. g = mean_t g_t; plain se = std(g_t, ddof=1)/sqrt(T).
3. NW: u = g_t - g; gamma_l = (1/T) sum_{t>l} u_t u_{t-l}; S = gamma_0 + 2 sum_{l<=L} (1 - l/(L+1)) gamma_l; se = sqrt(S/T).
4. Lag: stated; else floor(4 (T/100)^(2/9)); alt floor(T^(1/4)).
5. Shanken EIV: var x (1 + g' Sigma_f^-1 g) when betas are estimated.
6. OOS forecast at t: mean of slopes from valid dates strictly before t, times z_t.
## Variants to decide
- V1 fm_se_type: premium standard error. Options: (a) plain std(ddof=1)/sqrt(T); (b) Newey-West Bartlett; (c) Shanken-corrected. Default when silent: (a) - Fama-MacBeth 1973.
- V2 nw_gamma_scale: autocovariance divisor. Options: (a) 1/T; (b) 1/(T-1) or statsmodels HAC small-sample form. Default when silent: (a) - Newey-West 1987.
- V3 nw_lag_rule: lag when unstated. Options: (a) stated lag; (b) floor(4(T/100)^(2/9)); (c) floor(T^(1/4)). Default when silent: (b) - Newey-West 1994.
- V4 cs_weights: cross-section weights. Options: (a) OLS; (b) WLS by market cap; (c) GLS. Default when silent: (a) - Fama-MacBeth 1973.
## Invariants
- T = dates with a valid regression; NW with L=0 equals plain se with ddof 0.
## Pitfalls
- Pooled OLS; integer 2/9; Bartlett 1-l/L; dropping controls the model lists.
- Adding winsorisation the task did not ask for.
## Sources
- Fama-MacBeth 1973; Newey-West 1987, 1994; Shanken 1992; Cochrane, Asset Pricing ch. 12.
