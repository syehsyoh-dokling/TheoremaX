method: svi-raw-calibration
family: local-and-stochastic-vol
keywords: svi, raw svi, stochastic volatility inspired, svi calibration, svi parameters, total implied variance fit, gatheral svi, svi rmse, svi slice, butterfly arbitrage, calendar arbitrage, natural svi
---
## Formulas
1. Raw SVI per expiry: w(k) = a + b [rho (k - m) + sqrt((k - m)^2 + s^2)], k = ln(K/F_T), w = iv^2 T.
2. Derivatives: R = sqrt((k - m)^2 + s^2); w_k = b [rho + (k - m)/R]; w_kk = b s^2 / R^3.
3. Constraints: b >= 0, |rho| < 1, s > 0, a + b s sqrt(1 - rho^2) >= 0; Lee wing b (1 + |rho|) <= 2 (T-scaled form b T if b is per year).
4. Objective: least squares on w (or vega-weighted); report RMSE in vol units sqrt(mean((sqrt(w_fit/T) - iv)^2)).
5. Start: a = min w, b = 0.1, rho = -0.5, m = 0, s = 0.1; L-BFGS-B with bounds, several restarts.
## Variants to decide
- V1 svi_target: fit target. Options: (a) total variance w; (b) implied vol. Default when silent: (a) - Gatheral 2004.
- V2 svi_moneyness: k. Options: (a) ln(K/F_T); (b) ln(K/S0). Default when silent: (a) - Gatheral 2004.
- V3 svi_rmse_unit: reported RMSE. Options: (a) vol units; (b) total-variance units. Default when silent: (a) - Gatheral-Jacquier 2014 practice.
## Invariants
- w(k) >= 0; params inside bounds; w non-decreasing in T at fixed k (calendar), butterfly g(k) >= 0.
- Forward variance positive between consecutive slices for most periods.
## Pitfalls
- Fitting all expiries jointly with one parameter set.
- Mixing S and F in k across slices.
- Bounds so tight the optimiser sits on them: report but check RMSE.
## Sources
- Gatheral (2004) SVI presentation; Gatheral and Jacquier (2014) Arbitrage-free SVI; Lee (2004) moment formula.
