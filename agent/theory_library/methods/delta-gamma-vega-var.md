method: delta-gamma-vega-var
family: exotic-options
keywords: delta-gamma-vega var, greek var decomposition, delta var pct, gamma var pct, vega var pct, variance share, vol of vol, volatility of volatility, option portfolio var, parametric var options, delta-normal var, risk decomposition
---
## Formulas
sigma_d = annual vol / sqrt 252 (1-day return sd); n contracts, Greeks per contract.
1. Delta term: V_D = (n Delta S sigma_d)^2.
2. Gamma term (normal returns, E[z^4] - 1 = 2): V_G = 0.5 (n Gamma S^2 sigma_d^2)^2, i.e. variance of 0.5 Gamma (S sigma_d z)^2.
3. Vega term: V_V = (n Vega sigma_vv,d)^2; sigma_vv,d = daily sd of vol changes = std(diff(annualised conditional vol), ddof=1) [annualised x sqrt 252, then back to 1 day / sqrt 252].
4. Total variance V = V_D + V_G + V_V (independent factors); VaR = z_alpha sqrt(V); ES = phi(z_alpha)/(1 - alpha) sqrt(V).
5. Shares on 0-100: pct_i = 100 V_i / V.
## Variants to decide
- V1 share_basis: decomposition. Options: (a) variance shares summing to 100; (b) Euler component VaR shares; (c) std shares. Default when silent: (a) - Jorion 2007 (independent factors: (a) equals (b)).
- V2 vov_scaling: vol-of-vol horizon. Options: (a) sd of daily vol changes (1-day); (b) annualised sd. Default when silent: (a) - horizon consistency, Jorion 2007.
- V3 gamma_term: gamma variance. Options: (a) 0.5 (Gamma S^2 sigma_d^2)^2; (b) omit (delta-vega only). Default when silent: (a) - Britten-Jones and Schaefer 1999.
## Invariants
- Shares sum to 100 and are >= 0; VaR > 0; z from the stated confidence.
## Pitfalls
- Dividing the vol-of-vol by sqrt 252 twice.
- Mixing percent-scaled GARCH output (x100 returns) with decimal returns in one recursion.
- Gamma term without the 0.5 factor or without S0^2; portfolio VaR reported as variance (delta_gamma_vega_var).
## Sources
- Jorion, Value at Risk (2007); Britten-Jones and Schaefer (1999); RiskMetrics Technical Document (1996).
