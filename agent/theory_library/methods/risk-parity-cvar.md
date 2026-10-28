method: risk-parity-cvar
family: portfolio-construction
keywords: risk parity, equal risk contribution, erc, inverse volatility, risk budgeting, cvar, conditional value at risk, expected shortfall, rockafellar-uryasev, absorption ratio, regime risk budget, spinu
---
## Formulas
1. RC_i = w_i (Sigma w)_i/sqrt(w'Sigma w); ERC RC_i = sigma_p/N; budgets RC_i = b_i sigma_p.
2. Spinu: min 0.5 y'Sigma y - sum b_i ln y_i, w = y/sum y; or SLSQP on sum (RC_i - RC_j)^2.
3. Inverse vol w_i = (1/sigma_i)/sum(1/sigma_j).
4. Historical CVaR of L = -w'r: VaR = quantile_a(L); CVaR = mean(L | L >= VaR).
5. R-U LP: min z + sum u_s/((1-a)S), u_s >= -w'r_s - z, u_s >= 0.
6. Absorption ratio: eigvalsh of correlation, top-k sum/N; regimes by threshold.
## Variants to decide
- V1 rp_type: risk parity definition. Options: (a) ERC on full covariance; (b) inverse volatility. Default when silent: (a) - Maillard-Roncalli-Teiletche 2010.
- V2 cvar_estimator: tail average. Options: (a) mean of losses at or beyond the empirical VaR; (b) mean of worst ceil((1-a)S) losses; (c) Gaussian closed form. Default when silent: (b) - Acerbi-Tasche 2002.
- V3 rebalance_day: monthly rebalance date. Options: (a) last trading day, weights earn from the next day; (b) first trading day. Default when silent: (a) - month-end convention (Roncalli 2013).
- V4 cov_window: estimation window. Options: (a) trailing W valid returns up to the rebalance close; (b) expanding. Default when silent: (a) - Roncalli 2013.
## Invariants
- sum RC = sigma_p; CVaR >= VaR; weights >= 0 and sum to the regime budget.
## Pitfalls
- Diagonal covariance (gives inverse vol); weights set at close t earning return t.
- Cleaning anomalies after taking the window tail (clean first).
## Sources
- Maillard-Roncalli-Teiletche 2010; Spinu 2013; Roncalli 2013; Rockafellar-Uryasev 2000; Acerbi-Tasche 2002; Kritzman-Li-Page-Rigobon 2011.
