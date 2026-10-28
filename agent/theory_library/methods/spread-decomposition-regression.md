method: spread-decomposition-regression
family: structural-credit-and-spreads
keywords: spread decomposition, credit spread changes, collin-dufresne goldstein martin, default component, non-default component, liquidity component, cds-bond basis, expected loss component, variance share, r-squared decomposition, newey-west t-stat, longstaff mithal neis
---
## Formulas
1. LMN split: default part = CDS spread; non-default part = bond yield spread - CDS (both bp, same maturity).
2. Expected-loss part (Elton et al): EL = lambda_P (1 - R) with physical hazard; risk premium = spread - EL.
3. Change regression (CDGM 2001): dCS_t = b0 + b1 dr_t + b2 dslope_t + b3 dVIX_t + b4 dLev_t + b5 dJump_t + e_t.
4. Covariance shares: share_k = b_k Cov(X_k, Y)/Var(Y) x 100; sum_k share_k = 100 R^2 (OLS with intercept, same ddof everywhere).
5. Shapley / LMG: average incremental R^2 of X_k over all orderings; sums to R^2.
6. HAC: Newey-West with m lags, weights 1 - j/(m+1); t = b/se_HAC.
## Variants to decide
- V1 regress_on: regression variables. Options: (a) first differences; (b) levels. Default when silent: (a) - Collin-Dufresne-Goldstein-Martin 2001.
- V2 share_method: explained-variance split. Options: (a) covariance (Pratt) shares; (b) Shapley / LMG; (c) squared standardised betas. Default when silent: (a) - Pratt 1987.
- V3 hac_lag: lag length. Options: (a) the task's m; (b) floor(4 (T/100)^(2/9)). Default when silent: (b) - Newey-West 1994.
- V4 basis_sign: CDS-bond basis. Options: (a) bond spread - CDS (non-default part); (b) CDS - bond spread (market basis). Default when silent: (a) - Longstaff-Mithal-Neis 2005.
## Invariants
- Shares + residual share = 100; 0 <= R^2 <= 1.
- Spread rises with leverage and VIX, falls with the riskless rate (CDGM signs).
## Pitfalls
- Mixing percent points and bp; FRED missing as '.' -> NaN, align by date before differencing.
- Var(Y) ddof differing from the covariances' ddof.
- ffill of a stopped series BEFORE the merge leaves NaN after its end.
## Sources
- Collin-Dufresne-Goldstein-Martin (2001); Longstaff-Mithal-Neis (2005); Elton-Gruber-Agrawal-Mann (2001); Newey-West (1987, 1994).
