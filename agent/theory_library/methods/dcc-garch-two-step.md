method: dcc-garch-two-step
family: volatility-models
keywords: dcc-garch, dcc garch, dynamic conditional correlation, engle 2002, two-step estimation, quasi-maximum likelihood, dcc log-likelihood, dcc_alpha, dcc_beta, conditional correlation matrix, conditional covariance matrix, pairwise correlations, portfolio conditional volatility, qbar
---
## Formulas
1. Stage 1: univariate GARCH per asset -> sigma_{i,t} (decimal), z_{i,t} = r_{i,t}/sigma_{i,t}.
2. Qbar = Z'Z/T; Q_1 = Qbar; Q_t = (1-a-b)Qbar + a z_{t-1}z_{t-1}' + b Q_{t-1}.
3. R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}; H_t = D_t R_t D_t, D_t = diag(sigma_{1,t}..sigma_{N,t}).
4. Stage-2 LL (correlation part) = -0.5 sum_t [ln|R_t| + z_t'R_t^{-1}z_t - z_t'z_t]; full LL adds -0.5 sum [N ln 2pi + 2 ln|D_t| + z_t'z_t].
5. sigma_p,t = sqrt(w'H_t w); VaR = z_c sigma_p,t; ES = sigma_p,t phi(z_c)/(1-c); annual = sqrt(252) sigma_p,t.
## Variants to decide
- V1 qbar_form: target matrix. Options: (a) covariance Z'Z/T; (b) correlation of Z. Default when silent: (a) - Engle 2002.
- V2 loglik_report: reported likelihood. Options: (a) correlation-part LL; (b) full Gaussian LL. Default when silent: (a) - Engle-Sheppard 2001 two-step.
- V3 q_start: initial Q. Options: (a) Q_1 = Qbar; (b) Q_0 = Qbar then update with z_0. Default when silent: (a) - Engle 2002.
- V4 constraint: a+b<1 enforcement. Options: (a) L-BFGS-B bounds plus penalty; (b) SLSQP inequality. Default when silent: (a) - text names L-BFGS-B.
## Invariants
- R_t unit diagonal, |rho| < 1, symmetric PSD; diag(H_t) = sigma_t^2.
- Violation rate of 1% VaR in a plausible band (not zero); a > 0, b > 0, a+b < 1.
## Pitfalls
- H_t built from sqrt(diag Q_t) instead of GARCH sigma (VaR wrong scale, zero violations).
- Using Q_t^{-1} instead of R_t^{-1} in the LL.
- Slow T-loop in the optimiser: precompute outer products once.
- Qbar = Z'Z/T and Q_0 = Qbar; dcc_fit / dcc_garch_portfolio do the R_t likelihood in one call.
## Sources
- Engle 2002; Engle-Sheppard 2001; Christoffersen 2012 ch.7.
