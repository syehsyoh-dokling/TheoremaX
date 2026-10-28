method: black-litterman-posterior
family: portfolio-construction
keywords: black-litterman, black litterman, equilibrium returns, implied returns, reverse optimisation, views, pick matrix, omega, tau, posterior mean, he-litterman, idzorek, view confidence, risk aversion delta
---
## Formulas
1. Pi = delta Sigma w_mkt; delta = (E[r_m] - rf)/sigma_m^2.
2. mu_BL = [(tau Sigma)^-1 + P'Omega^-1 P]^-1 [(tau Sigma)^-1 Pi + P'Omega^-1 Q] = Pi + tau Sigma P'(P tau Sigma P' + Omega)^-1 (Q - P Pi).
3. M = [(tau Sigma)^-1 + P'Omega^-1 P]^-1; Sigma_post = Sigma + M.
4. Omega = diag(P tau Sigma P'); Idzorek c -> omega_k = ((1-c)/c) p_k tau Sigma p_k'.
5. w = (delta Sigma)^-1 mu_BL (or Sigma_post); QP for long-only.
6. Regime overlay: Sigma from the current regime's days.
## Variants to decide
- V1 bl_weight_cov: covariance in the weight step. Options: (a) Sigma; (b) Sigma + M. Default when silent: (a) - Black-Litterman 1992.
- V2 omega_form: view uncertainty. Options: (a) diag(P tau Sigma P'); (b) Idzorek confidence; (c) given view variances. Default when silent: (a) - He-Litterman 1999.
- V3 tau_value: tau when unstated. Options: (a) 0.05; (b) 0.025; (c) 1/T. Default when silent: (a) - He-Litterman 1999.
- V4 bl_normalise: weight scaling. Options: (a) rescale to sum 1; (b) gross normalisation sum|w| = 1; (c) leave unnormalised. Default when silent: (a) - Idzorek 2005.
## Invariants
- No views -> mu_BL = Pi, w = w_mkt; posterior between prior and views.
## Pitfalls
- tau in Pi; Omega as std; annual Q with daily Sigma (daily Q = Q/P, omega = s^2/P, not (s/P)^2).
- Inverting singular Omega at full confidence (use the second form of 2).
## Sources
- Black-Litterman 1992; He-Litterman 1999; Idzorek 2005; Walters 2014.
