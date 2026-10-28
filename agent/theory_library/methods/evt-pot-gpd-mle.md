method: evt-pot-gpd-mle
family: evt-and-tails
keywords: peaks over threshold, pot, generalized pareto, gpd, genpareto fit, floc=0, threshold quantile, exceedances, excesses, hill estimator, tail index, garch-evt, mcneil frey, evt var, evt es
---
## Formulas
1. L = -r; u = Q_q0(L) with the stated quantile rule; exceedances L > u; y = L - u; N_u, n.
2. (xi, 0, beta) = genpareto.fit(y, floc=0).
3. VaR_p = u + (beta/xi)[((n/N_u)(1-p))^(-xi) - 1]; ES_p = VaR_p/(1-xi) + (beta - xi u)/(1-xi).
4. Hill: xi_H = (1/k) sum_{i<=k} ln X_(i) - ln X_(k+1), descending order.
5. GARCH-EVT: z = r/sigma (same units), losses -z, own threshold, POT -> VaR_z, ES_z; VaR = sigma_dec VaR_z, ES = sigma_dec ES_z.
## Variants to decide
- V1 threshold_rule: u. Options: (a) empirical loss quantile at a stated level; (b) u = X_(k+1); (c) mean-excess plot. Default when silent: (a) - McNeil-Frey 2000.
- V2 quantile_method: interpolation. Options: (a) numpy linear; (b) lower; (c) inverted_cdf. Default when silent: (a) - Hyndman-Fan type 7; a pinned method applies to every call.
- V3 sigma_scale: GARCH-EVT vol. Options: (a) forecast sigma_{T+1}; (b) last in-sample sigma_T. Default when silent: (a) - McNeil-Frey 2000.
- V4 hill_k: sample fraction. Options: (a) stated k; (b) k = N_u. Default when silent: (b) - Embrechts-Kluppelberg-Mikosch 1997.
## Invariants
- u < VaR_p < ES_p; beta > 0; GARCH-EVT VaR of the order of the unconditional VaR.
## Pitfalls
- Decimal r over percent sigma: GARCH-EVT VaR 100x too small.
- Losses fed to a helper that expects returns (negated twice).
- Fitting raw losses, or a free location.
## Sources
- McNeil-Frey 2000; Smith 1987; Hill 1975; Embrechts-Kluppelberg-Mikosch 1997.
