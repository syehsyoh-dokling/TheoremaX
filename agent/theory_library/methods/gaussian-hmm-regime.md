method: gaussian-hmm-regime
family: portfolio-construction
keywords: hidden markov model, hmm, baum-welch, baum welch, viterbi, forward-backward, regime switching, bull bear, two-state, transition matrix, filtered probability, smoothed probability, regime covariance, em algorithm
---
## Formulas
1. Emission log b_k(x) = log N(x; mu_k, sigma_k^2) computed directly (no exp then log).
2. Log forward-backward; beta_t(i) = lse_j[log A_ij + log b_j(x_{t+1}) + beta_{t+1}(j)].
3. M-step: pi = gamma_0; A_ij = sum_{t<T} xi_t(i,j)/sum_{t<T} gamma_t(i); mu_k gamma-weighted; sigma_k^2 = sum gamma (x - mu_k_new)^2/sum gamma (MLE).
4. Viterbi: delta_t(j) = max_i[delta_{t-1}(i) + log A_ij] + log b_j(x_t), backtrack.
5. Filtered P(s_t | x_1..t) = normalised alpha_t; smoothed uses all data.
6. Regime moments from days assigned to a state (gamma > 0.5), fallback to full sample when too few.
## Variants to decide
- V1 hmm_init: EM starting values. Options: (a) deterministic quantile split of returns, diagonal-heavy A; (b) k-means; (c) seeded random. Default when silent: (a) - Rabiner 1989.
- V2 state_prob: probability used for decisions. Options: (a) filtered; (b) smoothed; (c) Viterbi path. Default when silent: (a) - Hamilton 1989 (real-time use).
- V3 state_label: state naming. Options: (a) bull = higher mean; (b) calm = lower variance. Default when silent: (a) - Hamilton 1989.
- V4 em_stop: convergence rule. Options: (a) |dlogL| < tol or max_iter; (b) max parameter change < tol. Default when silent: (a) - Rabiner 1989.
## Invariants
- Rows of A and gamma sum to 1; logL non-decreasing; relabelling permutes A too.
## Pitfalls
- M-step keeping A and pi fixed; backward with b(x_t); sigma updated with the old mu.
- log A computed once before the loop, so the E-step never sees the new A (gaussian_hmm_fit).
- Smoothed probabilities in a walk-forward backtest.
## Sources
- Rabiner 1989; Hamilton 1989; Baum et al. 1970; Viterbi 1967.
