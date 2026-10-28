method: creditmetrics-migration-simulation
family: portfolio-credit-risk
keywords: creditmetrics, rating migration simulation, transition matrix thresholds, rating transition thresholds, single-factor gaussian copula, upgrade rate, downgrade rate, default rate, loss by initial rating, credit var monte carlo, gupton finger bhatia
---
## Formulas
1. Transition row (best..worst, Default last) in decimals, sum 1.
2. Worst-first cumulative C_D = p_D, C_k = sum of p_j for j at or below k; z_k = Phi^-1(C_k), increasing toward the best rating (best = +inf).
3. Z_i = sqrt(rho) M + sqrt(1-rho) eps_i; Z < z_D -> default; state index = np.searchsorted(z_row, Z, side='right') per origin rating (0 = Default).
4. Default-only loss L = sum_i LGD_i F_i 1{default_i}; mark-to-market revalues with the new rating's forward curve.
5. VaR_q = quantile of L; ES_q = mean(L | L >= VaR_q).
## Variants to decide
- V1 threshold_direction: Z to state map. Options: (a) default in the left tail, worst-first cumulation; (b) best-first cumulation from the right tail. Default when silent: (a) - Gupton-Finger-Bhatia 1997.
- V2 loss_mode: what counts as loss. Options: (a) default losses only; (b) full revaluation. Default when silent: (a) - CreditMetrics TD when only recovery is given; (b) with forward curves.
- V3 rating_bucket_loss: loss by initial rating. Options: (a) mean of the bucket total loss (buckets sum to the portfolio mean); (b) mean per bond. Default when silent: (a) - CreditMetrics TD contributions.
- V4 draw_order: draws. Options: (a) M (n_sim,) then eps (n_sim, n_bonds), one default_rng; (b) eps first. Default when silent: (a) - text order.
## Invariants
- Bucket default frequency ~ matrix default column; upgrade + stay + downgrade + default = 1.
## Pitfalls
- Bond masks applied on the simulation axis: use Z[:, mask].
- Default counted as downgrade; percent matrix not divided by 100.
- 'Random seed N' scripts use the legacy np.random.seed stream, M first (creditmetrics_default_sim, rng='legacy').
## Sources
- Gupton-Finger-Bhatia 1997; Hull RMFI ch.21.
