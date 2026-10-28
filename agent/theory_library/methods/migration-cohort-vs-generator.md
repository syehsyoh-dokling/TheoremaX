method: migration-cohort-vs-generator
family: rating-migration
keywords: cohort estimator, cohort-weighted average, pooled transition counts, generator matrix, matrix logarithm, diagonal adjustment, aalen-johansen, duration estimator, continuous-time migration, hidden defaults, withdrawals censored, chi-squared homogeneity, markov test, cumulative default probability, cerep, deviation matrix
---
## Formulas
1. Cohort: C_ij(t) counts, P_ij(t) = C_ij(t)/sum_k C_ik(t); empty row -> identity row.
2. Pooled average: P_bar_ij = sum_t C_ij(t)/sum_t sum_k C_ik(t).
3. Duration MLE (Lando-Skodeberg): q_ij = N_ij/int Y_i(s) ds, q_ii = -sum_{j!=i} q_ij; P(t) = expm(Q t).
4. Aalen-Johansen: P(s,t) = prod_k (I + dA(t_k)), dA_ij = dN_ij/Y_i (i != j), dA_ii = -dN_i/Y_i.
5. Generator from P: Q = real(logm(P)); DA fix: negative off-diagonals -> 0, q_ii = -sum_{j!=i} q_ij; default row 0.
6. Cumulative PD_i(h) = [P^h]_{i,D}.
7. Homogeneity per origin i: cohorts x destinations table, E = R C/N, X2 = sum (O - E)^2/E, df = (r-1)(k-1), p = chi2.sf.
8. Hidden defaults: max(true defaults - visible defaults, 0) added to C_iD (row total grows).
## Variants to decide
- V1 estimator: transition estimator. Options: (a) cohort; (b) duration MLE generator; (c) Aalen-Johansen. Default when silent: (a) - S&P / Moody's studies.
- V2 averaging: multi-year matrix. Options: (a) pool counts; (b) mean of annual matrices. Default when silent: (a) - Schuermann 2008.
- V3 generator_fix: invalid log. Options: (a) diagonal adjustment; (b) weighted adjustment; (c) duration MLE. Default when silent: (a) - Israel-Rosenthal-Wei 2001.
- V4 sparse_filter: chi-squared sparsity. Options: (a) drop destination columns whose total over cohorts is below the threshold; (b) require expected counts >= 5; (c) none. Default when silent: (b) - Cochran 1954.
## Invariants
- Rows sum to 1; diagonal is the row max; Default row = e_D; PD rises with horizon; Q rows sum 0.
## Pitfalls
- Align by label (BOP rows, EOP columns), never by position; np.divide where= needs out=.
## Sources
- Lando-Skodeberg (2002); Aalen-Johansen (1978); Israel-Rosenthal-Wei (2001); Schuermann (2008).
