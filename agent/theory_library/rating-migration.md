family: rating-migration
category: credit
keywords: transition matrix, migration matrix, rating migration, cohort, generator matrix, matrix logarithm, absorbing default, cumulative default probability, markov property, time homogeneity, chi-squared, withdrawn, nr, censored, duration estimator, matrix power, s&p, moody's, cerep, investment grade, speculative grade, default rate, aalen-johansen, hidden defaults, withdrawals, cohort-weighted average, homogeneity test, deviation matrix, selective default
---
## Worked frame
DIKETAHUI: transitions (issuer, from, to) [n_obs], or count/percent matrix [R x R]; horizons [H] (years). Check: rows = from, columns = to; labels mapped; percent -> probability.
DITANYAKAN: P [R x R], cumulative PD [R x H], generator [R x R] (per year), test statistics [R].
RUMUS: card Core 1 or 2 -> 3 -> 4 -> 5-6; toolkit cohort_matrix, matrix_power_migration, generator_from_matrix, cumulative_default_probs.
JAWAB: 1) map ratings to scale [R]; drop withdrawn. 2) counts [R x R], totals [R]. 3) P = counts / totals; default row absorbing. 4) powers -> PD [R x H]. 5) generator [R x R] with its fix. 6) probability -> percent if asked. 7) check: rows sum to 1, PD non-decreasing in horizon, generator rows sum to 0; then write.

## Core formulas
1. Cohort (discrete) estimator: P_ij = N_ij / N_i, N_i = issuers in i at period start (after censoring).
   Pooled multi-year: P_bar_ij = sum_t N_ij(t) / sum_t N_i(t) (issuer-weighted), NOT mean of annual P(t).
2. Duration / continuous-time estimator (Lando-Skodeberg 2002): q_ij = N_ij(0,T) / int_0^T Y_i(s) ds
   (transitions i->j over total issuer-years spent in i), q_ii = -sum_{j!=i} q_ij; P(t) = expm(Q t).
   Gives non-zero AAA default prob through intermediate paths; cohort gives exactly 0 if none observed.
3. h-period matrix (Markov, time-homogeneous): P(h) = P^h (np.linalg.matrix_power). Cumulative
   PD_i(h) = [P^h]_{i,D}. Marginal PD(h) = PD(h)-PD(h-1); conditional (forward) PD =
   (PD(h)-PD(h-1))/(1-PD(h-1)); average annual PD = 1-(1-PD(h))^(1/h).
4. Generator from annual P: Q = real(scipy.linalg.logm(P)); series log(P) = sum_k (-1)^(k+1)(P-I)^k/k.
   A valid generator needs q_ij >= 0 (i!=j), rows sum 0. Fixes (Israel-Rosenthal-Wei 2001;
   Kreinin-Sidelnikova 2001): DA (diagonal adjustment, default): set negative off-diagonals to 0,
   then q_ii = -sum_{j!=i} q_ij. WA (weighted adjustment): zero negatives, then subtract row excess in
   proportion to |q_ij|. Default row of Q = 0 after any fix.
5. Homogeneity test per origin i: table rows = cohorts, cols = destinations; E_tj = R_t K_j / N;
   X2 = sum (O-E)^2/E; df = (r-1)(k-1); p = chi2.sf(X2, df). LR alternative G2 = 2 sum O ln(O/E)
   (Anderson-Goodman 1957), same df.
6. CI for a cell: Wald p +/- z sqrt(p(1-p)/n); for rare defaults use Clopper-Pearson:
   lo = Beta.ppf(a/2, x, n-x+1), hi = Beta.ppf(1-a/2, x+1, n-x); x=0 -> lo=0, hi=1-(a/2)^(1/n)
   (one-sided 95% ~ 3/n, rule of three). Wilson if the task says "score interval".
## Conventions and defaults when the task is silent
- Rows = FROM (beginning of period, BOP), columns = TO (EOP). Check header "BOP\EOP"; if column sums
  are ~1 and row sums are not, the file is transposed.
- Withdrawn / NR: censored, dropped from numerator AND denominator ("NR-adjusted", S&P default studies).
  Alternatives only on cue: "NR as a state" (keep column), "redistribute pro rata".
- Mapping to 7+1: strip +/- notches (AA+,AA,AA- -> AA); S&P CCC+/CCC/CCC-/CC/C -> CCC; Moody's
  Caa1..Caa3/Ca/C -> CCC; SD, D (S&P) and D/LD -> Default. Apply the SAME map to every auxiliary file.
- Percent vs probability: row sums ~100 -> divide by 100. After dropping NR, renormalise rows to 1.
- Empty origin row in a cohort: identity row (stays) unless told otherwise; never NaN.
- Default absorbing: P row D = e_D, Q row D = 0.
- IG = AAA..BBB, SG = BB..CCC. Aggregate default rate = sum defaults / sum issuers (issuer-weighted).
- Q from a 1-year P is a per-year rate; P(0.5) = expm(0.5 Q).
- Chi-squared: task-given sparsity filter (e.g. column total >= 5) is applied to column TOTALS across
  cohorts; keep the Default column if it passes. Drop cohort rows with zero issuers.
## Completing missing inputs
- Hidden defaults: true defaults (default-rate table) minus visible D/SD in the matrix, floored at 0,
  added to the D cell (cohort size grows with them).
- If only percentages and cohort sizes are given, counts = round(pct/100 * N_i).
- Multi-year matrix from annual only via P^h (assumes Markov); from generator via expm(hQ).
- Missing destination columns in a reference matrix: add as 0, reorder to the target scale.
## Invariants to self-check
- Every row of P sums to 1 (atol 1e-9 on unrounded arrays), entries in [0,1], P[D] = e_D.
- PD_i(h) non-decreasing in h; PD(1) = column D of P; PD(h) -> 1 as h -> inf if D reachable.
- Q: off-diagonals >= 0, diagonals <= 0, rows sum 0, Q[D] = 0; expm(Q) ~ P (close, not exact after fix).
- p in [0,1]; reject <=> p < alpha; df integer >= 0.
- SG mean PD >= IG mean PD (grade-monotonicity of PD is typical but NOT guaranteed row by row; do not
  force it).
## Variants to decide
- V1 nr_treatment: withdrawn / NR ratings. Options: (a) removed from numerator and denominator; (b) NR as a state; (c) redistribute pro rata. Default when silent: (a) - S&P default studies.
- V2 estimator: transition estimator. Options: (a) cohort; (b) duration / generator. Default when silent: (a) - S&P / Moody's; (b) Lando-Skodeberg 2002.
- V3 generator_fix: Q from matrix log. Options: (a) set negative off-diagonals to 0, adjust diagonal; (b) weighted adjustment; (c) duration MLE. Default when silent: (a) - Israel-Rosenthal-Wei 2001.
- V4 aggregation: default rate weighting. Options: (a) issuer-weighted; (b) volume-weighted. Default when silent: (a) - S&P default studies.
- V5 cumulative_pd: multi-year PD. Options: (a) Markov P^n; (b) empirical cohort cumulative. Default when silent: (a) - Jarrow-Lando-Turnbull 1997.
- V6 rating_map: notch aggregation. Options: (a) strip notches, CCC..C pooled; (b) keep notches. Default when silent: (a) - S&P.
- V7 hidden_defaults: defaults after withdrawal. Options: (a) add max(true - visible defaults, 0) to the Default cell, row total grows; (b) ignore, use matrix defaults only. Default when silent: (a) - S&P default studies, whenever a default-rate table is supplied.
- V8 sparse_filter: chi-squared sparsity rule. Options: (a) drop destination columns whose total over cohorts is below the stated threshold; (b) require expected counts >= 5; (c) no filter. Default when silent: (b) - Cochran 1954; (a) when a threshold is stated.
- V9 default_states: which labels are default. Options: (a) D, SD and R (regulatory supervision) -> Default; (b) D only. Default when silent: (a) - S&P definitions / Basel default definition.
- V10 reference_matrix: published percent matrix with NR. Options: (a) drop NR, divide by 100, renormalise rows; (b) keep NR as a state. Default when silent: (a) - S&P NR-adjusted studies.

## Pitfalls
- Degenerate chi-squared table (fewer than 2 surviving columns or rows): stat 0, df 0, p 1, not
  rejected. Never return p=0 with reject False, never let scipy raise on zero expected.
- scipy.stats.chi2_contingency applies Yates correction when df == 1 (correction=True default): pass
  correction=False for the plain Pearson formula; it also raises if any expected cell is 0 - compute
  manually or drop zero rows/cols first.
- logm returns complex dtype with ~0 imaginary parts: take .real. Clip BEFORE setting the diagonal.
  det(P) <= 0 or diagonals < 0.5 -> log may not exist/be unique; still apply DA and report.
- Averaging annual P matrices instead of pooling counts (different weights).
- Semicolon CSVs, latin-1 encoding, stray whitespace in labels; strip before mapping.
- Checking row sums on 6-dp rounded CSV output with tight atol: rounding error ~n*5e-7.
- Argmax label ties: first occurrence in row-major order ("FROM->TO").
- Using duration estimator when "cohort" is stated, or vice versa.
- CEREP-style matrices are ';'-separated: a one-column frame named 'BOP\EOP;AAA;...' means the separator was missed - read_delimited_matrix, then collapse_rating_counts (NR/withdrawn censored), chi2_homogeneity on columns with total >= the stated minimum.
- read_delimited_matrix output is already split (BOP index, EOP columns); the pooled CEREP study is one call: cerep_migration_study.
- Yearly files: cerep_migration_report(files) reads years from the names; never pass bare lists of tables.
## Sources
Jarrow-Lando-Turnbull 1997; Lando-Skodeberg 2002 (J Banking Finance); Israel-Rosenthal-Wei 2001
(Math Finance); Kreinin-Sidelnikova 2001; Anderson-Goodman 1957; S&P Annual Global Corporate Default
and Rating Transition Study; Moody's default studies; Schuermann 2008 "Credit Migration Matrices";
Lando 2004 "Credit Risk Modeling" ch.5.
