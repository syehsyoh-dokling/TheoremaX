method: lead-lag-cross-correlation
family: factor-models-and-sorts
keywords: lead-lag, lead lag, cross-correlation, cross correlation, lagged correlation, lead-lag asymmetry, peak lag, fisher z, leader, follower, cross-asset, granger, ccf, residual lead-lag
---
## Formulas
1. rho(k) = corr(A_t, B_{t+k}), k in [-K, K]; k > 0 means A leads B. Pair by row position, drop NaN pairwise.
2. Asymmetry = max_{k>0} rho(k) - max_{k<0} rho(k) (or |rho| as stated); peak lag = argmax.
3. Fisher z = atanh(rho), rho clipped inside +-1; se = 1/sqrt(n-3) for one corr, sqrt(2/(n-3)) for a difference.
4. Residuals e = r - (a + b r_m), coefficient vector order [const, slope]; in-sample a, b reused out of sample.
5. Annualised log return = sum log r/(calendar days/365.25) or mean x P.
## Variants to decide
- V1 lag_sign: lag orientation. Options: (a) corr(A_t, B_{t+k}), positive k = A leads; (b) corr(A_{t-k}, B_t), same meaning; (c) raw numpy correlate index. Default when silent: (a) - Box-Jenkins ccf convention.
- V2 n_obs_gate: n in the Fisher se. Options: (a) overlapping pairs n-|k|; (b) full return count of the period. Default when silent: (a) - Fisher 1921.
- V3 lag_selection_sample: where the lag is chosen. Options: (a) in-sample only, frozen for OOS; (b) re-estimated per period. Default when silent: (a) - no look-ahead practice.
- V4 tie_order: ranking ties. Options: (a) stable sort keeping pair-generation order; (b) alphabetical. Default when silent: (a) - pandas stable sort.
## Invariants
- rho in [-1, 1]; rho_AB(k) = rho_BA(-k); k = 0 is not tradeable.
## Pitfalls
- shift sign reversed; correlating price levels; alpha and beta swapped.
## Sources
- Box-Jenkins-Reinsel; Lo-MacKinlay 1990; Fisher 1921; Granger 1969.
