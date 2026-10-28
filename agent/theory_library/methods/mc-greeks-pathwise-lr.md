method: mc-greeks-pathwise-lr
family: monte-carlo-greeks-and-hedging
keywords: pathwise greeks, ipa, infinitesimal perturbation analysis, likelihood ratio greeks, score function, finite difference greeks, bump and revalue, monte carlo delta, monte carlo vega, lr gamma, variance ratio, asian option greeks, common random numbers
---
## Formulas
GBM: S_T = S0 exp((r - q - sigma^2/2)T + sigma sqrt T Z), D = e^{-rT}; Asian A = mean of S_j at t_j = jT/n, W_j cumulative Brownian.
1. Pathwise European call (put: indicator S_T < K, sign flipped): delta D 1{S_T>K} S_T/S0; vega D 1{} S_T (sqrt T Z - sigma T); rho D 1{} K T; dV/dT = D [-r (S_T - K)+ + 1{} S_T ((r - q - sigma^2/2) + sigma Z/(2 sqrt T))], theta = -dV/dT; gamma not available (null).
2. Pathwise Asian call: delta D 1{A>K} A/S0; vega D 1{} mean_j S_j (W_j - sigma t_j).
3. LR weights x D payoff: delta Z/(S0 sigma sqrt T); gamma (Z^2 - 1 - Z sigma sqrt T)/(S0 sigma sqrt T)^2; vega (Z^2 - 1)/sigma - Z sqrt T; rho Z sqrt T/sigma - T; theta null unless derived.
4. FD with the SAME normals in every bumped leg; theta bump rebuilds the date grid with T - h.
5. Estimator SE = std(per-path estimator, ddof=1)/sqrt(N); variance comparison uses those per-path samples.
## Variants to decide
- V1 vega_scale: vega unit. Options: (a) per 1.00 sigma as the written formula gives; (b) per vol point (/100). Default when silent: (a) - Glasserman 2003 (literal derivative).
- V2 theta_scale: theta unit. Options: (a) as the written formula yields (per year when h is in years); (b) per day /365; (c) per trading day /252. Default when silent: (a) - Hull (literal formula over prose label).
- V3 unavailable_greeks: undefined estimators. Options: (a) JSON null for value and se; (b) NaN. Default when silent: (a) - JSON spec (NaN is not valid JSON).
## Invariants
- Pathwise delta variance < LR delta variance; all methods near BS; FD call-put delta difference = e^{-qT} with CRN.
## Pitfalls
- Broadcasting (n,1) date vectors against (N, n) paths: keep time on axis 1 as a (1, n) row.
- 500k x 12 arrays per bump: reuse one Z block, avoid Python loops.
## Sources
- Broadie and Glasserman (1996) Management Science; Glasserman (2003) ch. 7; Hull.
