method: local-vol-monte-carlo-barrier
family: local-and-stochastic-vol
keywords: local vol monte carlo, local-vol barrier, discretely monitored barrier, daily close monitoring, down-and-out call monte carlo, antithetic variates, knockout discount, survival probability, barrier hit probability, inverse option premium, btc option surface, call surface repair
---
## Formulas
1. Log-Euler step: X_{i+1} = X_i + (r - q - 0.5 s_i^2) dt + s_i sqrt(dt) Z, s_i = sigma_loc(t_i, S_i) at the step start (bilinear in K, linear in T, flat outside the grid).
2. Discrete barrier: check S <= H only at the last sub-step of each day (daily close); knocked paths pay 0.
3. Antithetic: pair Z and -Z; estimator per pair Y = (f(Z) + f(-Z))/2; SE = std(Y, ddof=1)/sqrt(N_pairs).
4. Hit probability = mean(knocked); survival = 1 - hit; knockout discount pct = 100 (vanilla - barrier)/vanilla.
5. Inverse premiums: USD = coin premium x forward (or underlying) price of that expiry; forward per expiry from cleaned call-put pairs.
6. Surface repair per expiry: enforce C >= intrinsic, non-increasing and convex in K (e.g. projection / PCHIP if pinned).
## Variants to decide
- V1 monitoring: barrier checks. Options: (a) daily close only; (b) every sub-step; (c) continuous with Brownian-bridge correction. Default when silent: (a) - contract terms (exchange daily fixing).
- V2 lv_interp: local vol between nodes. Options: (a) linear in T, linear in K, flat extrapolation; (b) nearest expiry. Default when silent: (a) - Gatheral 2006.
- V3 premium_conversion: coin to USD. Options: (a) multiply by the expiry forward; (b) by the row's underlying price. Default when silent: (a) - Deribit inverse contract specification.
## Invariants
- 0 <= barrier <= vanilla; hit + survival = 1; LV-MC vanilla near surface vanilla.
## Pitfalls
- Expiry labels: keep the venue label (e.g. DDMMMYY) in outputs when the data uses it; do not reformat to ISO.
- Datetime minus integer arrays fails in pandas: use pd.to_timedelta(ms, unit='ms').
## Sources
- Dupire (1994); Glasserman (2003) ch. 4, 6; Broadie-Glasserman-Kou (1997); Deribit contract specs.
