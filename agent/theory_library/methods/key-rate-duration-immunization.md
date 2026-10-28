method: key-rate-duration-immunization
family: bonds-duration-immunization
keywords: key rate duration, krd, krd matching, immunization portfolio, dollar allocation, long-only, slsqp, equality constraints, match present value, liability hedge, hedge weights, stale hedge, re-hedge, hedging error bps, stress scenarios, ho 1992, redington
---
## Formulas
1. KRD_k = -(P(+h at k) - P(-h at k))/(2 P h); one-sided -(P(+h) - P)/(P h) when the task says bump +1bp.
2. Bump shape: tent (1 at key k, linear to 0 at neighbouring keys) or point (only pillar k; flows off that pillar get zero KRD).
3. Dollar KRD of holding i: x_i KRD_i,k (x_i = dollars); portfolio KRD = sum_i x_i KRD_i,k / sum_i x_i.
4. Hard immunization: min ||x||^2 (or sum (x - x_ref)^2) s.t. sum x_i = PV_L, sum x_i KRD_i,k = PV_L KRD_L,k for each target k, x_i >= 0; units_i = x_i / price_i.
5. Soft hedge (task lists goals to balance): weights w (sum 1, bounds) minimise J = sum_k (KRD_p,k - KRD_L,k)^2 + sum_f (PC_p,f - PC_L,f)^2 + sum_s (err_s/scale)^2.
6. Stress hedging error (bps) = (dV_P/V_P - dV_L/V_L) 1e4.
## Variants to decide
- V1 krd_bump_shape: key-rate shock. Options: (a) tent between neighbouring keys; (b) point bump of one pillar spot rate. Default when silent: (a) - Ho 1992; (b) when told to bump each spot rate individually.
- V2 krd_difference: finite difference. Options: (a) central +-1bp; (b) one-sided +1bp. Default when silent: (a) - Tuckman-Serrat.
- V3 hedge_formulation: constraint handling. Options: (a) equality constraints in SLSQP; (b) scaled penalty least squares. Default when silent: (a) - Redington 1952 / Fabozzi.
- V4 krd_aggregation: portfolio KRD. Options: (a) dollar KRDs add; (b) value-weighted average. Default when silent: (a) - Tuckman-Serrat.
## Invariants
- Constraint residuals ~0 relative to PV_L; x >= 0; tent KRDs sum to D_eff; re-hedge objective <= stale.
## Pitfalls
- Unscaled penalty: dollar PV error swamps KRD terms, KRDs do not match.
- Trusting res.success instead of residuals; KRDs all zero = bump never reached pricing.
## Sources
- Ho (1992); Redington (1952); Tuckman-Serrat ch. 5-6.
