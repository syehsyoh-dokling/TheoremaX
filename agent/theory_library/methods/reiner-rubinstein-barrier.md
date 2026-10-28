method: reiner-rubinstein-barrier
family: exotic-options
keywords: reiner-rubinstein, barrier option closed form, down-and-out call, down-and-in call, up-and-out, knock-out, knock-in, analytical barrier price, barrier greeks, finite difference barrier delta, rebate, haug barrier
---
## Formulas
b = r - q, mu = (b - sigma^2/2)/sigma^2, lam = sqrt(mu^2 + 2r/sigma^2), v = sigma sqrt T; terms A, B, C, D, E, F as in the exotic-options family card (Haug 2007 ch. 4.17).
1. Down-and-out call, rebate R: K > H: A - C + F; K <= H: B - D + F. Down-and-in call = vanilla - down-and-out (R = 0).
2. Knocked already (S0 <= H for a down barrier): price = rebate (0 if none), all Greeks 0.
3. Central FD Greeks on the SAME closed-form function: delta (V(S+h) - V(S-h))/(2h), gamma (V(S+h) - 2V + V(S-h))/h^2, vega (V(sig+e) - V(sig-e))/(2e); h relative to S0 when a percent bump is given, e absolute.
4. Volatility input: the stated estimate (e.g. GARCH one-step-ahead conditional vol annualised by sqrt 252), not the long-run vol.
## Variants to decide
- V1 barrier_monitoring: monitoring. Options: (a) continuous closed form; (b) discrete with BGK shift of H by exp(+-beta1 sigma sqrt dt), beta1 = -zeta(1/2)/sqrt(2 pi); (c) MC. Default when silent: (a) - Reiner-Rubinstein 1991.
- V2 rebate_timing: rebate. Options: (a) paid at hit; (b) paid at expiry. Default when silent: (a) - Reiner-Rubinstein 1991.
- V3 bump_type: spot bump. Options: (a) relative h = pct x S0; (b) absolute h. Default when silent: (a) - Haug 2007 numerical Greeks.
## Invariants
- 0 <= DO <= vanilla; DI + DO = vanilla; DO -> vanilla as H -> 0; DO delta can exceed vanilla delta near H.
## Pitfalls
- Swapping the K > H and K <= H formula sets; using (H/S)^{2mu} where (H/S)^{2(mu+1)} belongs.
- Toolkit barrier_price argument order/labels: check kind and barrier strings before trusting a value.
## Sources
- Reiner and Rubinstein (1991) Risk; Haug (2007) Complete Guide ch. 4; Broadie-Glasserman-Kou (1997).
