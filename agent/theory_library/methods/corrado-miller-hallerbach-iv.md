method: corrado-miller-hallerbach-iv
family: implied-vol-and-parity
keywords: corrado-miller, corrado miller, hallerbach, corrado-miller-hallerbach, cmh, brenner-subrahmanyam, brenner subrahmanyam, sigma_cmh, sigma_bs_approx, quadratic implied vol approximation, closed-form implied volatility, atm approximation
---
## Formulas
Notation: S~ = S e^{-qT}, K~ = K e^{-rT}, C call price, x = sigma sqrt T.
1. Brenner-Subrahmanyam (ATM forward): sigma = sqrt(2 pi / T) C / S~ (the zero-order term of CM).
2. Corrado-Miller: x = sqrt(2 pi)/(S~ + K~) * [C - (S~ - K~)/2 + sqrt((C - (S~ - K~)/2)^2 - (S~ - K~)^2 / pi)].
3. Hallerbach (CMH): x = sqrt(2 pi)/(2 (S~ + K~)) * [2C + K~ - S~ + sqrt((2C + K~ - S~)^2 - 1.85 (S~ + K~)(K~ - S~)^2 / (pi sqrt(K~ S~)))].
4. Put input: convert with parity C = P + S~ - K~ first.
## Variants to decide
- V1 cm_discriminant: negative discriminant. Options: (a) floor at 0; (b) NaN. Default when silent: (a) - Corrado-Miller 1996 (set the root to zero).
- V2 cm_form: which formula. Options: (a) Hallerbach 2004 when CMH or Hallerbach is named; (b) original Corrado-Miller 1996. Default when silent: (b) - Corrado-Miller 1996.
- V3 bs_atm_scope: Brenner-Subrahmanyam rows. Options: (a) ATM rows only, NaN elsewhere; (b) every row. Default when silent: (a) - Brenner-Subrahmanyam 1988.
## Invariants
- At K~ = S~ CM reduces to Brenner-Subrahmanyam; CMH error smaller than CM away from ATM.
- sigma > 0 for C above intrinsic; BS reprices to within the approximation error.
## Pitfalls
- Raw S and K instead of discounted S~, K~.
- Dividing by sqrt T twice, or forgetting it (x is total vol).
- Brenner-Subrahmanyam at K = S0 with r != q carries a real bias: that is the method, not a bug.
## Sources
- Brenner and Subrahmanyam (1988) FAJ; Corrado and Miller (1996) J. Banking and Finance; Hallerbach (2004) ERIM report.
