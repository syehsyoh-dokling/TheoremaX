method: lookback-closed-form
family: exotic-options
keywords: lookback option, floating strike lookback, fixed strike lookback, goldman sosin gatto, conze-viswanathan, running maximum, running minimum, lookback call, lookback put, extreme value option, partial lookback
---
## Formulas
b = r - q, v = sigma sqrt T, lambda = 2b/sigma^2; m = running min, M = running max (= S at inception).
1. Floating call (GSG 1979): S e^{-qT} N(a1) - m e^{-rT} N(a2) + S e^{-rT} (sigma^2/2b) [ (S/m)^{-lambda} N(-a1 + 2b sqrt T/sigma) - e^{bT} N(-a1) ], a1 = [ln(S/m) + (b + sigma^2/2) T]/v, a2 = a1 - v.
2. Floating put: M e^{-rT} N(-b2) - S e^{-qT} N(-b1) + S e^{-rT} (sigma^2/2b) [ e^{bT} N(b1) - (S/M)^{-lambda} N(b1 - 2b sqrt T/sigma) ], b1 on ln(S/M).
3. Fixed call (Conze-Viswanathan 1991), K > M: S e^{-qT} N(d1) - K e^{-rT} N(d2) + S e^{-rT} (sigma^2/2b) [ e^{bT} N(d1) - (S/K)^{-lambda} N(d1 - 2b sqrt T/sigma) ]; K <= M: e^{-rT}(M - K) + the same with K -> M.
4. b = 0: take the limit (evaluate at a tiny b) or use the b = 0 Haug form.
## Variants to decide
- V1 lookback_monitoring: extreme sampling. Options: (a) continuous closed form; (b) discrete with BGK-type shift of the extreme by exp(+-beta1 sigma sqrt dt); (c) MC on the grid. Default when silent: (a) - Goldman-Sosin-Gatto 1979.
- V2 lookback_seed: initial extreme. Options: (a) M = m = S at inception; (b) observed running extreme for seasoned trades. Default when silent: (a) - Haug 2007.
## Invariants
- Floating lookback >= vanilla ATM; fixed call(K = S) - floating put = S e^{-qT} - S e^{-rT}.
- Discrete MC lookback < continuous closed form.
## Pitfalls
- Grid MC maxima understate continuous ones; use a Brownian-bridge maximum.
- Division by b when r = q.
## Sources
- Goldman, Sosin, Gatto (1979); Conze and Viswanathan (1991); Haug (2007) ch. 4.15.
