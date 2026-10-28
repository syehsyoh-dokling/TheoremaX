method: dupire-local-vol
family: local-and-stochastic-vol
keywords: dupire, dupire formula, local volatility, local vol surface, sigma_loc, call price surface, total variance local vol, gatheral local vol, flat vol test, forward variance, dC_dT, d2C_dK2, local variance floor
---
## Formulas
1. Call-price form (yield q): sigma_loc^2 = [C_T + (r - q) K C_K + q C] / (0.5 K^2 C_KK), C_T at fixed K; r = q = 0 or forward prices: 2 C_T/(K^2 C_KK).
2. Total-variance form (y = ln(K/F_T), w = iv^2 T, w_T at fixed y): sigma_loc^2 = w_T / [1 - (y/w) w_y + 0.25(-0.25 - 1/w + y^2/w^2) w_y^2 + 0.5 w_yy].
3. Derivatives: analytic w_y, w_yy from SVI; w_T or C_T by forward/central differences across expiries; on a raw grid np.gradient (edge_order as stated; C_KK = gradient of gradient when told so).
4. Safeguards: denominator < tiny -> NaN; floor sigma_loc^2 at the given floor; cap vol at the given cap.
5. Forward variance per period: (w(T_{i+1}) - w(T_i))/(T_{i+1} - T_i) at fixed y (ATM: y = 0).
## Variants to decide
- V1 dupire_input: surface differentiated. Options: (a) total implied variance w(y,T); (b) call prices C(K,T). Default when silent: (a) - Gatheral 2006.
- V2 time_derivative: dT scheme. Options: (a) forward difference to the next expiry (last expiry backward); (b) central / np.gradient. Default when silent: (a) - Chin-Olafsson-Nel 7.2.2.
- V3 interior_only: strike edges. Options: (a) interior strikes only for C_KK; (b) one-sided edges. Default when silent: (b) - np.gradient convention.
## Invariants
- Flat iv surface returns sigma_loc = iv to ~1e-6 on a near-ATM grid.
- sigma_loc > 0 after floor; ATM local vol near ATM implied.
## Pitfalls
- C_T at fixed y or w_T at fixed K (coordinate mix).
- Deep wings: C_KK ~ 0 gives exploding local vol; keep the diagnostic grid near the money.
- Date columns: compute T with (expiry - valuation).dt.days on Series, never scalar Timestamp ops on a Series.
## Sources
- Dupire (1994) Risk; Gatheral, The Volatility Surface (2006); Chin, Olafsson, Nel Vol II 7.2.2.
