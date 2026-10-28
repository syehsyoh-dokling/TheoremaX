method: hull-white-calibration-jamshidian
family: short-rate-and-rate-options
keywords: hull-white calibration, caplet calibration, hull-white caplet, zero-coupon bond put, jamshidian decomposition, jamshidian trick, r star, critical rate, european payer swaption, atm swap rate, analytical swaption, mean reversion speed, short-rate volatility, rmse bps
---
## Formulas
1. B(t,T) = (1 - e^{-a(T-t)})/a; P(t,T) = A(t,T) e^{-B r}, ln A = ln(P(0,T)/P(0,t)) + B f(0,t) - sigma^2 (1 - e^{-2at}) B^2/(4a).
2. f(0,t) from the market curve: log-linear DF -> piecewise-flat forward; r0 = f(0,0).
3. ZBP(T,S,K) = K P(0,T) N(-h + sp) - P(0,S) N(-h), sp = sigma sqrt((1 - e^{-2aT})/(2a)) B(T,S), h = ln(P(0,S)/(K P(0,T)))/sp + sp/2.
4. Caplet on [T1,T2], accrual d = T2 - T1, strike K: (1 + K d) ZBP(T1, T2, 1/(1 + K d)).
5. Calibration: min_{a,sigma} sum (model_i - market_i)^2 (L-BFGS-B, bounds); RMSE_bps = sqrt(mean diff^2) 1e4.
6. ATM rate K_s = (P(0,T0) - P(0,Tn))/sum_i d_i P(0,Ti).
7. Jamshidian: c_i = K_s d_i, c_n += 1; solve sum c_i P(T0,Ti,r*) = 1 (brentq); payer = sum c_i ZBP(T0, Ti, P(T0,Ti,r*)); receiver uses ZBC.
## Variants to decide
- V1 caplet_timing: caplet option expiry. Options: (a) fixing date T1, bond maturing at payment T2; (b) payment date as expiry. Default when silent: (a) - Brigo-Mercurio 2006.
- V2 curve_interp: DFs between pillars. Options: (a) log-linear DF; (b) linear zero; (c) cubic on zero. Default when silent: (a) - Hagan-West 2006.
- V3 calib_objective: fit target. Options: (a) price SSE; (b) Black-vol SSE; (c) relative price errors. Default when silent: (a) - Brigo-Mercurio 2006 ch. 3.
## Invariants
- P_HW(0,T,r0) = market P(0,T); caplets and swaptions > 0; payer - receiver = annuity (S - K).
- Parameters inside bounds, not stuck at a bound unless the data force it.
## Pitfalls
- Market arrays of different lengths (expiries vs prices) give empty or broadcast errors inside the optimiser; build aligned numpy arrays first and test the objective once.
- Using T2 as option time in sp; forgetting the (1 + K d) scaling.
## Sources
- Hull-White (1990); Jamshidian (1989); Brigo-Mercurio, Interest Rate Models ch. 3.3; Hull OFOD ch. 32.
