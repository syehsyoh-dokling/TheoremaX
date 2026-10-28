method: mtm-xccy-basis-swap
family: fx-forwards-carry-xccy
keywords: mtm cross-currency swap, resettable xccy swap, cross-currency basis, implied basis, par basis, all-in spread, notional reset, mtm reset cashflow, usd collateral, broken-date fx forward, historical fixing, cip forward no basis, basis term structure, roll down
---
## Formulas
1. No-basis CIP (pair f/d, d per f, USD collateral): F_cip(T) = S D_f(T)/D_d(T) with OIS DFs.
2. Implied basis on the non-USD leg: F_mkt = S D_f(T) e^(-b T)/D_d(T), so b = -ln(F_mkt/F_cip)/T (bps = 1e4 b); simple form: F_mkt = S (1 + r_d t_d)/(1 + (r_f + b) t_f).
3. Notionals: N_0 = N_USD/S_inception; N_i = N_USD/F(reset_i) (F = USD per f).
4. Reset flow (receiver of the f leg): N_{i-1} - N_i in f at reset_i; final + N_last f, - N_USD USD.
5. Coupons: f leg N_i (L_f + s) tau_f; USD leg N_USD L_d tau_d; L = historical fixing if fixing date <= valuation date, else projection forward.
6. PV_USD = sum_f CF F(pay) D_d(pay) + sum_USD CF D_d(pay); flows strictly before valuation excluded.
7. All-in par basis: s* = s - PV/(sum_i N_i tau_i F_i D_d,i).
## Variants to decide
- V1 basis_form: implied basis. Options: (a) continuous spread on the non-USD discount curve; (b) simple spread on the non-USD deposit rate. Default when silent: (a) - Fujii-Shimada-Takahashi 2010.
- V2 broken_fx: dates between FX nodes. Options: (a) interpolate the stated quantity (points, log F or basis) between spot at t=0 and tenor nodes; (b) CIP with interpolated basis. Default when silent: (a) - Clark 2011.
- V3 fixing_rule: coupon rate source. Options: (a) fixing on or before valuation uses history; (b) strictly before only. Default when silent: (a) - ISDA 2006.
- V4 par_basis: reported spread. Options: (a) all-in spread; (b) increment over contract spread. Default when silent: (a) - market par CCS quoting.
## Invariants
- Basis reconciles to FX nodes exactly; legs sum to trade PV; stress PnL = stressed - base PV.
## Pitfalls
- Spread on the USD leg; reset sign flipped; percent and pips not decimalised.
## Sources
- Fujii-Shimada-Takahashi (2010); Clark (2011); Tuckman-Serrat ch. 18.
