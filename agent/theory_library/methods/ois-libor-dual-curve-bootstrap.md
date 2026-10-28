method: ois-libor-dual-curve-bootstrap
family: curve-bootstrapping
keywords: dual curve, dual-curve bootstrap, ois discount curve, libor projection curve, 3m libor, projection curve, ois discounting, multi-curve, irs par rate, model par rate, repriced quotes, residual bps, forward libor, ametrano-bianchetti
---
## Formulas
1. OIS / deposit single-payment pillar: D(T) = 1/(1 + r tau(t0,T)), tau money-market basis.
2. Multi-period OIS: S = (D(T0) - D(Tn))/sum_i tau_i D(Ti); solve D(Tn) pillar by pillar.
3. Projection forward: F_j = (P(s_{j-1})/P(s_j) - 1)/tau_j, P = LIBOR DFs, float accrual tau_j.
4. IRS pillar: brentq on ln P(Tn) so S sum_i tau^fix_i D(t_i) = sum_j tau^flt_j F_j D(s_j), D = OIS DFs; P between the last pillar and Tn from the stated interpolation of the trial curve.
5. Model par = float PV / fixed annuity (OIS discounting); residual_bps = (model - input) 1e4.
6. Zero z = -ln D / t, t on the CURVE time basis (not the accrual basis).
7. Projection PV01: P' = exp(-(z_L + h) t), OIS fixed; PV01 = |NPV' - NPV|.
## Variants to decide
- V1 ois_structure: OIS pillar cash flows. Options: (a) single payment up to 1Y, annual fixed vs compounded overnight beyond; (b) single payment at every tenor. Default when silent: (a) - Ametrano-Bianchetti 2013.
- V2 curve_time_basis: time axis for interpolation and zeros. Options: (a) ACT/365F; (b) ACT/365.25; (c) the leg accrual basis. Default when silent: (a) - Henrard 2014.
- V3 gap_dfs: dates between pillars. Options: (a) interpolate the trial curve, root-find; (b) closed form with earlier DFs only. Default when silent: (a) - Hagan-West 2006.
- V4 pv01_bump: risk definition. Options: (a) bump projection zeros, OIS fixed; (b) bump par quotes, re-bootstrap. Default when silent: (b) - Henrard 2014.
## Invariants
- OIS built first; every quote reprices below 1e-6 bp; DFs decreasing; par swap NPV 0.
## Pitfalls
- Discounting IRS legs on the projection curve.
- pd.Timestamp on a Series: loop dates; hand-code ACT/365.25 if a helper rejects it.
- 3M forward to pillar + 3 calendar months, not + 0.25 years.
## Sources
- Ametrano-Bianchetti (2013); Henrard (2014) Multi-curve Framework; Hagan-West (2006).
