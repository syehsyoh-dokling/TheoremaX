family: bonds-duration-immunization
category: fixed-income
keywords: bond price, clean price, dirty price, accrued interest, yield to maturity, macaulay duration, modified duration, effective duration, convexity, dv01, pv01, key rate duration, krd, immunization, liability, cash-flow matching, z-spread, frn, callable, pca, level slope curvature, explained variance, hedge, stress, immunization portfolio, krd matching, hedging error, stale hedge, re-hedge, slsqp, dollar allocation, par-yield bootstrap, call cap, between reset dates
---
## Worked frame
DIKETAHUI: bonds one row each [B] (coupon and yield in percent -> decimal, maturity in years, freq, face); curve [P]; curve history [T x P]. Check: sorted tenors, unique ids.
DITANYAKAN: per bond [B]: dirty/clean price, durations, convexity, DV01 (currency per bp); KRD [B x P]; hedge weights [B]; loadings [k x P].
RUMUS: card Core 1-7, 9 (KRD), 13 (immunization), 14 (PCA); toolkit bond_price, bond_ytm, duration_convexity.
JAWAB: 1) cash flows per bond [n_b]. 2) dirty, accrued, clean [B]. 3) duration, convexity [B]. 4) KRD [B x P]. 5) weights [B], or changes [T-1 x P] -> loadings [P]. 6) price per face -> notional currency. 7) check: dirty = clean + accrued, D_mod < D_mac, KRD row sum = effective duration; then write.

## Core formulas
y = yield, f = coupon freq, CF_k at time t_k (years), P = DIRTY price unless stated.
1. Price: P = sum CF_k / (1 + y/f)^(f t_k); curve price P = sum CF_k DF(t_k). Street convention with fractional first period: exponent = w + k - 1, w = days to next coupon / days in period.
2. Clean = dirty - AI; AI = (c/f) F * (accrual days / period days) (ICMA) or c F * yf_30/360. AI = 0 on a coupon date.
3. YTM: solve price(y) = P with brentq; compounding = the bond's coupon frequency (UST semi, annual-pay bonds annual).
4. Macaulay D = sum t_k PV_k / P. Modified D = D_mac / (1 + y/f); continuous yield: D_mod = D_mac. Curve-based variants use PV_k = CF_k DF(t_k).
5. Convexity (periodic y): C = sum CF_k t_k (t_k + 1/f) (1+y/f)^(-f t_k) / (P (1+y/f)^2); continuous: C = sum t_k^2 PV_k / P.
6. Effective: D_eff = (P_down - P_up)/(2 P h); C_eff = (P_up + P_down - 2P)/(P h^2); bump the curve actually used for pricing (if it is bootstrapped from par yields, bump PAR yields and re-bootstrap).
7. DV01 = -dP/dy * 1e-4 = D_mod * P * 1e-4 > 0 for a long bond, scaled to notional (price per 100 -> * face/100).
8. dP/P ~ -D dy + 0.5 C dy^2.
9. KRD_k = -(P_up,k - P_down,k)/(2 P h) with a bump shape at key rate k: tent/triangular between neighbouring keys (Ho 1992) so sum_k KRD_k = D_eff; a point bump (only pillar k) is used only when stated (then flows off-pillar get zero KRD).
10. Z-spread s: sum CF_k exp(-(z(t_k) + s) t_k) = P_dirty (continuous zeros). Priced off the same curve -> s ~ 0.
11. FRN between resets: dirty = (N + known coupon) DF(t_next) (flat index, zero spread); D_eff ~ t_next.
12. Callable (deterministic first call): P = min(P_noncall, PV of flows to call + call price * DF(t_call)); yield-to-worst; call binds when coupon > market yield.
13. Redington immunization: PV_A = PV_L, dollar duration D_A PV_A = D_L PV_L, convexity C_A >= C_L. Two bonds: w1 + w2 = 1, w1 D1 + w2 D2 = D_L (value weights); units = dollars / dirty price. KRD matching: sum_i x_i KRD_i,k = PV_L KRD_L,k (dollar KRDs additive; portfolio KRD = sum x_i KRD_i / sum x_i).
14. PCA of curve changes: covariance of daily CHANGES (not levels), eigh, sort descending; explained_k = lambda_k / sum(all lambda); factor exposure = sum_k KRD_k * loading_k.
## Conventions and defaults when the task is silent
- Durations and DV01 use DIRTY price (Fabozzi, Tuckman). Report clean price as quoted price.
- Yield compounding = coupon frequency; accrued on ACT/ACT ICMA for govts, 30/360 for US corporates.
- KRD keys: the curve's own tenors; bump size 1bp; central differences unless "bump up by +1bp" (then one-sided -(P_up - P)/(P h)).
- PCA signs are arbitrary: default flip PC1 so loadings sum > 0 (level: all rates up); PC2 so (long - short) loading > 0 (steepener); PC3 so belly loading < wings (or as specified, e.g. make the liability's exposures positive). Top-3 PCs of Treasury changes typically explain > 95% (level ~80-90%).
- Units: yields in percent -> divide by 100; changes in bp scale eigenvalues by 1e4 vs percent; covariance ddof=1 (np.cov default, sklearn explained_variance_).
- Immunization with more bonds than constraints: many solutions; default min-norm or the task's objective; enforce long-only bounds and sum-to-one of value weights when stated.
## Completing missing inputs
- No price given: price off the bootstrapped curve; YTM then from that price.
- Liability PV/duration: treat as zero-coupon flows, same curve.
- DV01 from D_mod when only yields known; convexity from finite differences if formula ambiguous.
- Convexity of a zero: t(t + 1/f)/(1+y/f)^2 (periodic), t^2 (continuous).
- Stress scenarios: parallel = constant shift; steepener = linear in maturity between two anchor points; butterfly/curvature = key points with linear interpolation (np.interp, flat beyond ends) or a Gaussian bump if defined.
## Invariants to self-check
- Price decreases in yield; par bond (coupon = yield, on a coupon date) prices at 100; dirty = clean + AI exactly.
- D_mac of a zero = maturity; D_mac <= maturity for coupon bonds; D_mod < D_mac for y > 0.
- DV01 ~ D_mod * P * 1e-4 within 0.1%; sum of tent KRDs = D_eff.
- Callable price <= non-callable; callable duration <= non-callable when call binds.
- FRN effective duration ~ time to next reset (short), not maturity.
- Immunized portfolio: PV and dollar duration match to tolerance, weights within bounds, sum to 1; parallel-shift hedging error small.
- PCA: eigenvalues positive, descending; cumulative = sum of ratios; unit-norm loadings.
## Variants to decide
- V1 duration_price: price used for duration/DV01. Options: (a) dirty (full) price; (b) clean price. Default when silent: (a) - Fabozzi / Tuckman.
- V2 yield_compounding: YTM compounding. Options: (a) coupon frequency (bond-equivalent); (b) annual; (c) continuous. Default when silent: (a) - Fabozzi / Tuckman.
- V3 mod_duration: modified duration form. Options: (a) D_mac/(1+y/m); (b) effective -(P_up-P_dn)/(2 P dy) from bumped yields; (c) continuous D_mac (y continuous). Default when silent: (a) - Fabozzi; (b) for callables/embedded options.
- V4 convexity_def: convexity scaling. Options: (a) C=(1/P) d2P/dy2 with dP/P=-D dy+0.5 C dy^2; (b) C/100 or half-convexity (dealer quoting). Default when silent: (a) - Fabozzi / Hull.
- V5 bump_scheme: KRD / effective duration bump. Options: (a) central +-1bp; (b) one-sided +1bp. Default when silent: (a) - Tuckman.
- V6 pca_input: PCA input for curve factors. Options: (a) covariance of yield changes, ddof=1; (b) correlation matrix; (c) yield levels. Default when silent: (a) - Litterman-Scheinkman 1991.
- V7 hedge_formulation: immunization solve. Options: (a) hard equality constraints (PV, target KRDs) in SLSQP, bounds on weights; (b) penalty objective with each residual scaled by its target. Default when silent: (a) - Redington 1952 / Fabozzi; (b) when the task asks to balance several goals.
- V8 frn_between_resets: FRN observed mid-period. Options: (a) dirty = (100 + known coupon) DF(t_next), clean = dirty - AI, D_eff ~ t_next; (b) flat 100. Default when silent: (a) - Fabozzi / Tuckman-Serrat.
- V9 callable_rule: deterministic call. Options: (a) min(non-call price, PV of flows to call + call price DF(t_call)); (b) yield-to-worst only; (c) lattice with option value. Default when silent: (a) - Fabozzi.
- V10 pca_sign: factor sign alignment. Options: (a) PC1 loadings sum > 0, PC2 long-minus-short > 0; (b) align so the liability's exposures are positive. Default when silent: (a) - Litterman-Scheinkman 1991; (b) when stated.

## Pitfalls
- Clean price in duration; accrued at coupon date set to full coupon.
- Semi-annual y treated as annual (D_mod = D_mac/(1+y) instead of /(1+y/2)).
- Percent-KRD vs dollar-KRD mixing in constraints (portfolio KRDs are value-weighted, dollar KRDs add).
- Bumping zero rates when the task says bump the par curve used to bootstrap (or vice versa).
- PCA on levels or on correlation when covariance of changes is asked; unsorted np.linalg.eig; dividing by top-3 sum.
- Stale hedge: re-use old weights on new exposures; re-hedge solved fresh.
- Optimiser scaling: scale constraints by PV_L; check the residuals, not success flags.
- Undated coupon grids ((i+1)/freq from the valuation date) misprice dated bonds: roll coupon dates back from maturity (coupon_schedule, dated_bond_cashflows), discount ACT/365, accrue with the instrument's day count. Immunisation in dollars: sum x_i KRD_i = PV_L KRD_L, never price*KRD*x.
- Glue crashes (curve dict keys, coupon_schedule tuple, spot_bump_krd tuple) cost whole runs: par_curve_immunization_bundle / bond_hedge_immunization_bundle return every deliverable.
- Immunize with SLSQP EQUALITY constraints (PV and each target KRD), not a penalty objective; a PV-squared penalty swamps the KRD terms.
- 'Bootstrap in order': each node solved with g = zT linear to the last node; spline g only afterwards.
## Sources
Fabozzi, Handbook of Fixed Income Securities; Tuckman-Serrat, Fixed Income Securities ch. 3-6; Ho (1992) Key Rate Durations; Redington (1952); Litterman-Scheinkman (1991) Common Factors Affecting Bond Returns; Hull OFOD ch. 4.
