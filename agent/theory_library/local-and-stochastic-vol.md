family: local-and-stochastic-vol
category: derivatives-pricing
keywords: local volatility, dupire, svi, total implied variance, implied volatility surface, heston, characteristic function, stochastic volatility, sabr, smile, skew, calendar arbitrage, butterfly arbitrage, forward variance, call surface, deribit, barrier, down-and-out, local vol monte carlo, feller, calibration, gauss-legendre, two-factor heston, double heston, chiarella-ziveyi, gil-pelaez, numpy.gradient, pchip, local vol surface
---
## Worked frame
DIKETAHUI: quotes one row each [M] (expiry, strike, price or IV in percent -> decimal), S, r, q; grids K [n_K], T [n_T]. Check: sorted strikes and expiries, one forward per expiry.
DITANYAKAN: surface [n_T x n_K] (IV, total variance, local vol; decimal); parameters per slice [n_T]; model prices [M]; MC price and SE.
RUMUS: card Core 3-4 (SVI), 1-2 (Dupire), 5-7 (Heston), 8 (SABR), 10 (MC); toolkit implied_vol, bs_price.
JAWAB: 1) T [n_T], F [n_T], k = ln(K/F) [n_T x n_K]. 2) total variance w [n_T x n_K]. 3) fit per slice [n_T]. 4) derivatives, local variance [n_T x n_K]. 5) MC paths [n_paths x n_steps+1] -> scalar. 6) variance -> annualised vol. 7) check: w non-decreasing in T, density >= 0, parity; then write.

## Core formulas
1. Dupire (call prices, cont. rate r, div yield q): sigma_loc^2(K,T) = [C_T + (r-q) K C_K + q C] / (0.5 K^2 C_KK). r=q=0 or forward-measure prices: 2 C_T/(K^2 C_KK). C_T at FIXED K.
2. Gatheral total-variance form, k = ln(K/F_T), F_T = S e^{(r-q)T}, w = iv^2 T: sigma_loc^2 = w_T / [1 - (k/w) w_k + 0.25(-0.25 - 1/w + k^2/w^2) w_k^2 + 0.5 w_kk], w_T at FIXED k. Drift terms vanish. Flat iv -> sigma_loc = iv exactly.
3. Raw SVI per slice: w(k) = a + b[rho(k-m) + sqrt((k-m)^2 + s^2)]; w_k = b[rho + (k-m)/R], w_kk = b s^2/R^3, R = sqrt((k-m)^2+s^2). Constraints b>=0, |rho|<1, s>0, a + b s sqrt(1-rho^2) >= 0; Lee wing b(1+|rho|) <= 2. Natural SVI: w = D + (om/2)[1 + z rho(k-mu) + sqrt((z(k-mu)+rho)^2 + 1-rho^2)].
4. No-arbitrage: butterfly g(k) = (1 - k w_k/(2w))^2 - (w_k^2/4)(1/w + 1/4) + w_kk/2 >= 0 (density >= 0); calendar: w(k,T) non-decreasing in T at every k. Forward variance (w(T2)-w(T1))/(T2-T1) > 0.
5. Heston: dS=(r-q)S dt+sqrt(v)S dW1, dv=kappa(theta-v)dt+xi sqrt(v) dW2, corr rho. Feller 2 kappa theta >= xi^2 (v stays >0). E[v_t] = theta + (v0-theta)e^{-kappa t}.
6. Heston CF of ln S_T ("little trap", Albrecher 2007): b = kappa - rho xi i u, d = sqrt(b^2 + xi^2(i u + u^2)), g = (b-d)/(b+d), e = exp(-dT); D = (b-d)/xi^2 (1-e)/(1-g e); C = (r-q) i u T + kappa theta/xi^2 [(b-d)T - 2 ln((1-g e)/(1-g))]; phi(u) = exp(i u ln S0 + C + D v0). Independent multi-factor variance: sum C_j + D_j v_j0.
7. P2 = 1/2 + (1/pi) int_0^inf Re[e^{-iu lnK} phi(u)/(iu)] du; P1 same with phi(u-i)/phi(-i). Call = S e^{-qT} P1 - K e^{-rT} P2; Put = K e^{-rT}(1-P2) - S e^{-qT}(1-P1).
8. SABR (Hagan 2002) Black vol: sigma_B = alpha/[(FK)^{(1-beta)/2}(1 + (1-beta)^2 L^2/24 + (1-beta)^4 L^4/1920)] * z/x(z) * [1 + ((1-beta)^2 alpha^2/(24 (FK)^{1-beta}) + rho beta nu alpha/(4 (FK)^{(1-beta)/2}) + (2-3rho^2) nu^2/24) T], L = ln(F/K), z = (nu/alpha)(FK)^{(1-beta)/2} L, x(z) = ln[(sqrt(1-2rho z+z^2)+z-rho)/(1-rho)]; z/x -> 1 at K=F.
9. Smile quotes (FX): RR25 = iv25C - iv25P; BF25 = (iv25C+iv25P)/2 - ivATM; iv25C = ATM + BF + RR/2, iv25P = ATM + BF - RR/2.
10. LV MC log-Euler: X += (r-q-0.5 s^2)dt + s sqrt(dt) Z, s = sigma_loc(t_i, S_i) at the START of the step.

## Conventions and defaults when the task is silent
- T = ACT/365 from calendar days or exact ms timestamps (/(1000*86400*365)); never from the expiry label.
- Moneyness for SVI/Gatheral: forward log-moneyness ln(K/F), not ln(K/S). Fit SVI on total variance w, report RMSE in vol units sqrt(w/T).
- IV quoted like 55.3 -> percent, /100. Inverse (Deribit, BTC-quoted) premium: USD = BTC premium x F (or underlying); forward from parity F = K/(1 - (c-p)) with c,p in BTC, else F = K + e^{rT}(C-P) at the strike minimising |C-P|; one F per expiry.
- Surface from OTM quotes: K < F use put via parity C = P + S e^{-qT} - K e^{-rT}; K >= F use call.
- IV solver: Newton with vega, start sqrt(2 pi/T) C/S clipped, brentq fallback on [1e-4, 5]; reject prices outside (intrinsic, S e^{-qT}).
- Finite differences: central inside, one-sided at edges (np.gradient, edge_order as stated); C_KK on a price grid = 3-point (C+ - 2C + C-)/h^2, undefined at end strikes; a named stencil/interpolator (PCHIP, np.gradient twice) wins.
- LV interpolation between expiries: linear in T, flat outside grid; linear in K, flat beyond wings. Piecewise-constant "next/previous expiry" biases prices by several %.
- Heston calibration: least squares on prices or IVs, kappa, theta, xi, v0 > 0, |rho| < 1; Feller penalty only if asked.
- Barrier "daily close" monitoring: test the barrier only at the last sub-step of each day. Down-and-out knocks out at S <= H.

## Completing missing inputs
- Flat-vol diagnostic: build BS prices at const sigma on a near-ATM grid (e.g. K/S 0.9-1.1), apply Dupire; error ~1e-6 with small relative bumps. Avoid deep wings where C_KK ~ 0.
- Heston quadrature: integrand ~ 1/u at 0 (start at 1e-8); short T decays slowly, so [0,50] may truncate; Gauss-Legendre with many nodes on [0,100+] is safer.
- Local variance instability: floor sigma_loc^2 (e.g. 1e-10 or given floor), cap vol, NaN when denominator < tiny.
- Continuous vs discrete barrier: BGK shift H -> H exp(-0.5826 sigma sqrt(dt)) for down barriers (exp(+...) for up).

## Invariants to self-check
- C non-increasing and convex in K, C >= max(S e^{-qT} - K e^{-rT}, 0), C <= S e^{-qT}; C non-decreasing in T at fixed forward moneyness.
- phi(0) = 1, phi(-i) = S0 e^{(r-q)T} (martingale); put-call parity to ~1e-12 when puts use the same P1, P2.
- sigma_loc > 0 everywhere after floor; ATM local vol near ATM implied (short-T local skew ~ 2x implied skew).
- LV-MC vanilla within a few % of the surface price; barrier <= vanilla, >= 0; hit + survival = 1.
- SVI params inside bounds; w >= 0; forward variance positive.

## Variants to decide
- V1 moneyness: SVI coordinate. Options: (a) forward log-moneyness ln(K/F); (b) ln(K/S). Default when silent: (a) - Gatheral 2004.
- V2 svi_target: SVI fit target. Options: (a) total variance w; (b) implied vol. Default when silent: (a) - Gatheral 2006.
- V3 heston_cf: Heston characteristic function form. Options: (a) Albrecher "little trap" form; (b) original Heston 1993 form (branch-cut risk). Default when silent: (a) - Albrecher et al. 2007.
- V4 heston_mc: variance discretisation. Options: (a) full-truncation Euler; (b) reflection; (c) Andersen QE. Default when silent: (a) - Lord-Koekkoek-van Dijk 2010.
- V5 dupire_form: local vol formula. Options: (a) in total implied variance w(y,T); (b) on call prices (C_T, C_KK). Default when silent: (a) - Gatheral 2006.
- V6 time_basis: T for surfaces. Options: (a) ACT/365 calendar; (b) trading days/252. Default when silent: (a) - Gatheral 2006.
- V7 strike_derivatives: C_K, C_KK (or w_k, w_kk). Options: (a) analytic from the fitted SVI slice; (b) np.gradient on the grid (gradient twice for C_KK); (c) 3-point stencil (fd_second_derivative). Default when silent: (a) - Gatheral 2006; a raw price grid means (c), a spec naming np.gradient (b).
- V8 cf_quadrature: Heston P1/P2 integral. Options: (a) Gauss-Legendre with many nodes on a wide [0, U]; (b) adaptive quad on [tiny, U]. Default when silent: (a) - Kahl-Jaeckel 2005.
- V9 expiry_label: expiry written to outputs. Options: (a) the source label exactly as in the data (venue code such as DDMMMYY); (b) ISO date. Default when silent: (a) - Deribit instrument naming (identifiers are not reformatted).

## Pitfalls
- Taking w_T at fixed K in Gatheral form, or C_T at fixed k in call form.
- Original Heston CF has a branch cut in the complex log for long T; use little-trap form.
- Forgetting e^{-qT} in delta, parity or forward; using S not F in k.
- Antithetic SE from 2N correlated samples; use N pair means, SE = std(pair, ddof=1)/sqrt(N).
- LV lookup at S0 or at the previous day instead of current S_i.
- Monitoring the barrier on every sub-step when spec says daily close (or vice versa).
- Self-check tolerance too tight (2%) flips all_checks false; coarse Dupire grids don't reprice exactly.
- Flat-vol test in w form: w = sigma^2 T so w_T = sigma^2, never 0 (w_T = 0 gives local vol 0). Emit a report's files, never rebuild them (bundle_files).
- Scalar implied_vol fed a Series ("truth value ambiguous"): use implied_vol_vec; option chains with many rows per (expiry, type, strike) must be aggregated first.
- Coin-quoted premiums (inverse options): forward from the pair is F = K/(1 - (c - p)), USD premium = coin premium x F (forward_from_coin_parity); epoch-ms expiries subtracted as int64 ms, never after datetime parsing.

- newton_iv_vec returns a 3-tuple (never a column); liquid expiries: select_liquid_expiries; give report helpers the quotes AS LOADED (no parity pre-conversion, no relabelled option_type).
- One call per study: svi_dupire_report (S0 = close ON the valuation date; converged is a per-point mask), localvol_barrier_report (RAW coin snapshot), heston2_surface_deliverables (every file).
## Sources
Dupire 1994; Gatheral 2006; Gatheral-Jacquier 2014; Heston 1993; Albrecher et al. 2007 (little Heston trap); Christoffersen-Heston-Jacobs 2009; Hagan et al. 2002 (Managing Smile Risk); Broadie-Glasserman-Kou 1997; Glasserman 2003; Lee 2004.
