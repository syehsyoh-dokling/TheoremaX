family: exotic-options
category: derivatives-pricing
keywords: digital, cash-or-nothing, asset-or-nothing, gap option, barrier, knock-in, knock-out, rebate, lookback, floating strike, asian, geometric average, levy, curran, cliquet, forward start, compound option, geske, chooser, spread option, kirk, margrabe, exchange option, variance swap, down-and-out call, structured note, capital protected, participation, breakeven volatility, markup, zero-coupon bond, delta-gamma-vega var, vol of vol
---
## Worked frame
DIKETAHUI: S, K, barrier (currency), T (years), r, q, sigma (percent -> decimal), paths/steps/seed; scalars or one row per contract [M]. Check: barrier side vs spot, kind labels.
DITANYAKAN: price [M] (currency); MC price and SE; identity residuals; hit probability in [0,1].
RUMUS: card Core 1-8 by product; toolkit digital_price, barrier_price, asian_geometric_price, compound_option_price, margrabe_price, gbm_paths, bs_price.
JAWAB: 1) inputs [M]. 2) vanilla reference [M]. 3) closed form [M]. 4) paths [n_paths x n_steps+1] -> payoff [n_paths] -> price, SE. 5) companion legs for identities [M]. 6) discount payoffs to currency today. 7) check: in + out = vanilla, bounds, MC within its SE band; then write.

## Core formulas
b = r - q; M(a,b;rho) bivariate normal cdf.
1. Cash-or-nothing (pays 1): call e^{-rT} N(d2), put e^{-rT} N(-d2). Asset-or-nothing: call S e^{-qT} N(d1), put S e^{-qT} N(-d1). Vanilla C = AoN_c - K CoN_c. Gap call (pays S_T - K1 if S_T > K2): S e^{-qT} N(d1) - K1 e^{-rT} N(d2), d1,d2 on TRIGGER K2.
2. Barriers (Reiner-Rubinstein/Haug; strike X, barrier H, rebate R; phi=+1 call/-1 put; eta=+1 down/-1 up): mu = (b - sigma^2/2)/sigma^2, lam = sqrt(mu^2 + 2r/sigma^2), v = sigma sqrt T;
   x1 = ln(S/X)/v + (1+mu)v, x2 = ln(S/H)/v + (1+mu)v, y1 = ln(H^2/(SX))/v + (1+mu)v, y2 = ln(H/S)/v + (1+mu)v, z = ln(H/S)/v + lam v.
   A = phi S e^{(b-r)T} N(phi x1) - phi X e^{-rT} N(phi x1 - phi v); B = same with x2;
   C = phi S e^{(b-r)T}(H/S)^{2(mu+1)} N(eta y1) - phi X e^{-rT}(H/S)^{2mu} N(eta y1 - eta v); D = same with y2;
   E = R e^{-rT}[N(eta x2 - eta v) - (H/S)^{2mu} N(eta y2 - eta v)]; F = R[(H/S)^{mu+lam} N(eta z) + (H/S)^{mu-lam} N(eta z - 2 eta lam v)].
   X>H: DIc C+E, DOc A-C+F, UIc A+E, UOc F, DIp B-C+D+E, DOp A-B+C-D+F, UIp A-B+D+E, UOp B-D+F.
   X<H: DIc A-B+D+E, DOc B-D+F, UIc B-C+D+E, UOc A-B+C-D+F, DIp A+E, DOp F, UIp C+E, UOp A-C+F.
   Cash digital DO call (X>H): e^{-rT}[N(d2) - (H/S)^{2mu} N(d2_H)], d2_H = [ln(H^2/(SX)) + (b - sigma^2/2)T]/v; UO cash put (X<H): e^{-rT}[N(-d2) - (H/S)^{2mu} N(-d2_H)]. Exponent 2mu = 2b/sigma^2 - 1.
   BGK discrete monitoring every dt: H -> H e^{+0.5826 sigma sqrt dt} (up), H e^{-0.5826 sigma sqrt dt} (down).
3. Lookbacks (lambda = 2b/sigma^2, M = m = S at inception): float call = S e^{-qT} N(a1) - m e^{-rT} N(a2) + S e^{-rT}(sigma^2/2b)[(S/m)^{-lambda} N(-a1 + 2b sqrt T/sigma) - e^{bT} N(-a1)], a1 = [ln(S/m) + (b + sigma^2/2)T]/v, a2 = a1 - v (GSG).
   Float put = M e^{-rT} N(-b2) - S e^{-qT} N(-b1) + S e^{-rT}(sigma^2/2b)[e^{bT} N(b1) - (S/M)^{-lambda} N(b1 - 2b sqrt T/sigma)], b1 on ln(S/M).
   Fixed call, K > M: S e^{-qT} N(d1) - K e^{-rT} N(d2) + S e^{-rT}(sigma^2/2b)[e^{bT} N(d1) - (S/K)^{-lambda} N(d1 - 2b sqrt T/sigma)]; K <= M: e^{-rT}(M-K) + same with K -> M.
4. Geometric Asian, dates t_i = iT/n (i=1..n): ln(G/S) ~ N(Mu, H), Mu = (b - sigma^2/2) T(n+1)/(2n), H = sigma^2 T (n+1)(2n+1)/(6n^2); call = e^{-rT}[F_G N(d1) - K N(d2)], F_G = S e^{Mu + H/2}, d1 = [ln(F_G/K) + H/2]/sqrt H. Continuous: sigma_G = sigma/sqrt3, b_G = (b - sigma^2/6)/2.
   Levy: M1 = (S/n) sum e^{b t_i}, M2 = (S^2/n^2) sum_ij e^{b(t_i+t_j) + sigma^2 min(t_i,t_j)}, v^2 = ln(M2/M1^2); call = e^{-rT}[M1 N(d1) - K N(d2)], d1 = [ln(M1/K) + v^2/2]/v (Turnbull-Wakeman: continuous moments).
   Curran: mu_i = ln S + (b - sigma^2/2)t_i, s_i^2 = sigma^2 t_i, mu = mean(mu_i), s_x^2 = (sigma^2/n^2) sum_ij min(t_i,t_j), s_xi = (sigma^2/n) sum_j min(t_i,t_j); K^ = 2K - (1/n) sum_i exp(mu_i + s_xi (ln K - mu)/s_x^2 + (s_i^2 - s_xi^2/s_x^2)/2); call = e^{-rT}[(1/n) sum_i e^{mu_i + s_i^2/2} N((mu - ln K^)/s_x + s_xi/s_x) - K N((mu - ln K^)/s_x)].
5. Forward-start ATM call on [t_s, t_s+tau]: S e^{-q t_s}[e^{-q tau} N(d1) - e^{-r tau} N(d2)], d1 = (b + sigma^2/2) sqrt(tau)/sigma. Uncapped cliquet = sum.
6. Geske, rho = sqrt(T1/T2), S* solves inner BS(S*, K2, T2-T1) = K1; y1 = [ln(S/S*) + (b + sigma^2/2)T1]/(sigma sqrt T1), y2 = y1 - sigma sqrt T1; z1,z2 = d1,d2 at (K2, T2). CoC = S e^{-qT2} M(z1,y1;rho) - K2 e^{-rT2} M(z2,y2;rho) - K1 e^{-rT1} N(y2); PoP = S e^{-qT2} M(-z1,y1;-rho) - K2 e^{-rT2} M(-z2,y2;-rho) + K1 e^{-rT1} N(y2); PoC = K2 e^{-rT2} M(z2,-y2;-rho) - S e^{-qT2} M(z1,-y1;-rho) + K1 e^{-rT1} N(-y2). Parity: CoC - PoC = C_BS(T2) - K1 e^{-rT1}. Simple chooser (choose at t): C_BS(K,T) + K e^{-rT} N(-y2) - S e^{-qT} N(-y1), y1 = [ln(S/K) + bT + sigma^2 t/2]/(sigma sqrt t).
7. Margrabe: s = sqrt(s1^2 + s2^2 - 2 rho s1 s2), V = S1 e^{-q1T} N(d1) - S2 e^{-q2T} N(d2), d1 = [ln(S1 e^{-q1T}/(S2 e^{-q2T})) + s^2T/2]/(s sqrt T); r drops out. Kirk: F_i = S_i e^{(r-q_i)T}, w = F2/(F2+K), s_K = sqrt(s1^2 - 2 rho s1 s2 w + s2^2 w^2), V = e^{-rT}[F1 N(d1) - (F2+K) N(d2)], d1 = [ln(F1/(F2+K)) + s_K^2 T/2]/(s_K sqrt T).
8. Variance swap (DDKZ/Carr-Madan): K_var = (2 e^{rT}/T) sum_i dK_i Q(K_i)/K_i^2, Q = OTM mid (put K < F, call K >= F); dK interior (K_{i+1}-K_{i-1})/2, ends one-sided. VIX term -(1/T)(F/K0 - 1)^2 (K0 = first strike <= F) only if asked. Long P&L = N_var(sigma_R^2 - K_var); N_var = N_vega/(2 K_vol). Realized var = (252/n) sum ln^2 returns, no demeaning.
## Conventions and defaults when the task is silent
- Continuous monitoring for closed forms; Asian dates exclude t=0, include T. MC: seed once, reuse draws across strikes; SE = std(ddof=1)/sqrt(N) of discounted payoffs.
- Moneyness K/S0. MC drift r - q whenever the closed form carries q.
## Completing missing inputs
- No rebate -> R = 0. Lookback b = 0: use the limit (tiny b).
## Invariants to self-check
- CoN_c + CoN_p = e^{-rT}; AoN_c + AoN_p = S e^{-qT}; KI + KO = vanilla; gap(K1=K2) = vanilla.
- Lookback >= vanilla; fixed call(K=S) - float put = S e^{-qT} - S e^{-rT}.
- Geometric < arithmetic Asian; Kirk(K=0) = Margrabe; exchange value falls with rho; chooser >= max(C, P); CoC - PoC parity 1e-10.
- Cliquet period 0 = vanilla ATM(tau); more resets -> larger cliquet.
## Variants to decide
- V1 barrier_monitoring: barrier monitoring. Options: (a) continuous closed form; (b) discrete via BGK shift H e^{+-0.5826 sigma sqrt dt}; (c) raw discrete MC. Default when silent: (a) - Reiner-Rubinstein 1991; (b) Broadie-Glasserman-Kou 1997.
- V2 asian_dates: averaging dates. Options: (a) exclude t=0, include T; (b) include S0. Default when silent: (a) - Kemna-Vorst 1990.
- V3 asian_approx: arithmetic Asian approximation. Options: (a) Levy 1992 lognormal moment match; (b) Turnbull-Wakeman 1991; (c) Curran 1994. Default when silent: (a) - Levy 1992 / Haug 2007.
- V4 mc_se: Monte Carlo standard error. Options: (a) std(ddof=1)/sqrt(N) of discounted payoffs; (b) ddof=0. Default when silent: (a) - Glasserman 2004.
- V5 rebate_timing: knock-out rebate. Options: (a) paid at hit; (b) paid at expiry. Default when silent: (a) - Reiner-Rubinstein 1991 / Haug 2007.
- V6 greek_var_share: Greek VaR decomposition. Options: (a) variance shares 100 V_i/V on a 0-100 scale; (b) Euler component VaR shares; (c) std shares. Default when silent: (a) - Jorion 2007 (equal to (b) for independent factors).
- V7 note_option_count: participation leg of a capital-protected note. Options: (a) participation x notional / S0; (b) participation x notional / K. Default when silent: (a) - Hull ch. 26.
- V8 risk_horizon: VaR/ES horizon for an option or note position. Options: (a) 1 day with sigma/sqrt 252; (b) remaining life T; (c) 10 days. Default when silent: (a) - RiskMetrics 1996.
- V9 markup_base: issuer markup. Options: (a) 100 (issue price - fair)/issue price; (b) relative to fair value. Default when silent: (a) - Das, Structured Products 2001.

## Pitfalls
- Compound parity pairs options on the SAME inner option: CoC - PoC = C_BS(K2,T2) - K1 e^{-rT1} and CoP - PoP = P_BS(K2,T2) - K1 e^{-rT1}. A task that prices CoC and PoP must still COMPUTE PoC (put on the call) for the call parity check - pairing CoC with PoP gives a large constant error.
- Daily-grid MC extrema are biased vs continuous closed forms: Brownian-bridge max, x_max = (x_i + x_{i+1} + sqrt((x_{i+1} - x_i)^2 - 2 sigma^2 dt ln U))/2.
- Put-on-put: outer put exercised when S_T1 > S*.
- Var-swap filters use the OTM side's quotes; recompute dK on the filtered grid.
- barrier_price(S, K, H, T, r, sigma): T BEFORE r - swapping them raises nothing. Cash-or-nothing barrier digitals need cash_digital_barrier, not barrier_price.
- Forward-start ATM call = S0 e^{-q t0} BS(1, 1, tau, r, sigma, q): keep q inside the period formula; q=0 on a discounted spot overprices every cliquet.
- structured_note_report intermediates are {'value': x} dicts (float() fails): emit n['results'] and {'intermediates': n['intermediates']}.
## Sources
Haug 2007 ch.4; Reiner-Rubinstein 1991; Broadie-Glasserman-Kou 1997; GSG 1979; Conze-Viswanathan 1991; Kemna-Vorst 1990; Levy 1992; Curran 1994; Geske 1979; Margrabe 1978; Kirk 1995; DDKZ 1999; CBOE VIX.
