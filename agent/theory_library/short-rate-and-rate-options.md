family: short-rate-and-rate-options
category: fixed-income
keywords: short rate, vasicek, cir, cox-ingersoll-ross, hull-white, feller, mean reversion, affine, zero-coupon bond, trinomial tree, jamshidian, swaption, cap, floor, caplet, black, bachelier, normal volatility, caplet stripping, annuity, ornstein-uhlenbeck, half-life, jump diffusion, bermudan, calibration, jamshidian decomposition, arrow-debreu, r star, co-terminal, exercise boundary, caplet calibration, backward induction, zero-coupon bond put
---
## Worked frame
DIKETAHUI: model parameters (per year, decimal) or rates [T] (percent -> decimal); DFs and times [P]; options one row each [M] (strike, expiry, vol: lognormal decimal or normal bp). Check: sorted times, vol type.
DITANYAKAN: parameters (scalars; half-life unit stated), bond prices [n_tau], option prices [M] (currency).
RUMUS: card Core 2, 1/3/5 (bonds), 6, 8-10 (options); toolkit ar1_fit, ou_from_ar1, vasicek_zcb, black_caplet, black_swaption, swap_rate_and_annuity.
JAWAB: 1) rates as decimals [T]. 2) AR(1) -> parameters. 3) DFs at dates [n+1]. 4) forwards [n], annuity scalar. 5) prices [n] or [M]. 6) bp vol -> decimal; x notional -> currency. 7) check: 0 < P <= 1, cap - floor = swap, payer - receiver parity; then write.

## Core formulas
tau = maturity, P = A exp(-B r). OU/Vasicek: dr = a(b - r)dt + sigma dW.
1. Vasicek: B = (1 - e^{-a tau})/a; ln A = (B - tau)(a^2 b - sigma^2/2)/a^2 - sigma^2 B^2/(4a). E[r_t] = b + (r0-b)e^{-at}; Var = sigma^2 (1 - e^{-2at})/(2a); half-life = ln2/a.
2. OU from data, AR(1) on levels x_{t+1} = alpha + beta x_t + e (dt in years, 1/252 daily): EXACT: a = -ln(beta)/dt, b = alpha/(1-beta), sigma = sd(e) sqrt(2a/(1-beta^2)). EULER/plug-in on differences dx = alpha + beta x + e: a = -beta/dt, b = -alpha/beta, sigma = sd(e)/sqrt(dt). Use whichever the task names; exact is default. Exact simulation: x' = b + (x-b)e^{-a dt} + sigma sqrt((1-e^{-2a dt})/(2a)) Z.
3. CIR: dr = kappa(theta - r)dt + sigma sqrt(r) dW. g = sqrt(kappa^2 + 2 sigma^2); den = (g+kappa)(e^{g tau}-1) + 2g; B = 2(e^{g tau}-1)/den; A = [2g e^{(kappa+g)tau/2}/den]^{2 kappa theta/sigma^2}. Feller: 2 kappa theta >= sigma^2. Long yield -> 2 kappa theta/(g + kappa).
4. CIR transition: c = 2kappa/(sigma^2(1-e^{-kappa dt})); 2c r' ~ ncx2(df = 4 kappa theta/sigma^2, nc = 2c r e^{-kappa dt}); log p(r') = ncx2.logpdf(2c r') + ln(2c) (Jacobian). If the task writes nc = 2 c r, follow the task literally.
5. Hull-White: dr = (theta(t) - a r)dt + sigma dW. B(t,T) = (1-e^{-a(T-t)})/a; ln A = ln(P(0,T)/P(0,t)) + B f(0,t) - sigma^2 (1-e^{-2at}) B^2/(4a). ZCB option (expiry T on bond S): sp = sigma sqrt((1-e^{-2aT})/(2a)) B(T,S); h = ln(P(0,S)/(P(0,T)K))/sp + sp/2; ZBC = P(0,S)N(h) - K P(0,T)N(h-sp); ZBP = K P(0,T)N(-h+sp) - P(0,S)N(-h). Caplet (T1->T2, accrual d) = (1+Kd) ZBP(T1, T2, 1/(1+Kd)).
6. Jamshidian: payer swaption = put on coupon bond, c_i = K d_i, c_n += 1, strike 1. Solve sum c_i P(T0,Ti,r*) = 1; price = sum c_i ZBP(T0, Ti, X_i = P(T0,Ti,r*)). Receiver = same with ZBC.
7. HW trinomial tree: dx = sigma sqrt(3dt), jmax = ceil(0.184/(a dt)), M = -a dt; normal branch pu = 1/6 + (j^2M^2 + jM)/2, pm = 2/3 - j^2M^2, pd = 1/6 + (j^2M^2 - jM)/2; edge branches at +-jmax. Arrow-Debreu: alpha_m = [ln sum_j Q_m(j) e^{-j dx dt} - ln P(0,t_{m+1})]/dt.
8. Black-76 caplet: d DF(T_pay)[F N(d1) - K N(d2)], d1 = (ln(F/K) + s^2 T_fix/2)/(s sqrt(T_fix)); floorlet d DF[K N(-d2) - F N(-d1)]. Cap - floor = payer swap = sum d DF (F - K).
9. Bachelier: caplet = d DF[(F-K)N(x) + s_n sqrt(T) n(x)], x = (F-K)/(s_n sqrt T); s_n ~ s_black * F.
10. Swaption (Black): payer = A[S N(d1) - K N(d2)], receiver = A[K N(-d2) - S N(-d1)], A = sum tau_i DF(T_i) over fixed-leg dates, S = (DF(T0) - DF(Tn))/A; payer - receiver = A(S-K).
11. Log-OU with jumps (X = ln S, dX = kappa(theta - X)dt + sigma dW + J dN, J~N(mu_J, s_J^2)), e = e^{-kappa tau}: E[X] = theta + (x0-theta)e + lambda mu_J(1-e)/kappa; Var = (sigma^2 + lambda(mu_J^2 + s_J^2))(1-e^2)/(2kappa); E[S] = exp(m + v/2).
## Conventions and defaults when the task is silent
- Option expiry for caplet = FIXING time (accrual start); discount to PAYMENT date. A caplet with T_fix = 0 is intrinsic; USD market cap quotes exclude the first (already fixed) caplet.
- Vol units: lognormal as decimal (20 -> 0.20); normal in bp/yr (85 -> 0.0085). Cue: vol ~ 0.5-1.5% of rate level => normal; 10-60% => lognormal. Negative/zero rates => Bachelier or shifted lognormal.
- dt = 1/252 for daily business-day data; rates percent -> decimal; sample std ddof=1.
- HW curve: interpolate DFs log-linearly unless stated; f(0,t) at a pillar = right-hand forward.
- Swaption expiry/tenor "1Yx5Y" = option 1Y into a 5Y swap; ATM strike = forward par rate.
- Payer swaption/payer swap: value rises with rates (DV01 > 0).
## Completing missing inputs
- Caplet stripping from flat cap vols: cap_n(s_n flat) - cap_{n-1}(s_{n-1} flat) = caplet_n(sigma_n); solve caplet vols sequentially (brentq).
- r0 for HW = f(0,0) (short end of curve); theta(t) is implied by the curve, never fitted.
- MLE bounds hit (kappa at bound): report anyway, note it; use multi-start and keep best LL.
- Forward rate for caplet: F = (DF(T1)/DF(T2) - 1)/d.
## Invariants to self-check
- HW/Vasicek/CIR: 0 < P <= 1, decreasing in tau; P(0,T)_HW reproduces market DFs exactly; tree sum_j Q_n(j) = P(0,t_n) (1e-10).
- Bermudan >= European; tree vs Jamshidian within a few bp; exercise boundary monotone.
- Caplet, floorlet >= 0; caplet increasing in s and F; per-period parity to machine precision.
- Feller ratio reported = 2 kappa theta/sigma^2; B(tau) < tau; y(tau) -> r as tau -> 0.
- Telescoping: sum forward_i dt_i = -ln P(tau_last).
## Variants to decide
- V1 vol_type: cap/swaption vol. Options: (a) lognormal Black-76; (b) normal Bachelier; (c) shifted lognormal. Default when silent: (a) - Black 1976 / Hull; magnitude cue wins.
- V2 first_caplet: fixed first caplet. Options: (a) exclude; (b) include. Default when silent: (a) - Hull (USD market cap).
- V3 ou_estimation: Vasicek estimation. Options: (a) exact AR(1) kappa=-ln(b)/dt; (b) Euler kappa=(1-b)/dt. Default when silent: (a) - Brigo-Mercurio 2006.
- V4 cir_sampling: CIR MC scheme. Options: (a) exact noncentral chi-square; (b) full-truncation Euler. Default when silent: (a) - Glasserman 2004.
- V5 dt_basis: daily step. Options: (a) 1/252; (b) 1/365. Default when silent: (a) - Hull.
- V6 caplet_expiry: HW caplet option time. Options: (a) fixing date T1 = T2 - accrual, bond maturity T2; (b) payment date T2. Default when silent: (a) - Brigo-Mercurio 2006.
- V7 hw_calibration_target: fit objective. Options: (a) SSE of model vs market prices; (b) SSE of Black implied vols; (c) relative price errors. Default when silent: (a) - Brigo-Mercurio 2006 ch. 3.
- V8 bermudan_exercise: exercise value at t_k. Options: (a) coupons strictly after t_k, alpha of that step; (b) include the coupon at t_k. Default when silent: (a) - Andersen-Piterbarg 2010.
- V9 tree_bond_value: bond at tree nodes. Options: (a) analytic P_HW(t,T,r_node); (b) rolled back on the tree. Default when silent: (a) - Hull-White 1994.

## Pitfalls
- Payment time instead of fixing time in d1; dividing by sqrt(0) for the first caplet.
- Forgetting the ncx2 Jacobian (LL shifts); maximising the wrong sign.
- Vasicek variance with sigma^2/(2a) where A needs sigma^2 B^2/(4a).
- beta >= 1 in AR(1) => ln(beta) >= 0, no mean reversion: kappa <= 0 -> set half-life NaN.
- Bermudan exercise value including the coupon paid at t_k (use payments strictly after); using one alpha for all exercise dates.
- Annuity with float-leg accruals; swaption with normal vol in a Black formula.
- Jump detection: residual std threshold (|z| > 3) on OU residuals, do not refit unless told.
- Long Hull-White pipelines get their replies cut when the code carries prose comments; call hw_caplet_price, hw_jamshidian_swaption, hw_trinomial_tree and hw_tree_bermudan instead of re-deriving them.
- Hull-White helpers return dicts ({price, r_star}, {price, boundary}), never tuples; hw_swaption_bundle builds every deliverable in one call.
## Sources
Vasicek (1977); Cox-Ingersoll-Ross (1985); Hull-White (1990, 1994 trees); Jamshidian (1989); Brigo-Mercurio Interest Rate Models ch. 3; Black (1976); Bachelier; Hull OFOD ch. 29-32; Glasserman Monte Carlo ch. 3.4.
