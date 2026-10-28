family: bsm-greeks-pde
category: derivatives-pricing
keywords: black-scholes, greeks, delta, gamma, vega, theta, rho, pde, finite difference, crank-nicolson, explicit, implicit, grid, boundary condition, dividend yield, pde residual, binomial, crr, leisen-reimer, put-call parity, per calendar day, richardson, merton jump diffusion, bump and revalue
---
## Worked frame
DIKETAHUI: S, K (currency), T (years), r, q, sigma (percent -> decimal), scalars or one row per option [M]; price history [T] if sigma is estimated; grid sizes N_S, N_t. Check: row order and id dtype kept.
DITANYAKAN: price and Greeks [M] in the asked unit (theta per year or day, vega/rho per unit or point); grid values [N_S+1]; PDE residual [M].
RUMUS: card Core 1-3 (closed form), 4 (residual), 5-7 (grid, tree); toolkit bs_price, bs_greeks, crr_price.
JAWAB: 1) inputs [M]. 2) price [M]. 3) Greeks [M] per year, per unit. 4) grid branch: layer [N_S+1] stepped N_t times, read at S0 -> scalar. 5) residual, parity [M]. 6) rescale Greeks to output unit. 7) check: parity, delta bounds, gamma and vega > 0; then write.

## Core formulas
1. BSM with continuous yield q (cost of carry b = r - q): d1 = [ln(S/K) + (r - q + sigma^2/2)T]/(sigma sqrt T), d2 = d1 - sigma sqrt T. C = S e^{-qT} N(d1) - K e^{-rT} N(d2); P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1). Forward form: C = e^{-rT}[F N(d1) - K N(d2)], F = S e^{(r-q)T}.
2. Delta_C = e^{-qT} N(d1); Delta_P = e^{-qT}(N(d1) - 1). Gamma = e^{-qT} n(d1)/(S sigma sqrt T). Vega = S e^{-qT} n(d1) sqrt T (per 1.00 of sigma). Rho_C = K T e^{-rT} N(d2); Rho_P = -K T e^{-rT} N(-d2).
3. Theta (dV/dt, per YEAR): C: -S e^{-qT} n(d1) sigma/(2 sqrt T) - r K e^{-rT} N(d2) + q S e^{-qT} N(d1); P: -S e^{-qT} n(d1) sigma/(2 sqrt T) + r K e^{-rT} N(-d2) - q S e^{-qT} N(-d1).
4. PDE: V_t + (r - q) S V_S + 0.5 sigma^2 S^2 V_SS - r V = 0, so residual Theta + (r-q) S Delta + 0.5 sigma^2 S^2 Gamma - r V = 0 to ~1e-12 with per-year theta.
5. FD theta-scheme on S-grid (index i, S_i = i dS): a_i = 0.5 dt (sigma^2 i^2 - (r-q) i), b_i = -dt(sigma^2 i^2 + r), c_i = 0.5 dt(sigma^2 i^2 + (r-q) i). Explicit: V^n_i = a V^{n+1}_{i-1} + (1+b) V^{n+1}_i + c V^{n+1}_{i+1}; stable only if dt <= 1/(sigma^2 N_S^2) roughly. Implicit: unconditionally stable, O(dt). CN: average (halve a,b,c each side), O(dt^2 + dS^2); solve tridiagonal (Thomas / scipy.linalg.solve_banded).
6. Log grid x = ln S: V_t + (r - q - sigma^2/2) V_x + 0.5 sigma^2 V_xx - rV = 0 (constant coefficients).
7. CRR: u = e^{sigma sqrt dt}, d = 1/u, p = (e^{(r-q)dt} - d)/(u - d), discount e^{-r dt}. Leisen-Reimer (N odd): h(z) = 0.5 + sign(z) sqrt(0.25 - 0.25 exp(-(z/(N + 1/3 + 0.1/(N+1)))^2 (N + 1/6))); p = h(d2), p' = h(d1), u = e^{(r-q)dt} p'/p, d = (e^{(r-q)dt} - p u)/(1 - p). Jarrow-Rudd: u,d = exp((r-q-sigma^2/2)dt +/- sigma sqrt dt), p = 0.5.
## Conventions and defaults when the task is silent
- Theta: default dV/dt (negative for long vanillas). "per day" -> /365 calendar (Hull); /252 only if "trading day". "per year" or a PDE-residual check -> no scaling. -dV/dT convention only if stated.
- Vega/rho: default per 1.00 (decimal). "per vol point"/"per 1%" -> /100; "per bp" -> /10000.
- sigma, r, q as decimals; values like 20 or 5.25 are percent -> divide by 100. r continuous unless "annual compounding" (then r_c = ln(1+r)).
- Historical sigma: log returns, ddof=1, x sqrt(252); n_returns = n_prices - 1; S0 = last close after sorting by date.
- FD boundaries (tau = T - t): call V(0)=0, V(Smax) = Smax e^{-q tau} - K e^{-r tau}; put V(0) = K e^{-r tau}, V(Smax)=0. Alternative: V_SS = 0 (linearity) at both ends. Smax default 3-5 K or K exp(5 sigma sqrt T); put K (and S0) on nodes.
- Greeks from grid: delta/gamma by central differences at S0 node (non-uniform: use 3-point Lagrange weights); theta from t=0 and t=dt layers; vega/rho by bumping the parameter and re-solving on the SAME grid (central, bump 1e-4..1e-3).
- Moneyness: "m = S/K" vs "K/S0" - read which is fixed. When the grid varies S at fixed K = S0, m = S/K.
## Completing missing inputs
- F and DF from C-P regressed on K across strikes: slope = -DF, intercept = DF F.
- q from forward: q = r - ln(F/S)/T. Discrete dividends: European -> S minus PV(divs); FD -> jump V(S, t_d-) = V(S - D, t_d+).
- If a task says "solve the PDE" but checks only consistency, the closed form IS the exact PDE solution; FD Greeks must still match differences of your own price function.
## Invariants to self-check
- C - P = S e^{-qT} - K e^{-rT} (1e-10 for closed form, ~1e-3 for grids).
- 0 < Delta_C < e^{-qT}; -e^{-qT} < Delta_P < 0; Delta_C - Delta_P = e^{-qT}; Gamma, Vega > 0 and equal for C and P; Rho_C - Rho_P = K T e^{-rT}.
- max(0, S e^{-qT} - K e^{-rT}) <= C <= S e^{-qT}; P <= K e^{-rT}.
- Greek vs bump: delta ~ [V(S+h)-V(S-h)]/2h with h = 1e-4 S; theta_day ~ V(T - 1/365) - V(T).
- Gamma argmax over spot is NOT exactly ATM: S* = K exp(-(r - q + 1.5 sigma^2)T) (below K). Report the computed argmax.
- Grid convergence: CN error ratio ~4 per halving of both dS and dt; Richardson (4 P_fine - P_coarse)/3.
## Variants to decide
- V1 theta_unit: theta time unit. Options: (a) per year dV/dt; (b) per calendar day /365; (c) per trading day /252. Default when silent: (a) - Hull; (b) when "per day" is said.
- V2 vega_rho_unit: vega/rho scaling. Options: (a) per 1.00 (decimal); (b) per 1 vol point / 1% (/100). Default when silent: (a) - Hull.
- V3 rate_compounding: r convention. Options: (a) continuous; (b) annual compounding converted r_c=ln(1+r). Default when silent: (a) - Black-Scholes 1973 / Hull.
- V4 hist_vol: historical sigma estimate. Options: (a) log returns, ddof=1, x sqrt(252); (b) ddof=0; (c) simple returns. Default when silent: (a) - Hull.
- V5 fd_scheme: finite-difference scheme. Options: (a) Crank-Nicolson; (b) fully implicit; (c) explicit. Default when silent: (a) - Wilmott / Hull.
- V6 fd_boundary: far-field boundary. Options: (a) Dirichlet asymptotic values (call Smax e^{-q tau}-K e^{-r tau}); (b) V_SS=0 linearity. Default when silent: (a) - Wilmott-Howison-Dewynne 1995.
- V7 tree_param: binomial parametrisation. Options: (a) CRR; (b) Jarrow-Rudd; (c) Leisen-Reimer. Default when silent: (a) - Cox-Ross-Rubinstein 1979.
- V8 euro_cash_dividend: European with discrete cash dividends. Options: (a) escrowed S - PV(D) in BS; (b) PDE jump V(S, t_d-) = V(S - D, t_d+); (c) equivalent continuous yield. Default when silent: (a) - Hull (escrowed model).
- V9 bump_type: numerical Greek bump. Options: (a) relative h = eps x S (central); (b) absolute h; (c) complex-step. Default when silent: (a) - Haug 2007 numerical Greeks.

## Pitfalls
- Dropping e^{-qT} from delta/gamma/vega, or the +q S e^{-qT} N(d1) term from call theta (PDE residual then equals that term).
- CN on a payoff kink: oscillating gamma near K for short T. Fix: Rannacher start (2-4 fully implicit half-steps) or smooth payoff (cell average).
- T < 5 days on a coarse S grid: gamma spike unresolved -> closed form or a fine local grid.
- CRR odd/even oscillation: average N and N+1, or use Leisen-Reimer. Tree delta = (V_u - V_d)/(S u - S d) at step 1.
- Tolerance-free float equality when selecting ATM rows; use np.isclose.
- Preserve input row order and id dtype (string); don't write the pandas index.
- Never assert a textbook expectation (e.g. gamma peak at m=1) and abort; report computed values.
## Sources
Hull, Options Futures and Other Derivatives (ch. 15, 19, 21); Wilmott, Paul Wilmott on Quantitative Finance (FD ch.); Haug, Complete Guide to Option Pricing Formulas (2007); Leisen-Reimer (1996); Rannacher (1984); Duffy, Finite Difference Methods in Financial Engineering (2006).
