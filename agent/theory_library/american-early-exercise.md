family: american-early-exercise
category: derivatives-pricing
keywords: american option, early exercise, exercise boundary, critical price, barone-adesi-whaley, baw, bjerksund-stensland, quadratic approximation, psor, projected sor, crank-nicolson, binomial tree, trinomial, crr, early exercise premium, cash dividend, free boundary, richardson, perpetual, projected successive over-relaxation, discrete dividends, cash dividends, early exercise boundary, finite differences, dirichlet boundary
---
## Worked frame
DIKETAHUI: S, K (currency), T (years), r, q, sigma (decimal per year), scalars or one row per contract [M]; cash dividends [D]; steps N. Check: percent -> decimal, unique ids.
DITANYAKAN: American and European price, premium (currency) [M]; critical price [M] or boundary [N+1].
RUMUS: card Core 1 (tree), 2 (PSOR) or 3 (BAW); toolkit crr_price, baw_american, bs_price.
JAWAB: 1) inputs as [M] columns. 2) European [M]. 3) American by the named method [M]. 4) premium = American - European [M]. 5) boundary [N+1] or critical price [M]. 6) convert to output unit (currency, rounding last). 7) check: American >= European >= 0 and >= intrinsic, put boundary < K; then write.

## Core formulas
1. Tree/grid (CRR as in bsm card): at every node V = max(continuation, payoff); continuation = e^{-r dt}[p V_u + (1-p) V_d]. Trinomial (Boyle): u = e^{sigma sqrt(2 dt)}, pu = ((e^{b dt/2} - e^{-sigma sqrt(dt/2)})/(e^{sigma sqrt(dt/2)} - e^{-sigma sqrt(dt/2)}))^2, pd analogous with signs swapped, pm = 1 - pu - pd.
2. PSOR on CN system A v = rhs: y = (rhs_i + alpha_i v_{i-1} + gamma_i v_{i+1})/(1 - beta_i) (A has sub -alpha, diag 1-beta, super -gamma); v_i <- max(payoff_i, v_i + omega(y - v_i)); Gauss-Seidel order (new v_{i-1}); omega 1.2-1.5; stop when max|dv| < tol. Alternatives: Brennan-Schwartz direct solve (exact for put), or penalty method.
3. BAW (b = r - q): M = 2r/sigma^2, N = 2b/sigma^2, K(T) = 1 - e^{-rT}. q2 = [-(N-1) + sqrt((N-1)^2 + 4M/K)]/2 (call), q1 = [-(N-1) - sqrt((N-1)^2 + 4M/K)]/2 (put, < 0).
   Call: S* solves S* - X = c(S*) + [1 - e^{(b-r)T} N(d1(S*))] S*/q2. A2 = (S*/q2)[1 - e^{(b-r)T} N(d1(S*))]. C = c(S) + A2 (S/S*)^q2 if S < S*, else S - X.
   Put: S** solves X - S** = p(S**) - [1 - e^{(b-r)T} N(-d1(S**))] S**/q1. A1 = -(S**/q1)[1 - e^{(b-r)T} N(-d1(S**))]. P = p(S) + A1 (S/S**)^q1 if S > S**, else X - S.
   If b >= r (q <= 0) the American call = European (no early exercise); report S* = inf or NaN as the task wants, premium 0.
4. Bjerksund-Stensland (1993) call: beta = (1/2 - b/sigma^2) + sqrt((b/sigma^2 - 1/2)^2 + 2r/sigma^2); B_inf = beta X/(beta-1); B0 = max(X, r X/(r-b)); h = -(bT + 2 sigma sqrt T) B0/(B_inf - B0); I = B0 + (B_inf - B0)(1 - e^{h}); alpha = (I - X) I^{-beta}.
   C = alpha S^beta - alpha phi(S,T,beta,I,I) + phi(S,T,1,I,I) - phi(S,T,1,X,I) - X phi(S,T,0,I,I) + X phi(S,T,0,X,I) for S < I, else S - X.
   phi(S,T,g,H,I) = e^{lam T} S^g [N(d) - (I/S)^kap N(d - 2 ln(I/S)/(sigma sqrt T))], lam = -r + g b + 0.5 g(g-1) sigma^2, d = -[ln(S/H) + (b + (g - 0.5)sigma^2)T]/(sigma sqrt T), kap = 2b/sigma^2 + 2g - 1.
   Put by transformation: P(S,X,T,r,b,sigma) = C(X,S,T,r-b,-b,sigma). (BS2002 two-step version is more accurate; use only if named.)
5. Perpetual put: g = [-(N-1) - sqrt((N-1)^2 + 4M)]/2 (< 0), S* = X g/(g-1), value (X - S*)(S/S*)^g for S > S*.
6. Richardson for a 2nd-order scheme halving dS AND dt: P = (4 P_fine - P_coarse)/3; ratio |c - rich|/|f - rich| ~ 4.
## Conventions and defaults when the task is silent
- Early exercise checked at every time step (Bermudan on the grid). Payoff at exercise uses intrinsic, undiscounted.
- Exercise boundary S*(t) for a put: largest S with V(S) = payoff(S) and payoff > 0 (scan high S downward); record 0.0 when none. Tolerance: PSOR stops on increments, so use tol ~1e-6..1e-4 on |V - payoff|, not machine zero. At expiry S*(T) = X min(1, r/q) for puts (= X when q = 0); calls: X max(1, r/q).
- Discrete cash dividend at t_d (backward): V(S, t_d-) = V(S - D, t_d+) by interpolation (clip S - D at 0), then re-apply max(V, payoff) for American. Step index = round(t_d/dt). Use continuous q only if given as a yield.
- Brent brackets for BAW critical prices: call [X, 2X] doubling upper until sign change; put [X/2 halving, X]. Seed (Haug): S* from the T -> inf limit.
- CRR default N from the task; if silent N >= 500; average N and N+1 to kill oscillation.
## Completing missing inputs
- American call on a non-dividend stock (q = 0, no cash divs) = European call exactly; report the European value (a grid will differ by discretisation ~1e-4).
- Early exercise premium = American - European, same inputs and same method where possible.
- Put-call parity is an INEQUALITY for Americans: S e^{-qT} - X <= C - P <= S - X e^{-rT} (q = 0 form: S - X <= C - P <= S - X e^{-rT}).
## Invariants to self-check
- American >= European >= 0; American >= intrinsic everywhere on the grid.
- Put boundary S** < X, decreasing in time-to-maturity T (towards perpetual S*); call boundary S* > X, increasing in T.
- Call premium increases with q; put premium largest deep ITM (until exercised immediately: then value = intrinsic, premium = intrinsic - European).
- BAW vs CRR(N=1000): rel diff typically 0.1-1%, larger for long T.
- Dividends raise puts, lower calls. Delta_put in [-1, 0]; American put delta = -1 below S**.
## Variants to decide
- V1 tree_param: binomial tree parametrisation. Options: (a) CRR u=e^{sigma sqrt dt}, d=1/u, p=(e^{(r-q)dt}-d)/(u-d); (b) Jarrow-Rudd equal-probability; (c) Leisen-Reimer (odd N). Default when silent: (a) - Cox-Ross-Rubinstein 1979 / Hull.
- V2 tree_smoothing: odd/even oscillation treatment. Options: (a) plain N steps; (b) average of N and N+1 steps; (c) BBS / Richardson extrapolation. Default when silent: (b) - Hull (oscillation note); Broadie-Detemple 1996 for (c).
- V3 exercise_dates: where early exercise is tested. Options: (a) every grid/tree step incl. t=0; (b) only stated Bermudan dates. Default when silent: (a) - Hull (American = check max(V, payoff) at every node).
- V4 analytic_approx: closed-form American approximation. Options: (a) Barone-Adesi-Whaley 1987 quadratic; (b) Bjerksund-Stensland 1993; (c) Bjerksund-Stensland 2002. Default when silent: (a) - Barone-Adesi-Whaley 1987 / Haug 2007.
- V5 cash_dividend: discrete cash dividend model. Options: (a) jump condition V(S,t_d-)=V(S-D,t_d+) on the grid; (b) escrowed: model S-PV(D), add PV(D) back; (c) equivalent continuous yield. Default when silent: (a) - Wilmott (PDE); Hull uses (b) for trees.
- V6 fd_scheme: PDE time stepping for the free boundary. Options: (a) Crank-Nicolson + PSOR; (b) fully implicit + PSOR; (c) explicit with max(V, payoff). Default when silent: (a) - Wilmott-Howison-Dewynne 1995.
- V7 cn_operator: CN matrices. Options: (a) (I - 0.5 dt L) V^n = (I + 0.5 dt L) V^{n+1} with Dirichlet identity rows at 0 and Smax; (b) Rannacher start (implicit steps) then (a). Default when silent: (a) - Crank-Nicolson 1947 / Wilmott.
- V8 boundary_scan: exercise boundary search. Options: (a) largest S with |V - payoff| < tol among nodes with payoff > 0 (S*(T) at the strike side); (b) over all nodes. Default when silent: (a) - Wilmott-Howison-Dewynne 1995.
- V9 dividend_projection: ex-dividend layer. Options: (a) jump V(S-D) then project max(V, payoff); (b) jump only, project at regular steps. Default when silent: (a) - Wilmott (exercise allowed just before the ex-date).
- V10 grid_layout: sampled V(S,t) output. Options: (a) rows S ascending, columns calendar t ascending from t=0; (b) columns time-to-maturity. Default when silent: (a) - task time grid t_n = n dt.

## Pitfalls
- Dividend jump direction (S - D, not S + D) or applied at the wrong layer; forgetting to re-project after the jump.
- Using e^{-qT} vs e^{(b-r)T}: identical, but BAW with futures (b = 0) uses e^{-rT}.
- In BAW, K(T) = 1 - e^{-rT} uses r, not b; N uses b, M uses r.
- PSOR in pure Python over 600 steps x 300 nodes x several runs: vectorise Thomas for Europeans, cap iterations, or numba.
- Exercise boundary with tolerance 1e-10 gives spurious zeros; with a too-loose one it drifts above X - cap at X.
- Richardson with only one of dS/dt halved is invalid.
- Cash dividends as {'time', 'amount'} dicts: a (time, amount) tuple read as (amount, time) drops the dividend silently (prices with = without dividends).
## Sources
Barone-Adesi and Whaley (1987, J. Finance); Bjerksund and Stensland (1993, 2002); Haug, Complete Guide to Option Pricing Formulas (2007, ch. 3); Hull (ch. 21, 27); Wilmott (PSOR, free boundary); Brennan-Schwartz (1977); Boyle (1986) trinomial.
