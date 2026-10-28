family: monte-carlo-greeks-and-hedging
category: derivatives-pricing
keywords: monte carlo, standard error, antithetic, control variate, common random numbers, pathwise, likelihood ratio, score function, finite difference, bump and revalue, greeks, asian option, delta hedging, rebalancing, hedging pnl, transaction costs, gamma theta, realised volatility, weather derivative, hdd, cdd, degree day, burn analysis, tick value, temperature, ipa, infinitesimal perturbation analysis, score function greeks, variance ratio, convergence study, delta surface, vega surface
---
## Worked frame
DIKETAHUI: S0, K (currency), T (years), r, q, sigma (decimal), n_paths, n_steps, seed, bumps; hedging: costs in bps. Check: named generator, draw order.
DITANYAKAN: price, SE, Greeks one key each (null where undefined); hedging P&L [n_paths] (currency).
RUMUS: card Core 1-3 (paths, SE), 6-8 (Greeks), 9 (hedging); toolkit gbm_paths, bs_price, bs_greeks, delta_hedge_pnl.
JAWAB: 1) draw Z [n_paths x n_steps] once. 2) paths [n_paths x n_steps+1]. 3) discounted payoff [n_paths] -> price, SE scalars. 4) Greeks reusing the same Z -> scalars. 5) hedge loop over steps: cash, shares [n_paths]. 6) rescale Greeks (per point, per day); bps -> fraction. 7) check: MC within its SE band of the closed form, delta bounds; then write.

## Core formulas
1. Exact GBM: S_T = S0 exp((r-q-sigma^2/2)T + sigma sqrt(T) Z); multi-date: cumsum of log increments on t_j = jT/n. Arithmetic Asian averages the n monitoring dates (t_1..t_n, excluding t_0 unless stated).
2. Price = e^{-rT} mean(payoff); SE = std(disc payoff, ddof=1)/sqrt(N).
3. Antithetic: Y_i = (f(Z_i) + f(-Z_i))/2, SE = std(Y, ddof=1)/sqrt(N_pairs). Control variate: Y - b(X - E[X]), b = Cov(Y,X)/Var(X); controls S_T (E = S0 e^{(r-q)T}) or geometric Asian.
4. Geometric Asian (n dates): mG = ln S0 + (r-q-sigma^2/2)T(n+1)/(2n), vG^2 = sigma^2 T(n+1)(2n+1)/(6n^2); call = e^{-rT}[e^{mG + vG^2/2} N(d1) - K N(d2)], d1 = (mG - ln K + vG^2)/vG, d2 = d1 - vG.
5. BS with q: delta_c = e^{-qT}N(d1), delta_p = delta_c - e^{-qT}; gamma = e^{-qT}n(d1)/(S sigma sqrt T); vega = S e^{-qT} n(d1) sqrt T; theta_c (dV/dt, per year) = -S e^{-qT}n(d1)sigma/(2 sqrt T) - rK e^{-rT}N(d2) + qS e^{-qT}N(d1); rho_c = KT e^{-rT}N(d2). Parity: delta_c-delta_p = e^{-qT}, gamma/vega equal, rho_c-rho_p = KT e^{-rT}.
6. Central bumps: delta (V+ - V-)/2h, gamma (V+ - 2V + V-)/h^2, same Z for all legs (CRN).
7. Pathwise (European call; put: flip sign, indicator S_T<K): delta e^{-rT}1{S_T>K} S_T/S0; vega e^{-rT}1{} S_T(sqrt(T)Z - sigma T); rho e^{-rT}1{} K T. Asian: delta e^{-rT}1{A>K}A/S0; vega e^{-rT}1{} mean_j S_j(W_j - sigma t_j). Pathwise gamma undefined (null).
8. Likelihood ratio (weights x discounted payoff): delta Z/(S0 sigma sqrt T); gamma (Z^2 - 1 - Z sigma sqrt T)/(S0 sigma sqrt T)^2; vega (Z^2-1)/sigma - Z sqrt T; rho Z sqrt(T)/sigma - T.
9. Hedging P&L of a delta-hedged long option over dt: 0.5 Gamma S^2 (realised var - implied var) dt (gamma-theta); total ~ int e^{-rt} 0.5 Gamma_i S^2 (sigma_r^2 - sigma_i^2)dt. Discrete hedging error sd ~ sqrt(pi/4) vega sigma/sqrt(N_rebal) (Kamal-Derman). Leland vol for a short option with round-trip cost k: sigma^2(1 + sqrt(2/pi) k/(sigma sqrt dt)).
10. Escrowed dividends: S_eff = S - sum_{ex_j > t} D_j e^{-r(t_j - t)}; price BS(S_eff), hedge delta N(d1) w.r.t. observed S.
11. Weather: Tavg = (Tmax+Tmin)/2; HDD = max(Tbase - Tavg, 0), CDD = max(Tavg - Tbase, 0), Tbase 65F = 18.333C ((F-32)5/9); index = monthly/seasonal sum; call payoff tick x max(I - K, 0), put tick x max(K - I, 0), optional cap. CME US HDD/CDD: USD 20 per index point.
12. Temperature model: T = s(t) + x, x OU; exact step x' = x e^{-kappa dt} + sigma sqrt((1-e^{-2kappa dt})/(2kappa)) Z; s(t) = A + Bt + C sin(2 pi t/365 + phi) or day-of-month average.
## Conventions and defaults when the task is silent
- A Monte Carlo "validates" / "agrees" flag with no tolerance stated: every |MC - closed form| <= 3 standard errors of that MC estimate (report the SE); an absolute cut like 0.01 is not the default.
- Unit scaling: follow the literal formula over the prose label. Market defaults if only prose: vega per 1 vol point = dV/dsigma/100, rho per 1% = /100, theta per day = dV/dt/365 (per trading day /252). Theta sign: dV/dt (negative for long vanilla).
- One RNG created once; draw order matters (e.g. European Z, then Asian increments). np.random.seed / RandomState vs default_rng are different streams: use the named one.
- Bumped revaluations re-use the same normals (re-seed each leg). Relative bumps: h = 0.01 x parameter, Greek = (P_up - P_dn)/(2h) in parameter units.
- Hedging loop order per day: accrue interest (cash e^{r dt}) -> credit dividend on shares held from previous close -> rebalance if t % f == 0 (0 < t < T) with TC on |dN| S. Expiry: no rebalance, liquidate (sell TC), pay payoff. P&L = terminal cash, not discounted.
- bps / 10000. Buy rate when shares increase, sell rate when decrease. Trading-day clock T_rem = (N - t)/N if stated.
- Burn analysis: equal-weight years, no detrending unless asked; discount by e^{-r days/365}.
- GHCN-Daily: headerless, TMAX/TMIN in tenths C (/10); filter station, element, month, year range; pivot; drop days missing either.
- OU on anomalies: seasonal mean by calendar day; regression pairs only for adjacent days in the same year; dt = 1/365; kappa = -beta/dt (or -ln(1+beta)/dt if exact inversion asked), sigma = std(eps, ddof=1)/sqrt(dt).
## Completing missing inputs
- No seed: fix one; no N: >= 1e5 paths. BS closed form validates European MC Greeks.
- MC day loop: accrue index from current x BEFORE advancing when told day 1 uses x_0.
## Invariants to self-check
- |MC - BS| <= ~3 SE; pathwise delta variance < LR; Asian price and ATM delta < European.
- Call delta in [0, e^{-qT}], put delta in [-e^{-qT}, 0], gamma, vega > 0; FD parity differences hold (exact with CRN).
- Rebalance count = floor((N-1)/f); TC total decreases with coarser rebalancing; zero-TC daily-hedge P&L small vs premium.
- HDD >= 0; Tmin <= Tavg <= Tmax; clean days = years x days_in_month if complete; HDD-call temp delta < 0, CDD-call > 0, vega > 0; burn price = DF x mean payoff.
## Variants to decide
- V1 rng_api: random generator. Options: (a) np.random.default_rng(seed); (b) legacy np.random.seed / RandomState. Default when silent: (a) - NumPy NEP 19 (named API always wins).
- V2 greek_method: MC Greek estimator. Options: (a) pathwise (smooth payoffs); (b) likelihood ratio; (c) bump and revalue with CRN. Default when silent: (a) - Broadie-Glasserman 1996.
- V3 bump_scheme: finite-difference bump. Options: (a) central, h=1% of parameter, same normals; (b) forward. Default when silent: (a) - Glasserman 2004.
- V4 se_ddof: standard error. Options: (a) std(ddof=1)/sqrt(N); (b) ddof=0. Default when silent: (a) - Glasserman 2004.
- V5 antithetic_se: antithetic SE. Options: (a) from N/2 pair averages; (b) naive over all N draws. Default when silent: (a) - Glasserman 2004.
- V6 vega_scale: MC vega unit when prose and formula disagree. Options: (a) the written formula ((V+ - V-)/(2h), per 1.00 sigma); (b) per vol point /100. Default when silent: (a) - Glasserman 2003 (literal formula over prose label).
- V7 theta_scale: MC theta unit. Options: (a) the written formula (per year when h is in years); (b) per calendar day /365; (c) per trading day /252. Default when silent: (a) - Hull (literal formula over prose label).
- V8 price_keys: price record keys when only prose names them. Options: (a) mc_price, mc_se, bs_price; (b) descriptive long names (mc_standard_error, bs_reference_price). Default when silent: (a) - Glasserman 2003 notation (price, SE, closed form).

## Pitfalls
- SE from undiscounted payoffs, ddof=0, or on 2N antithetic draws.
- Independent Z per bump -> noise swamps the Greek.
- Pathwise rho forgetting the discount term; LR rho without -T.
- Unavailable Greeks as 0/NaN when null required.
- Asian theta: rebuild the date grid with the shortened T, same Z.
- Using S instead of S_eff; extra e^{-qT} on escrowed delta; ex-day dividend counted in PV on its own day (use strict >).
- Rebalancing on expiry; forgetting interest the final day or TC on initial/terminal trades.
- HDD with Fahrenheit base on Celsius temps; cross-year OU pairs; Euler instead of exact step.
- Discrete hedge books: every rebalance and terminal fee must be deducted from cash, not only summed (escrowed_dividend_delta_hedge). MC Greek suites: one normal block for every bump (mc_greeks_suite), pathwise theta signed like the written FD formula.
## Sources
Glasserman 2003; Broadie-Glasserman 1996; Hull (Options, Futures); Kemna-Vorst 1990; Kamal-Derman 1999; Leland 1985; Wilmott; CME weather specs; Alaton et al. 2002; Jewson-Brix 2005; GHCN-Daily docs.
