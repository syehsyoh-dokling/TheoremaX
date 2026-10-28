method: heston-characteristic-function
family: local-and-stochastic-vol
keywords: heston characteristic function, heston pricing, two-factor heston, double heston, chiarella-ziveyi, christoffersen-heston-jacobs, little trap, gil-pelaez, p1 p2 integrals, gauss-legendre quadrature, semi-analytical option pricing, fourier inversion
---
## Formulas
1. Per variance factor j (kappa_j, theta_j, sigma_j, rho_j, v_j0), u real or complex: b_j = kappa_j - rho_j sigma_j i u, d_j = sqrt(b_j^2 + sigma_j^2 (i u + u^2)), g_j = (b_j - d_j)/(b_j + d_j), e_j = exp(-d_j T); D_j = (b_j - d_j)/sigma_j^2 (1 - e_j)/(1 - g_j e_j); C_j = kappa_j theta_j/sigma_j^2 [(b_j - d_j) T - 2 ln((1 - g_j e_j)/(1 - g_j))].
2. phi(u) = exp(i u (ln S0 + (r - q) T) + sum_j (C_j + D_j v_j0)) - independent factors multiply.
3. P2 = 1/2 + (1/pi) int_0^U Re[e^{-i u ln K} phi(u)/(i u)] du; P1 the same with phi(u - i)/phi(-i).
4. Call = S0 e^{-qT} P1 - K e^{-rT} P2; Put direct = K e^{-rT}(1 - P2) - S0 e^{-qT}(1 - P1).
5. Gauss-Legendre on [0, U]: u = (x + 1) U/2, w = U w_x/2; vectorise as a (n_K x n_nodes) matrix times weights.
## Variants to decide
- V1 cf_form: CF branch. Options: (a) little-trap form; (b) original Heston 1993. Default when silent: (a) - Albrecher et al. 2007.
- V2 quadrature: integral. Options: (a) Gauss-Legendre with many nodes on a wide [0, U]; (b) adaptive quad from a tiny lower limit. Default when silent: (a) - Kahl-Jaeckel 2005.
- V3 market_price_of_vol_risk: lambda. Options: (a) 0, risk-neutral kappa, theta; (b) kappa + lambda, kappa theta/(kappa + lambda). Default when silent: (a) - Heston 1993 with lambda = 0.
## Invariants
- phi(0) = 1; phi(-i) = S0 e^{(r-q)T}; direct put minus parity put ~ machine precision.
- IV decreasing in K for negative rho; price within [max(F - K, 0) e^{-rT}, S0 e^{-qT}].
## Pitfalls
- Short T, deep OTM: adaptive quad truncated at a small U leaves large errors.
- quad needs a scalar integrand: loop strikes, do not pass arrays.
## Sources
- Heston (1993); Christoffersen, Heston, Jacobs (2009); Chiarella-Ziveyi (2009); Albrecher et al. (2007).
