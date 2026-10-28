family: copulas-and-dependence
category: risk-management
keywords: copula, gaussian copula, student-t copula, clayton, gumbel, frank, archimedean, kendall tau, spearman rho, rank correlation, pseudo-observations, empirical cdf, tail dependence, upper tail, lower tail, tau inversion, degrees of freedom, profile likelihood, aic, bic, sampling, joint default, dependence structure, normal variance mixture, chi-squared mixing, cholesky
---
## Worked frame
DIKETAHUI: prices or returns [T x N] (decimal), or family, parameter, n, seed. Check: sorted dates; drop NaN rows jointly so all columns share [T].
DITANYAKAN: one row per pair x family (parameter, loglik, AIC, BIC, tail dependence in [0,1]); tau matrix [N x N]; samples [n x d].
RUMUS: card Core 1 -> 2 -> 3-4 -> 5-6 -> 7 -> 8; toolkit pseudo_observations, empirical_kendall_tau, fit_copula, tail_dependence, copula_sample.
JAWAB: 1) returns [T x N]. 2) pseudo-observations [T x N]. 3) per pair [T x 2] -> tau. 4) fit each family -> one row per pair x family. 5) tail dependence per row. 6) samples [n x d]; uniforms -> return units if asked. 7) check: u in (0,1), rows = pairs x families, BIC - AIC = k(ln n - 2); then write.

## Core formulas
1. Pseudo-observations: u_i = rank_i/(n+1) (average ranks for ties; scipy rankdata / pandas rank). Keeps u in (0,1) so ppf never returns inf.
2. Kendall tau = (concordant - discordant)/C(n,2) (tau-a); scipy.stats.kendalltau and pandas corr('kendall') give tau-b (tie-adjusted; equal to tau-a for continuous data). Spearman rho_S = Pearson on ranks = 1 - 6 sum d^2/(n(n^2-1)) without ties. Rank statistics are identical on raw data and on pseudo-obs.
3. Elliptical (Gaussian, t): tau = (2/pi) arcsin(rho) -> rho = sin(pi tau/2). Gaussian only: rho_S = (6/pi) arcsin(rho/2) -> rho = 2 sin(pi rho_S/6).
4. Clayton: tau = theta/(theta+2) -> theta = 2tau/(1-tau); lambda_L = 2^(-1/theta), lambda_U = 0.
   Gumbel: tau = 1 - 1/theta -> theta = 1/(1-tau) (theta >= 1); lambda_U = 2 - 2^(1/theta), lambda_L = 0.
   Frank: tau = 1 - (4/theta)(1 - D1(theta)), D1(x) = (1/x) int_0^x t/(e^t-1) dt; solve by brentq; no tail dependence.
   t: lambda_U = lambda_L = 2 t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho))). Gaussian: 0 for |rho| < 1.
5. Log densities (x = Phi^-1(u), y = Phi^-1(v)):
   Gaussian: -0.5 ln(1-rho^2) - (rho^2(x^2+y^2) - 2 rho x y)/(2(1-rho^2)).
   t (x = t_nu^-1(u), ...): ln f2(x,y) - ln f1(x) - ln f1(y); ln f2 = gammaln((nu+2)/2) - gammaln(nu/2) - ln(nu pi) - 0.5 ln(1-rho^2) - ((nu+2)/2) ln(1 + (x^2 - 2rho x y + y^2)/(nu(1-rho^2))); ln f1(x) = gammaln((nu+1)/2) - gammaln(nu/2) - 0.5 ln(nu pi) - ((nu+1)/2) ln(1 + x^2/nu).
   Clayton: ln(1+theta) - (1+theta)(ln u + ln v) - (2 + 1/theta) ln(u^-theta + v^-theta - 1).
   Gumbel (a = -ln u, b = -ln v, S = a^theta + b^theta): -S^(1/theta) + a + b + (theta-1) ln(a b) + (1/theta - 2) ln S + ln(S^(1/theta) + theta - 1).
   Frank: ln(theta(1-e^-theta)) - theta(u+v) - 2 ln((1-e^-theta) - (1-e^(-theta u))(1-e^(-theta v))).
6. AIC = -2LL + 2k; BIC = -2LL + k ln n; k = 1 (Gaussian, Clayton, Gumbel, Frank), 2 (t: rho, nu).
7. Empirical tail dependence at q: lambda_U(q) = #(u>q & v>q)/#(u>q); lambda_L(q) = #(u<1-q & v<1-q)/#(u<1-q). Alternative denominators n(1-q) (cue: "divided by n(1-q)" or C_n-based formula).
8. Sampling: Gaussian U = Phi(L eps), L = chol(R). t: Z as Gaussian, W ~ chi2(nu) ONE per row shared by all coordinates, T = Z/sqrt(W/nu), U = t_nu.cdf(T). Clayton (conditional): U2 = (U1^-theta (W^(-theta/(1+theta)) - 1) + 1)^(-1/theta); or frailty V ~ Gamma(1/theta,1), U_i = (1 + E_i/V)^(-1/theta). Frank (conditional): U2 = -(1/theta) ln(1 + W(e^-theta - 1)/(W + (1-W) e^(-theta U1))). Gumbel (Marshall-Olkin): positive stable S with index 1/theta, U_i = exp(-(E_i/S)^(1/theta)). Then X = F_i^-1(U).

## Conventions and defaults when the task is silent
- Fitting: tau inversion (moment method) when "invert tau" is said; otherwise canonical ML on pseudo-obs (CML); IFM = fit margins parametric then copula ML. t copula: rho by tau inversion, nu by profile likelihood with nu > 2 (bounded 1-D search, e.g. (2, 50]).
- tau <= 0 with Clayton/Gumbel: parameter at independence (theta = 0 / 1, LL = 0) rather than NaN.
- Pairs in the order the text lists the assets (itertools.combinations), key "A-B"; not alphabetical from a pivot.
- Returns: log returns from close, rows with any NaN dropped JOINTLY so all series share dates.
- One generator created once, families sampled in the stated order; use the stated API (default_rng vs RandomState).
- Tail counts use strict inequalities and condition on the FIRST variable.

## Completing missing inputs
- Multivariate R from pairwise tau may be non-PSD: clip eigenvalues and rescale to unit diagonal (nearest correlation).
- Typical equity pair tau 0.2-0.5; daily equity t-copula nu ~ 4-10.
- Only margins + copula given: simulate U, then X_i = F_i^-1(U_i); Pearson of X must be computed, not assumed equal to rho.

## Invariants to self-check
- u, v strictly in (0,1); margins of samples uniform (mean 0.5, var 1/12).
- Empirical tau of each sampled family within ~0.01 of target at N ~ 1e5.
- At equal tau: Spearman differs across families (not a bug); Gumbel and t strongest upper tail, Clayton and t strongest lower; Gaussian and Frank near 0 as q -> 1 (finite q gives positive values).
- BIC - AIC = k(ln n - 2); t LL -> Gaussian LL as nu -> infinity; LL(independence) = 0.
- 0 <= lambda <= 1; rows = C(m,2) x families.

## Variants to decide
- V1 pseudo_obs: rank scaling to (0,1). Options: (a) R/(n+1); (b) R/n; (c) (R-0.5)/n. Default when silent: (a) - Genest-Ghoudi-Rivest 1995.
- V2 fit_method: copula parameter estimation. Options: (a) canonical ML on pseudo-obs; (b) Kendall tau inversion; (c) IFM (parametric margins then copula ML). Default when silent: (a) - Genest-Ghoudi-Rivest 1995.
- V3 kendall_variant: Kendall tau with ties. Options: (a) tau-b (scipy kendalltau default); (b) tau-a. Default when silent: (a) - Kendall 1945 / scipy.
- V4 t_copula_nu: t-copula dof estimation. Options: (a) rho by tau inversion, nu by profile likelihood on (2, 50]; (b) joint ML of rho and nu. Default when silent: (a) - Demarta-McNeil 2005.
- V5 rng_api: random generator. Options: (a) np.random.default_rng(seed); (b) legacy np.random.seed / RandomState. Default when silent: (a) - NumPy NEP 19 (named API always wins).
- V6 t_mixing_scope: what the chi2 mixing variable scales in a factor model. Options: (a) systematic factors and idiosyncratic shocks together; (b) factors only. Default when silent: (a) - Demarta-McNeil 2005.
- V7 t_threshold: event threshold under the t copula. Options: (a) t_nu quantile of the marginal probability; (b) normal quantile. Default when silent: (a) - McNeil-Frey-Embrechts 2015 (marginals preserved).

## Pitfalls
- ranks/n (u = 1 -> infinite quantile).
- Using Pearson for the inversion; using rho = sin(pi tau/2) for Spearman.
- Independent chi2 per coordinate in the t sampler (gives a different, non-t copula).
- ln(gammaln) instead of gammaln; overflow in Clayton u^-theta for large theta (work in logs, clip u to [1e-12, 1-1e-12]).
- scipy levy_stable scale/parameterisation: margins become non-uniform unless the scale is (cos(pi/(2 theta)))^theta (S1 parameterisation).
- Resetting the seed per family when "set once" is stated.
- Linear correlation is not invariant to monotone transforms and has attainable bounds < 1 for non-elliptical margins; do not treat Pearson as a copula parameter.
- Theoretical tail_dependence(family, theta) returns (lower, upper) limits, not the empirical lambda at q a task asks for; on a sample use empirical_tail_table(u, levels) (named keys) and empirical_rank_correlations(u).

- 'Most common best copula' is the mode of the per-pair AIC winners, not a count over the fit table where every family appears once per pair (pairwise_copula_table).
- A long (date, symbol, close) panel is deduplicated on (date, symbol), never on date alone - that deletes the panel (equity_pair_copula_report pivots it); calibrate-sample-measure studies: copula_family_study (one shared generator, rank correlations from the sample).
## Sources
Nelsen, An Introduction to Copulas (2006); Joe (1997, 2014); McNeil-Frey-Embrechts (2015) ch.7; Embrechts-McNeil-Straumann (2002); Genest-Ghoudi-Rivest (1995) pseudo-likelihood; Marshall-Olkin (1988); Demarta-McNeil (2005) t copula; Li (2000) Gaussian copula.
