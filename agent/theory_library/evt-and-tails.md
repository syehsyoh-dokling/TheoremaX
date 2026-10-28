family: evt-and-tails
category: risk-management
keywords: extreme value theory, evt, peaks over threshold, pot, generalized pareto, gpd, genpareto, block maxima, gev, hill estimator, tail index, smith, mean excess, exceedances, garch-evt, heavy tail, compound poisson, aggregate loss, panjer, fft, characteristic function, discretization, severity, frequency, return level, standardised losses, threshold quantile, tail approximation, weissman, gpd tail var, operational risk
---
## Worked frame
DIKETAHUI: returns [n] (decimal) -> losses = -returns; threshold rule; levels [Q]. Aggregate branch: frequency, severity parameters (currency), step h, size n_grid.
DITANYAKAN: tail parameters, threshold (scalars); VaR and ES per level [Q] (positive loss); aggregate pmf [n_grid].
RUMUS: card Core 1-3 (POT), 6 (Hill), 9-12 (aggregate); toolkit suggest_lattice, discretize_severity, compound_poisson_fft, aggregate_var_es.
JAWAB: 1) losses [n], sorted. 2) threshold scalar, excesses [N_u]. 3) fit -> xi, beta. 4) VaR, ES [Q]. 5) aggregate: severity pmf [n_grid] -> aggregate pmf [n_grid]. 6) index x h -> currency; fraction x value if money asked. 7) check: ES > VaR > threshold, beta > 0, pmf sums to 1; then write.

## Core formulas
1. Losses L = -r (positive = down move). Threshold u; exceedances L > u (strict); excesses y = L - u; N_u = count, n = sample size.
2. GPD: G(y) = 1 - (1 + xi y/beta)^(-1/xi) (xi=0: 1 - exp(-y/beta)). scipy: genpareto.fit(y, floc=0) -> (c=xi, loc=0, scale=beta); c has the SAME sign as xi.
3. POT tail: P(L > x) ~ (N_u/n)(1 + xi (x-u)/beta)^(-1/xi). VaR_q = u + (beta/xi)[((n/N_u)(1-q))^(-xi) - 1]; ES_q = VaR_q/(1-xi) + (beta - xi u)/(1-xi), needs xi < 1. xi=0: VaR = u - beta ln((n/N_u)(1-q)), ES = VaR + beta.
4. Mean excess e(u) = E[L-u | L>u] = (beta + xi u)/(1-xi), linear in u above a valid threshold (choose u where the empirical plot turns linear).
5. GPD NLL: N ln beta + (1+1/xi) sum ln(1 + xi y/beta); infeasible if beta <= 0 or any 1 + xi y/beta <= 0. Expected-info SEs (xi > -0.5): Var(xi) = (1+xi)^2/N_u, Var(beta) = 2 beta^2 (1+xi)/N_u. Observed info: Cov = inv(-Hessian of LL) (invert -H, not H).
6. Hill (sorted descending X_(1) >= ...): xi_H = (1/k) sum_{i=1..k} ln X_(i) - ln X_(k+1); tail exponent alpha = 1/xi; se = xi_H/sqrt(k); Weissman quantile VaR_q = X_(k+1) (k/(n(1-q)))^xi_H.
7. GEV block maxima: H(x) = exp(-(1 + xi (x-mu)/sigma)^(-1/xi)). scipy genextreme c = -xi (OPPOSITE sign). m-block return level: mu - (sigma/xi)[1 - (-ln(1-1/m))^(-xi)]. Daily from block: F_daily(x) = H(x)^(1/blocksize).
8. GARCH-EVT (McNeil-Frey 2000): fit GARCH, z = r/sigma_t, standardised losses -z, POT on them (own threshold), VaR = sigma_{T+1} VaR_z (+ mu term if a mean model is used), ES likewise.
9. Compound Poisson S = sum_{i<=N} X_i: E[S] = lambda E[X]; Var[S] = lambda E[X^2]; PGF_N(z) = exp(lambda(z-1)).
10. Panjer (Poisson, a=0, b=lambda): g_0 = exp(-lambda(1-f_0)); g_s = (lambda/s) sum_{j=1..s} j f_j g_{s-j}. Neg-bin: a = beta/(1+beta), b = (r-1)a, g_0 = PGF_N(f_0).
11. FFT: f = discretised severity (length n, pad with zeros); g = real(ifft(PGF_N(fft(f)))) = real(ifft(exp(lambda(fft(f)-1)))); clip tiny negatives; CDF = cumsum(g).
12. Discretisation by rounding (midpoint): f_0 = F(h/2), f_k = F((k+1/2)h) - F((k-1/2)h); last node gets the remaining tail mass (or drop it). Alternatives: lower/upper (forward/backward differences), mean-preserving local moment matching (cue: "preserve the mean").

## Conventions and defaults when the task is silent
- Threshold: a high empirical quantile of losses (90-95%) or the top k order statistics; Hill and GPD at "k" both use X_(k+1) as u (k tail points). Cue "X_(k) as threshold" -> k-1 excesses.
- Excesses of exactly 0 (ties at u) are excluded when exceedance is strict.
- Discrete VaR_a = smallest x with F(x) >= a (grid: first index where cumsum >= a; sample: sorted[ceil(a n)-1]).
- ES for a lattice/sample: E[S | S > VaR] (strict) when the text writes ">"; proper (Acerbi-Tasche) ES = (1/(1-a))[E[S 1{S>VaR}] + VaR (F(VaR) - a)].
- Grid span n h must exceed the aggregate 99.99% quantile, else FFT wraps tail mass onto small losses (aliasing). Default: span ~ 2x max(MC 99.99% quantile, mean + 10 sd).
- Severity parameters: scipy lognorm(s=sigma, scale=exp(mu)); gamma(a=shape, scale=beta); pareto(b=alpha, scale=x_m). numpy rng.pareto is Lomax: classical Pareto = (rng.pareto(a)+1) x_m.
- MC aggregate: draw all counts first (rng.poisson(lambda, N)), then sum(counts) severities in one call, aggregate by np.repeat index + bincount; a pinned seed/order in the text overrides.

## Completing missing inputs
- Severity moments: gamma a*theta, a(a+1)theta^2; lognormal exp(mu + s^2/2), exp(2mu + 2s^2); Pareto alpha x_m/(alpha-1), alpha x_m^2/(alpha-2) (alpha > 2, else infinite variance).
- Starting values for GPD MLE: method of moments xi0 = 0.5(1 - m^2/v), beta0 = 0.5 m (1 + m^2/v) (m, v = mean, var of excesses); multi-start Nelder-Mead guards against bad local optima.
- Heavy tail: xi > 0 (Frechet domain); xi = 0 Gumbel domain (normal, lognormal); xi < 0 bounded tail. Daily equity losses: xi ~ 0.1-0.4.

## Invariants to self-check
- ES > VaR; VaR, ES increasing in q; EVT VaR near the empirical quantile at q just above the threshold level.
- u < VaR_q for q beyond the threshold probability; beta > 0; xi < 1 for finite ES.
- Hill CI symmetric; se shrinks with k. Hill positive only if losses above X_(k+1) are > 0.
- Aggregate pmf sums to 1 (1e-9); mean of g ~ lambda E[X]; g_0 = exp(-lambda(1-f_0)).
- FFT vs MC VaR/ES within ~1-2% for light tails; heavier tails need wider span.

## Variants to decide
- V1 threshold_k: threshold for k tail points. Options: (a) u=X_(k+1), k excesses; (b) u=X_(k), k-1 excesses. Default when silent: (a) - Hill 1975.
- V2 exceedance: tie handling at u. Options: (a) strict X > u; (b) X >= u. Default when silent: (a) - McNeil-Frey-Embrechts 2005.
- V3 gpd_fit: GPD estimation. Options: (a) MLE on excesses with loc fixed at 0 (genpareto floc=0); (b) PWM; (c) free location. Default when silent: (a) - Smith 1987 / McNeil-Frey 2000; (b) Hosking-Wallis 1987.
- V4 discrete_es: ES on a lattice/sample. Options: (a) E[S|S>VaR] strict; (b) Acerbi-Tasche with atom correction; (c) E[S|S>=VaR]. Default when silent: (b) - Acerbi-Tasche 2002; (a) when the text writes ">".
- V5 severity_discretisation: severity on grid h. Options: (a) rounding/mid-point F((j+.5)h)-F((j-.5)h); (b) lower or upper; (c) local moment matching. Default when silent: (a) - Klugman-Panjer-Willmot.
- V6 sample_quantile: empirical VaR. Options: (a) inverted CDF sorted[ceil(a n)-1]; (b) np.quantile linear. Default when silent: (a) - Hyndman-Fan type 1 (smallest x with F(x) >= a).
- V7 garch_evt_sigma: volatility that rescales the standardised tail. Options: (a) one-step forecast sigma_{T+1}; (b) last in-sample sigma_T. Default when silent: (a) - McNeil-Frey 2000; (b) when the text says last in-sample conditional vol.
- V8 standardise_units: residual filter units. Options: (a) decimal r / decimal sigma; (b) percent r / percent sigma (identical z). Default when silent: (a) - McNeil-Frey 2000; never decimal over percent.
- V9 hill_k: Hill sample fraction. Options: (a) stated k (e.g. floor of sqrt n); (b) k = number of POT exceedances. Default when silent: (b) - Embrechts-Kluppelberg-Mikosch 1997.
- V10 hist_es_tail: empirical ES set. Options: (a) losses >= VaR; (b) losses > VaR. Default when silent: (a) - Dowd 2005.

## Pitfalls
- genpareto.fit without floc=0 (location estimated too); reading params in the wrong order (c, loc, scale).
- Sign flip of genextreme c.
- Fitting GPD to raw losses instead of excesses; using returns instead of losses.
- Hill with k+1 points or threshold X_(k); ln of non-positive order statistics.
- GARCH-EVT: forgetting to rescale by sigma_{T+1}; mixing percent and decimal GARCH output.
- FFT without padding / too small span (aliasing); using continuous pdf*h instead of CDF differences; forgetting real() and renormalisation.
- Quantile rule: follow a pinned method (e.g. method='lower') in EVERY quantile call, including thresholds and rolling windows.
- Percentile (0-100) vs quantile (0-1) APIs.
- Rolling GARCH-EVT backtests refit about a thousand GARCH models: a serial loop times out, use rolling_tail_backtest (process pool). Level keys drop the leading '0.' (level_key), never int(p*1000).
- Coverage-test p-values are chi2.sf(LR, df): df 1 for Kupiec and independence, 2 for conditional coverage; exp(-LR/2) is only the df-2 survival.

## Sources
McNeil-Frey-Embrechts (2015) ch.5; Embrechts-Kluppelberg-Mikosch (1997); Smith (1987); Hill (1975); Weissman (1978); McNeil-Frey (2000); Coles (2001); Klugman-Panjer-Willmot, Loss Models; Panjer (1981); Embrechts-Frei (2009) FFT aggregate losses.
