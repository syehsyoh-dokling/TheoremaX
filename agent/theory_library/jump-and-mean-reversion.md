family: jump-and-mean-reversion
category: derivatives-pricing
keywords: merton, jump-diffusion, fft, characteristic function, severity, claim count, loss distribution, panjer, jump intensity, compensator, kou, compound poisson, aggregate loss, ornstein-uhlenbeck, mean reversion, half-life, ar(1), exact discretisation, stationary distribution, commodity, schwartz, geometric mean-reverting, first passage time, running maximum, running minimum, reflection principle, inverse gaussian, hitting probability, euler, discretisation bias, funding rate, perpetual, basis carry, adf test
---
## Worked frame
DIKETAHUI: series [T+1] (native units), step dt (years), or model parameters (per year); barrier (price), horizons [n_h]; seed. Check: sorted dates, changes = levels - 1.
DITANYAKAN: parameters (scalars; half-life unit stated), moments [n_h], price, hit probabilities [n_h], MC estimate.
RUMUS: card Core 6-8 (OU), 1-3 (Merton), 10-12 (barrier); toolkit ar1_fit, ou_from_ar1, barrier_hit_prob, gbm_paths, mc_hit_probability.
JAWAB: 1) pairs x_t, x_t+1, each [T]. 2) AR(1) -> parameters. 3) residuals [T] -> jump flags. 4) moments or probabilities [n_h]. 5) paths [n_paths x n_steps+1] -> running extreme [n_paths] -> [n_h]. 6) per step -> per year. 7) check: 0 < b < 1, probabilities in [0,1], rising with horizon; then write.

## Core formulas
1. Merton: dS/S = (r-q-lambda kbar)dt + sigma dW + (J-1)dN, ln J ~ N(mu_J, sigma_J^2), kbar = exp(mu_J + sigma_J^2/2) - 1.
2. Merton series (exact): C = sum_n e^{-L T}(L T)^n/n! BS(S,K,r_n,sigma_n,T), L = lambda(1+kbar), r_n = r - lambda kbar + n ln(1+kbar)/T, sigma_n^2 = sigma^2 + n sigma_J^2/T; r_n used in d1 AND strike discount. Equivalent: lambda-weights with BS(S_n,K,r,sigma_n), S_n = S (1+kbar)^n e^{-lambda kbar T}. If a statement writes lambda-weights with r_n, follow it literally.
3. Merton log-return density over dt: sum_n Pois(n; lambda dt) N(x; (mu - sigma^2/2 - lambda kbar)dt + n mu_J, sigma^2 dt + n sigma_J^2). Annual variance sigma^2 + lambda(mu_J^2 + sigma_J^2).
4. Kou: log-jump density p e1 e^{-e1 y} (y>0) + (1-p) e2 e^{e2 y} (y<0), e1 > 1; kbar = p e1/(e1-1) + (1-p) e2/(e2+1) - 1.
5. Compound Poisson S = sum_{i<=N} X_i: E = lambda E[X], Var = lambda E[X^2], CF exp(lambda(phi_X(u) - 1)). FFT: discretise severity (rounding: f_0 = F(h/2), f_k = F((k+.5)h) - F((k-.5)h)), g = Re ifft(exp(lambda(fft(f) - 1))); span n h must exceed the aggregate tail (else wrap-around aliasing).
6. OU dX = kappa(theta - X)dt + sigma dW, exact: X_{t+dt} = theta + (X_t - theta)e^{-kappa dt} + sigma sqrt((1-e^{-2kappa dt})/(2kappa)) Z. AR(1) X_{t+1} = a + b X_t + e: kappa = -ln b/dt, theta = a/(1-b), sigma^2 = Var(e) 2kappa/(1-b^2). Euler map (only if told): kappa = (1-b)/dt, sigma = sd(e)/sqrt(dt).
7. OU moments, e = e^{-kappa tau}: E = theta + (x0-theta)e; Var = sigma^2(1-e^2)/(2kappa); stationary theta, sigma^2/(2kappa). Half-life ln2/kappa (years; x252 trading days).
8. OU + additive jumps J ~ N(mu_J, sigma_J^2): E = theta + (x0-theta)e + lambda mu_J(1-e)/kappa; Var = (sigma^2 + lambda(mu_J^2+sigma_J^2))(1-e^2)/(2kappa); stationary mean theta + lambda mu_J/kappa. Short-horizon (non-reverting) approx adds lambda mu_J tau and lambda E[J^2] tau; use only if the text says "add".
9. Geometric/Schwartz: X = ln S OU; lognormal E[S] = exp(m + v/2), Var[S] = (e^v - 1)exp(2m + v). Schwartz 1997: dS = kappa(mu - ln S)S dt + sigma S dW -> theta_X = mu - sigma^2/(2kappa); futures ln F(T) = e^{-kappa T} ln S + (1-e^{-kappa T})a* + sigma^2(1-e^{-2kappa T})/(4kappa), a* = theta_X - lambda_risk/kappa (lambda_risk=0 if silent).
10. X_t = mu t + sigma W (X_0=0), b>0, a<0, s = sigma sqrt T: P(M_T < b) = N((b - mu T)/s) - e^{2mu b/sigma^2} N((-b - mu T)/s); P(m_T > a) = N((-a + mu T)/s) - e^{2mu a/sigma^2} N((a + mu T)/s). Hit = 1 - these. mu=0: P(M_T >= b) = 2N(-b/s).
11. First passage tau_b: P(tau_b <= t) = P(M_t >= b); density b/(sigma sqrt(2 pi t^3)) exp(-(b-mu t)^2/(2 sigma^2 t)) = IG(mean b/mu, shape b^2/sigma^2); E[tau_b] = b/mu (mu>0), infinite if mu<=0; P(ever hit) = e^{2 mu b/sigma^2} if mu<0.
12. Joint, x <= b: P(X_T <= x, M_T <= b) = N((x - mu T)/s) - e^{2mu b/sigma^2} N((x - 2b - mu T)/s).
## Conventions and defaults when the task is silent
- Barrier given as a PRICE level H on GBM: use X = ln(S/S0) and b = ln(H/S0) in items 10-12 (never H or H - S0). H > S0 -> up barrier, running MAX, b > 0; H < S0 -> down barrier, running MIN, a = b < 0. MC must compare the running max/min of the same variable with the same level.
- Daily data: dt = 1/252, parameters annualised. GBM log-drift under Q: mu = r - q - sigma^2/2; sigma = std(log ret, ddof=1) sqrt(252).
- n = number of changes = len(levels) - 1. Descriptives ddof=1; skew/kurt as pandas (bias-corrected, EXCESS kurtosis).
- Keep native units unless told to convert (yields 4.25 stay percent; kappa invariant, theta/sigma scale). "Convert to decimal" -> /100 before log.
- Jumps from AR(1) residuals, |e| > 3 std(e, ddof=1); lambda = n_jumps/(n_resid dt); mu_J, sigma_J = mean/std(ddof=1) of flagged residuals; re-estimate sigma on non-jump residuals only if stated.
- MLE for Merton: log-sum-exp over n, L-BFGS-B, several deterministic starts, keep max LL; physical drift fixed if stated.
- MC seeds: np.random.seed(k) (legacy) differs from default_rng(k); use the one named.
- VaR_a = smallest x with CDF >= a (samples: sorted[ceil(a n)-1]); ES = E[S | S > VaR].
## Completing missing inputs
- S0/x0 = last observation; kappa from half-life: kappa = ln2/HL.
- Pareto (x_m, a): E = a x_m/(a-1), E[X^2] = a x_m^2/(a-2); numpy pareto is Lomax: (rng.pareto(a)+1) x_m. Lognormal E = e^{mu+s^2/2}, E[X^2] = e^{2mu+2s^2}. scipy lognorm(s, scale=e^mu).
- MC jumps per step: N ~ Poisson(lambda dt); jump sum = N mu_J + sqrt(N) sigma_J Z.
## Invariants to self-check
- 0 < b < 1 for mean reversion; theta near sample mean; Var increasing in tau to stationary limit.
- Merton price >= BS lower bound, increasing in T, decreasing in K; BS(iv) reprices; negative mu_J -> downward skew.
- Hit probabilities in [0,1], decreasing in |barrier distance|, increasing in T; FPT CDF non-decreasing; joint <= each marginal.
- FFT pmf sums to 1, mean ~ lambda E[X].
## Variants to decide
- V1 ou_discretisation: AR(1) to OU mapping. Options: (a) exact kappa=-ln(b)/dt, sigma from exact variance; (b) Euler kappa=(1-b)/dt. Default when silent: (a) - Vasicek 1977 / Brigo-Mercurio 2006.
- V2 dt_basis: time step for daily data. Options: (a) 1/252; (b) 1/365. Default when silent: (a) - Hull.
- V3 kurtosis_def: kurtosis reported. Options: (a) excess, bias-corrected (pandas); (b) scipy biased excess; (c) raw Pearson. Default when silent: (a) - Joanes-Gill 1998 (G2).
- V4 var_quantile: VaR on simulated losses. Options: (a) sorted[ceil(a n)-1] (smallest x, CDF >= a); (b) np.quantile linear. Default when silent: (a) - Hyndman-Fan type 1.
- V5 rng_api: random generator. Options: (a) np.random.default_rng(seed); (b) legacy np.random.seed. Default when silent: (a) - NumPy NEP 19 (named API always wins).
- V6 gamma_param: gamma severity parameters. Options: (a) shape k and SCALE theta, mean k*theta; (b) shape k and RATE beta, mean k/beta. Default when silent: (a) - Klugman-Panjer-Willmot Loss Models.
- V7 half_life_unit: half-life reporting unit. Options: (a) unit the field names (periods = ln2/(kappa dt), days = periods / periods-per-day); (b) ln2/kappa in dt's unit. Default when silent: (a) - dimensional consistency with dt.
- V8 date_end_inclusive: date-only end bound on timestamped data. Options: (a) include the whole end day (< end + 1 day); (b) <= end at midnight. Default when silent: (a) - calendar-date convention.
- V9 mc_accrual_start: simulated carry accrual. Options: (a) accrue from step 1 (f_0 is the known start); (b) include f_0. Default when silent: (a) - Glasserman 2003 path convention.

## Pitfalls
- Euler OU inversion or raw residual std as sigma when "exact"/"conditional distribution" is implied.
- sigma_n^2 without /T; missing compensator; sigma_J vs sigma_J^2.
- Discrete daily MC under-estimates continuous hit probabilities (BGK: shift barrier by 0.5826 sigma sqrt(dt)); include the t=0 column in running max/min; slice paths to round(T/dt) steps per horizon.
- MC E[tau] averages only finite hitting times unless told otherwise.
- Sign of e^{2mu a/sigma^2} with a < 0; reporting P(no hit) as hit prob.
- Euler with Bernoulli(lambda dt) jump vs Poisson count: equal only for small lambda dt.
- Analytic Var(X_tau) without the jump term lambda(mu_J^2 + sigma_J^2)(1 - e^{-2 kappa tau})/(2 kappa) while the MC simulates jumps: the MC check fails (ou_jump_moments).
- "|residual| > 3 std(residuals)" is ONE pass on the full AR(1) fit (ou_jump_threshold_fit); an iterated filter with refits (fit_ou_jump) flips mu_J.
- Running max/min MC must compare LOG paths ln(S/S0) with b = ln(H/S0), never price paths (gbm_first_passage_report).
- Stationary moments are the tau -> infinity limits (ou_jump_stationary); tau = 0 is the start (variance 0). Never invent an MC tolerance the task does not state.
## Sources
Merton 1976; Hull (Options, Futures) ch. jump-diffusion; Kou 2002; Klugman-Panjer-Willmot (Loss Models); Uhlenbeck-Ornstein 1930; Schwartz 1997; Shreve II ch. 3, 7; Borodin-Salminen 2002; Glasserman 2003; Broadie-Glasserman-Kou 1997.
