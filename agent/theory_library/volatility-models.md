family: volatility-models
category: risk-management
keywords: ewma, riskmetrics, decay factor, lambda, garch, gjr, arch package, conditional volatility, persistence, long-run variance, variance forecast, dcc, dynamic conditional correlation, realized variance, realized volatility, bipower variation, microstructure noise, subsampling, parkinson, garman-klass, rogers-satchell, yang-zhang, ohlc, annualized volatility, vol of vol, two-step estimation, quasi-maximum likelihood, standardised residuals, percentage returns, rescale, conditional covariance, garch var
---
## Worked frame
DIKETAHUI: returns [T] or [T x N] (decimal, log), OHLC [T x 4], intraday prices [n] per day, model parameters. Check: sorted unique timestamps, no forward-filled bars.
DITANYAKAN: variance or vol [T] (daily or annualised: say which), covariance [N x N], parameters, forecasts [h], realised variance [D].
RUMUS: card Core 1 (EWMA), 2-4 (GARCH), 5 (DCC), 6-8 (realised, range), 9; toolkit ewma_variance, ewma_vol, annualized_vol.
JAWAB: 1) returns [T] from prices [T+1]. 2) seed, recursion -> variance [T] or [N x N]. 3) forecast scalar; path [h]. 4) realised or range per day [D]. 5) percent-fit parameters -> decimal. 6) annualise variance, then sqrt. 7) check: variance > 0, persistence < 1, matrix symmetric PSD; then write.

## Core formulas
1. EWMA (RiskMetrics 1996, zero mean, NO demeaning): S_t = lambda S_{t-1} + (1-lambda) r_t r_t' (np.outer, not np.dot). S_t is the forecast for t+1. Finite-window form: weights (1-lambda) lambda^{i-1}/(1-lambda^n), i=1 newest.
2. GARCH(1,1): s2_t = omega + alpha e_{t-1}^2 + beta s2_{t-1}. Persistence P = alpha+beta < 1; V_L = omega/(1-P); half-life = ln 0.5/ln P. One-step forecast s2_{T+1} = omega + alpha e_T^2 + beta s2_T. h-step: E[s2_{T+h}] = V_L + P^{h-1}(s2_{T+1} - V_L); h-day variance = sum_{k=1..h}.
3. GJR: + gamma e_{t-1}^2 1{e_{t-1}<0}; persistence alpha + beta + gamma/2 (symmetric innovations).
4. Gaussian QML: LL = -0.5 sum [ln 2pi + ln s2_t + e_t^2/s2_t].
5. DCC (Engle 2002): z_t = e_t/sigma_t; Qbar = Z'Z/T; Q_1 = Qbar; Q_t = (1-a-b)Qbar + a z_{t-1}z_{t-1}' + b Q_{t-1}; R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}; H_t = D_t R_t D_t, D = diag(sigma). Stage-2 LL = -0.5 sum_t [ln|R_t| + z_t' R_t^{-1} z_t - z_t'z_t]. Constraints a,b > 0, a+b < 1. Q forecast: Q_{T+k} - Qbar decays by (a+b)^{k-1}.
6. Realised variance: RV = sum r_i^2 (intraday log returns, daily scale). BV = (pi/2) sum_{i=2..n} |r_i||r_{i-1}| (mu1 = sqrt(2/pi), BN-S 2004 Def.1); jump = max(RV-BV, 0).
7. Noise (iid, variance w2): E[RV_n] = IV + 2 n w2 -> w2_hat = RV/(2n) at the HIGHEST frequency (Bandi-Russell 2006; Zhang et al. 2005). Bias-corrected RV_f = RV_f - 2 n_f w2. TSRV = avg of K subgrid RVs - (nbar/n) RV_all, nbar = (n-K+1)/K.
8. OHLC (logs; o = ln O_t - ln C_{t-1}, u = ln H/O, d = ln L/O, c = ln C/O):
   Parkinson: mean((ln H/L)^2)/(4 ln 2).
   Garman-Klass: mean(0.5 (ln H/L)^2 - (2 ln2 - 1) c^2).
   Rogers-Satchell: mean(u(u-c) + d(d-c)) (drift-robust).
   Yang-Zhang: V_o + k V_c + (1-k) V_RS; V_o, V_c = sample variances of o and of open-to-close c; k = 0.34/(1.34 + (n+1)/(n-1)), n = days in window.
   Close-to-close: sample variance of log returns (or mean of squares if zero-mean stated).
9. Annualise VARIANCE by 252 (vol by sqrt(252)); crypto/calendar 365, weekly 52, monthly 12. Intraday per-minute vol * sqrt(390*252).

## Conventions and defaults when the task is silent
- EWMA lambda 0.94 daily, 0.97 monthly. Seed: sample covariance of an initial block, or of the full sample (both vanish after ~100 obs at 0.94). Cue "seed with first observation" -> S_0 = outer(r_1, r_1) and recurse from row 2. Short samples: seed choice matters, follow text.
- "Current"/"as of last date" covariance INCLUDES the last return r_T. For a backtest, VaR for day t uses S through t-1 only.
- arch package: arch_model default mean is 'Constant' - pass mean='Zero' if the text says zero mean. Fit on 100*r (percent); map back: omega_dec = omega/1e4, sigma_dec = sigma/100, alpha/beta unchanged, LL_dec = LL_pct + n ln 100. Initial s2 = backcast (0.94-weighted mean of first min(75,n) squared residuals).
- res.conditional_volatility = in-sample sigma_t; the one-step forecast is res.forecast(horizon=1).variance.iloc[-1] (percent^2). "Final observation conditional vol" in pricing texts usually means the one-step-ahead forecast; the long-run vol only if stated.
- DCC: Qbar from z (covariance form Z'Z/T) unless "correlation matrix" is said; start (a,b) = (0.05, 0.93); L-BFGS-B with bounds and a penalty for a+b >= 1.
- RV sampling at f minutes: keep observations whose minute offset from the open is divisible by f (subset, no forward fill) unless "previous-tick" is said. Exclude overnight returns. Deduplicate timestamps as instructed (median / last) before filtering sessions.
- BV: no n/(n-1) factor unless "finite-sample correction" is said.

## Completing missing inputs
- No GARCH fit possible: typical daily equity alpha ~0.05-0.10, beta ~0.88-0.93; use V_L = sample variance for variance targeting (omega = V_L (1-alpha-beta)).
- Vol-of-vol from a fitted model: std (ddof=1) of first differences of the annualised conditional-vol series; that std is already a per-day change - do not rescale twice.

## Invariants to self-check
- S symmetric PSD, diag > 0; w'Sw > 0. R_t unit diagonal, |rho| <= 1.
- omega > 0, alpha, beta >= 0, alpha+beta < 1; recomputing s2_{T+1} from params reproduces the forecast.
- Forecasts converge monotonically to V_L as h grows.
- RV_1min >= RV_30min typically (noise signature); BV <= RV approx; w2 tiny (~1e-8 for liquid ETFs).
- Range estimators have lower sampling variance than close-to-close (efficiency > 1); per-day RS >= 0 when H >= max(O,C), L <= min(O,C).

## Variants to decide
- V1 ewma_seed: EWMA seed. Options: (a) sample covariance of an initial block; (b) first observation outer(r_1, r_1); (c) full sample. Default when silent: (a) - RiskMetrics 1996.
- V2 ewma_mean: EWMA mean. Options: (a) zero mean; (b) demeaned. Default when silent: (a) - RiskMetrics 1996.
- V3 garch_mean: GARCH mean. Options: (a) constant mean; (b) zero mean. Default when silent: (a) - Bollerslev 1986 / arch package.
- V4 rv_sampling: RV sampling. Options: (a) sparse subset every f minutes; (b) previous-tick. Default when silent: (a) - Andersen et al. 2001.
- V5 bv_scale: bipower variation. Options: (a) (pi/2) sum |r_t||r_{t-1}|; (b) with n/(n-1). Default when silent: (a) - Barndorff-Nielsen-Shephard 2004.
- V6 dcc_qbar: DCC Qbar. Options: (a) covariance Z'Z/T; (b) correlation. Default when silent: (a) - Engle 2002.
- V7 arch_fit_scale: returns given to arch. Options: (a) percent (x100), then omega/1e4 and sigma/100; (b) native with rescale=False. Default when silent: (a) - arch docs (Sheppard); (b) when the text forbids rescaling.
- V8 forecast_sigma: sigma reported as "current". Options: (a) one-step forecast sigma_{T+1}; (b) in-sample sigma_T. Default when silent: (a) - Christoffersen 2012; (b) when "last in-sample" is said.
- V9 dcc_loglik: reported DCC likelihood. Options: (a) correlation part of the two-step QML; (b) full Gaussian LL. Default when silent: (a) - Engle-Sheppard 2001.
- V10 dcc_covariance: H_t assembly. Options: (a) D_t R_t D_t with D_t from univariate GARCH sigma; (b) EWMA sigma in D_t. Default when silent: (a) - Engle 2002.

## Pitfalls
- Fitting arch on decimal returns: optimiser stalls at start values (round alpha/beta, poor LL) - rescale to percent.
- Demeaning before EWMA; (1-lambda) and lambda swapped; np.dot(r, r) scalar instead of outer.
- Annualising vol by 252 or variance by sqrt(252).
- Using in-sample sigma_T where the forecast sigma_{T+1} is required (or vice versa).
- DCC: applying D R D on top of already-correlated simulated rows; using np.corrcoef when Z'Z/T is standard.
- YZ: using close-to-close for V_c (textbook is open-to-close; follow a text that defines otherwise); k with n = returns instead of days.
- Rolling windows: a W-day window has W-1 close-to-close returns but W range terms; count windows exactly as stated.
- Forward-filling missing minutes creates zero returns, biasing RV down.
- Term structure: average VARIANCES then sqrt, unless told otherwise.
- Rogers-Satchell = ln(H/O)(ln(H/O) - ln(C/O)) + ln(L/O)(ln(L/O) - ln(C/O)), not squared terms. Per-day realized measures (RV, BV, noise) are computed inside each day's group: a later loop that reuses the last day's frame gives every day the same value.

## Sources
RiskMetrics TD (1996); Bollerslev (1986); GJR (1993); Engle (2002); arch docs (Sheppard); ABDL (2003); BN-S (2004); Bandi-Russell (2006); ZMA (2005); Parkinson (1980); Garman-Klass (1980); Rogers-Satchell (1991); Yang-Zhang (2000); Hull RMFI.
