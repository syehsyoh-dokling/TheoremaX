family: structural-credit-and-spreads
category: credit
keywords: merton model, structural model, asset value, asset volatility, distance to default, kmv, edf, default point, hazard rate, default intensity, survival probability, cds, credit default swap, par spread, bootstrap, upfront, recovery rate, credit spread, spread decomposition, liquidity premium, default component, newey-west, cva, expected exposure, cds-bond basis, non-default component, variance share, credit spread changes
---
## Worked frame
DIKETAHUI: firms [N] (equity value, debt face, equity vol, r, T); or CDS quotes [P] (tenor in years, spread in bp -> decimal), recovery, curve. Check: sorted tenors, equity and debt in one money unit.
DITANYAKAN: per firm [N]: asset value, asset vol, DD, PD, spread; hazard, survival [P]; par spread, CVA (scalars).
RUMUS: card Core 1-3 (Merton), 4-6 (CDS), 7 (CVA); toolkit merton_calibrate, merton_model, hazard_from_spread, survival_prob, cds_bootstrap.
JAWAB: 1) align units. 2) asset value, vol [N]. 3) DD, PD, spread [N]. 4) CDS grid: DF, survival [n]. 5) hazards in tenor order [P]; legs. 6) decimal -> bp; x notional -> currency. 7) check: V > E, sigma_V < sigma_E, survival decreasing, inputs reprice; then write.

## Core formulas
1. Merton (1974): E = V N(d1) - D e^(-rT) N(d2), d1 = [ln(V/D) + (r + sigma_V^2/2)T]/(sigma_V sqrt T),
   d2 = d1 - sigma_V sqrt T. Volatility link: sigma_E E = N(d1) sigma_V V. Solve the 2x2 system
   (fsolve) from V0 = E + D e^(-rT), sigma_V0 = sigma_E E/V0.
2. Risk-neutral PD(T) = N(-d2). Real-world DD = [ln(V/D) + (mu - sigma_V^2/2)T]/(sigma_V sqrt T),
   PD = N(-DD). KMV: default point DPT = short-term debt + 0.5 long-term debt, DD = (V - DPT)/(sigma_V V),
   EDF from an empirical DD->default map (N(-DD) is only a proxy).
3. Debt: B0 = V - E = V N(-d1) + D e^(-rT) N(d2). Spread s = -(1/T) ln(B0/D) - r
   = -(1/T) ln[N(d2) + V N(-d1)/(D e^(-rT))] (continuous, decimal per year).
4. Hazard: S(t) = exp(-int_0^t lambda); piecewise-constant lambda_k on (t_{k-1}, t_k].
   Default prob in (a,b] = S(a) - S(b). Credit triangle: s ~ lambda (1-R); flat lambda = s/(1-R).
5. CDS (notional 1): RPV01 = sum_i Delta_i DF(t_i) S(t_i) + accrual-on-default
   sum_i (Delta_i/2) DF(t_i)(S(t_{i-1}) - S(t_i)). Protection = (1-R) sum_i DF(t_i)(S(t_{i-1}) - S(t_i))
   (finer time grid or midpoint DF improves it). Par spread = Protection/RPV01. Protection-buyer MTM =
   (s_mkt - s_contract) RPV01_mkt. Upfront (buyer pays, fraction of notional) = (s_par - c) RPV01.
6. Bootstrap: for tenors in increasing order solve lambda_k (brentq, bracket [1e-8, 5]) so PV of the
   tenor-k CDS at its par spread = 0, keeping earlier lambdas fixed.
7. Unilateral CVA = (1-R) sum_i DF(t_i) EE(t_i) [S(t_{i-1}) - S(t_i)], EE = E[max(V_t,0)]; DVA uses own
   S and E[max(-V,0)]; BCVA = CVA - DVA. PFE_q = q-quantile of max(V_t,0); EPE = time-average of EE.
8. Spread decomposition (Longstaff-Mithal-Neis 2005): default component = CDS spread; non-default
   (liquidity/tax) = bond yield spread - CDS. Expected-loss part = PD_P (1-R) (Elton et al 2001); the
   remainder is risk premium. Regression decomposition: pct_k = Cov(b_k X_k, Y)/Var(Y) x100,
   sum_k pct_k = 100 R^2 (OLS with intercept), residual share = 100(1-R^2).
9. Newey-West: S = G0 + sum_{j=1..m} (1 - j/(m+1))(G_j + G_j'), G_j = sum_{t>j} e_t e_{t-j} x_t x_{t-j}';
   V = (X'X)^-1 S (X'X)^-1. Automatic lag if unspecified: floor(4 (T/100)^(2/9)).
## Conventions and defaults when the task is silent
- T in years, r continuous, sigma annualised; E = market cap (price x shares); D = face of debt
  (book total liabilities or the KMV default point if "default point" appears).
- Equity vol from daily log returns: std(ddof=1) sqrt(252).
- ISDA standard CDS: quarterly premiums on IMM dates (20 Mar/Jun/Sep/Dec), ACT/360, R = 40% senior
  unsecured, 20% subordinated, 25% EM; standard coupons 100/500 bp (NA), accrued-on-default included.
  Simple textbook tasks: annual/quarterly year fractions 0.25, payment at period end, no accrual unless
  asked (Hull ch.25 includes the half-period accrual).
- Spreads quoted in bp -> /10000; FRED yields in percent points (5.80 = 5.80%): pp x100 = bp.
- Survival from constant hazard is continuous: exp(-lambda t), not (1-lambda)^t.
- Default lag for HAC is the task's m; no small-sample T/(T-k) factor (statsmodels HAC: check the
  use_correction flag so it matches the raw sandwich).
## Completing missing inputs
- No r: use given discount curve or flat risk-free; no R: 40%.
- Hazard from bond: lambda ~ (y_corp - y_rf)/(1-R). PD over T from spread: 1 - exp(-s T/(1-R)).
- Missing DF at payment dates: interpolate zero rates linearly (or log-DF linear) - state which.
- Asset value series (iterative KMV, Vassalou-Xing 2004): invert daily E_t for V_t given sigma_V,
  re-estimate sigma_V from log V returns, iterate to tolerance 1e-4.
- Naive DD (Bharath-Shumway 2008): sigma_D = 0.05 + 0.25 sigma_E, sigma_V = (E sigma_E + F sigma_D)/(E+F),
  mu = past-year equity return.
## Invariants to self-check
- Merton: V > E, E >= V - D e^(-rT), sigma_V < sigma_E, 0 < N(-d2) < 1, s >= 0; equity reprices input.
- PD increases with leverage D/V, sigma_V and T (for typical firms); DD decreases.
- S(t) strictly decreasing from 1, lambda > 0; bootstrapped curve reprices EVERY input CDS to PV 0.
- Par spread increases with lambda and with (1-R); RPV01 < annuity without default.
- CVA >= 0, CVA <= (1-R) max EE; EE >= 0; PFE_q >= EE for high q.
- Regression shares + residual share = 100; 0 <= R^2 <= 1.
## Variants to decide
- V1 merton_solve: asset value/vol. Options: (a) solve E and sigma_E equations jointly; (b) iterative KMV time series. Default when silent: (a) - Merton 1974 / Jones-Mason-Rosenfeld 1984.
- V2 default_point: D. Options: (a) total liabilities face; (b) KMV STD + 0.5 LTD. Default when silent: (a) - Merton 1974.
- V3 dd_drift: DD drift. Options: (a) risk-neutral r; (b) physical mu. Default when silent: (a) - Merton 1974 / Hull.
- V4 cds_accrual: accrued premium on default. Options: (a) include half-period accrual; (b) none. Default when silent: (b) - simple textbook convention; (a) Hull / ISDA model.
- V5 survival_form: survival from hazard. Options: (a) exp(-lambda t); (b) (1-lambda)^t. Default when silent: (a) - Hull / O'Kane 2008.
- V6 spread_share_method: explained-variance split. Options: (a) covariance shares b_k Cov(X_k,Y)/Var(Y); (b) Shapley / LMG R^2 decomposition; (c) squared standardised betas. Default when silent: (a) - Pratt 1987.
- V7 spread_regression_form: spread regression. Options: (a) first differences; (b) levels. Default when silent: (a) - Collin-Dufresne-Goldstein-Martin 2001.
- V8 hazard_shape: between CDS tenors. Options: (a) piecewise-constant hazard; (b) piecewise-linear hazard. Default when silent: (a) - O'Kane 2008.

## Pitfalls
- Using sigma_E as sigma_V in d1/d2; using N(-d1) as PD; forgetting e^(-rT) on D.
- Real-world DD with r instead of mu (or vice versa) when the task states which measure.
- Negative hazard from bootstrapping inverted/noisy curves - report, do not silently floor unless told.
- Premium leg without survival weighting; protection leg discounted at start instead of default time.
- Mixing bp and decimal in the RPV01 product; upfront sign (buyer pays when s_par > coupon).
- FRED "." as missing -> to_numeric(errors="coerce"); monthly alignment by last valid obs; window by
  rows after dropping; forward-fill discontinued series before dropping rows.
- Var(Y) and covariances with different ddof -> shares no longer sum to 100.
- Joining DAILY series on monthly dates keeps only 1st-of-month business days: collapse each series to its last valid value per month first (credit_spread_report).
## Sources
Merton 1974 (J Finance); Crosbie-Bohn 2003 (Moody's KMV); Vassalou-Xing 2004; Bharath-Shumway 2008;
Hull "Options, Futures" ch.24-25; O'Kane 2008 "Modelling Single-name and Multi-name Credit
Derivatives"; ISDA CDS Standard Model; Longstaff-Mithal-Neis 2005; Elton-Gruber-Agrawal-Mann 2001;
Newey-West 1987/1994; Gregory "Counterparty Credit Risk" (CVA).
