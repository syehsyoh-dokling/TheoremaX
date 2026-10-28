family: regulatory-capital
category: risk-management
keywords: basel, capital charge, internal models approach, ima, stressed var, svar, frtb, expected shortfall, liquidity horizon, traffic light, multiplier, plus factor, backtesting exceptions, rwa, risk weight, leverage ratio, leverage cap, vol targeting, volatility target, cta, trend following, garch, square root of time, holding period, ucits, basel_multiplier, stress window, ema crossover, sma-seeded ema, ewma volatility, calmar, standardised approach
---
## Worked frame
DIKETAHUI: returns [T] or [T x N] (decimal, daily), VaR series [T], multipliers, horizon (days), weights [N]. Check: sorted dates, lagged weights, percent vs decimal fit.
DITANYAKAN: capital and components (scalars), zone, plus factor, vol forecast (daily, annual), weights [T x N].
RUMUS: card Core 3 -> 2 -> 4 -> 1 or 5; 7 (vol target); toolkit parametric_var_es, scale_horizon, exceedances, basel_traffic_light, ewma_vol.
JAWAB: 1) returns [T]. 2) one-day VaR, stressed VaR: scalars or [T]. 3) scale to the horizon. 4) exceptions [T] -> count -> plus factor. 5) components -> capital scalar. 6) daily -> annual vol; fraction x value -> currency. 7) check: capital = sum of components, ES >= VaR, weights within caps; then write.

## Core formulas
1. Basel 2.5 IMA (BCBS 158/193, 2009): capital = max(VaR_{t-1}, m_c VaR_avg60) + max(sVaR_{t-1}, m_s
   sVaR_avg60); VaR and sVaR 99% one-tailed, 10-day; m_c, m_s >= 3 (+ plus factor). Plus IRC (99.9%, 1y).
   If only current numbers are given: capital = m VaR + m sVaR (report both components).
2. Horizon: VaR_h = VaR_1 sqrt(h) (iid, zero mean; 10-day from 1-day allowed). GARCH term structure
   alternative: var_h = sum_{k=1..h} [VL + (a+b)^(k-1)(sigma^2_{T+1} - VL)], VL = omega/(1-a-b).
3. Parametric VaR (loss > 0): z_q sigma sqrt(h) (zero mean) or z_q sigma - mu; ES_normal = sigma
   phi(z_q)/(1-q).
4. 1996 traffic light (99% VaR, 250 days): green 0-4 exceptions plus 0; yellow 5:0.40, 6:0.50, 7:0.65,
   8:0.75, 9:0.85; red >=10: 1.00. m = 3 + plus.
5. FRTB (BCBS d457 2019, MAR33): ES 97.5% one-tailed, base T = 10 days. Liquidity-adjusted
   ES = sqrt( ES_T(P)^2 + sum_{j>=2} [ES_T(P,j) sqrt((LH_j - LH_{j-1})/T)]^2 ), LH in {10,20,40,60,120},
   ES_T(P,j) = ES with only factors of horizon >= LH_j shocked. Stress calibration
   ES = ES_{R,S} x max(ES_{F,C}/ES_{R,C}, 1). IMCC = 0.5 IMCC(all) + 0.5 sum_classes IMCC(class).
   Capital = max(IMCC_{t-1} + SES_{t-1}, m_c IMCC_avg60 + SES_avg60) + DRC; m_c = 1.5 + plus;
   FRTB plus (99% VaR exceptions, 250d): 0-4:0, 5:0.20, 6:0.26, 7:0.33, 8:0.38, 9:0.42, >=10:0.50.
6. Credit: RWA = 12.5 K EAD (IRB K, see portfolio-credit-risk); SA corporate risk weights (Basel III 2017):
   AAA..AA- 20%, A+..A- 50%, BBB 75% (100% Basel II), BB 100%, below BB- 150%, unrated 100%.
   Minimums: CET1 4.5%, Tier 1 6%, total 8% of RWA, conservation buffer 2.5%; leverage ratio
   Tier1/exposure >= 3%.
7. CTA vol-targeting: w_i = signal_i (target/N)/sigma_i (equal risk budget, linear); alternative
   target/sqrt(N) assumes uncorrelated assets (use only if stated); portfolio-level scaling
   w = w_raw sigma_target/sigma_p_hat. Clip |w_i| <= L_asset; gross cap: if sum|w| > L scale all by L/sum|w|.
## Conventions and defaults when the task is silent
- Confidence 99% for VaR/sVaR, 97.5% for ES; loss reported as a positive number.
- sVaR window: 12 continuous months (250 business days) of significant stress; if the data have no
  known crisis date, pick the window that maximises the SAME VaR model (task-stated choice beats
  max realised vol / max drawdown).
- Weights at close t earn return t+1 (portfolio_r[t] = sum_i w_i[t-1] r_i[t]); first day return 0/dropped.
- EWMA variance: alpha = 2/(span+1) (span cue) or 1-lambda with RiskMetrics lambda = 0.94 daily (lambda
  cue). pandas ewm(adjust=True) default is NOT the recursion; use adjust=False with the stated seed.
- Annualisation 252; vol = sqrt(252 var_daily). Signal +1/-1 binary if "no flat state" is stated.
- GARCH (arch package): default mean='Constant'; "no mean term" -> mean='Zero'. Units: fit on raw
  returns (rescale=False) gives omega in daily return^2 (~1e-6..1e-5); if fitted on 100 x r, divide omega
  and variances by 1e4. Persistence = a + b; LRV = omega/(1-a-b) (daily); annual = sqrt(252 LRV).
  1-step forecast: res.forecast(horizon=1).variance.iloc[-1,0] = sigma^2_{T+1}; conditional_volatility[-1]
  is in-sample sigma_T (different number; cue "forecast"/"next day" -> the former).
- UCITS (CESR/10-788): absolute VaR <= 20% NAV (99%, 20-day, >= 1y history); relative VaR <= 2x
  reference portfolio; commitment approach global exposure <= 100% NAV.
## Completing missing inputs
- No multiplier: 3 (Basel 2.5) or 1.5 (FRTB). No holding period: 10 days. No plus factor: count
  exceptions over the last 250 days and read the table above.
- 10-day from daily: sqrt(10). Liquidity horizon per MAR33.12, e.g. IR major ccy 10, equity large-cap
  price 10, small-cap 20, FX major pairs 10, sovereign IG spread 20, corporate IG 40, HY 60,
  energy commodity price 20, other commodity 60.
- sigma_p from covariance: sqrt(w' Sigma w); from history: std of lagged-weight portfolio returns.
## Invariants to self-check
- capital = var_component + svar_component exactly; each component = m x its VaR.
- sVaR >= current VaR typically (guaranteed only if the current-VaR window is one of those searched).
- ES_97.5 ~ VaR_99 for normal (2.338 vs 2.326 sigma); ES >= VaR at same q.
- Liquidity-adjusted ES >= ES_T(P); plus factor non-decreasing in exceptions.
- a, b >= 0, a+b < 1; LRV > 0 of order daily var of the series; |w_i| <= cap after clipping.
## Variants to decide
- V1 hp_scaling: 10-day VaR. Options: (a) sqrt(10) x 1-day; (b) 10-day overlapping returns. Default when silent: (a) - Basel 1996.
- V2 ewma_lambda: EWMA decay. Options: (a) lambda=0.94 daily; (b) alpha=2/(span+1). Default when silent: (a) - RiskMetrics 1996.
- V3 ewma_seed: recursion seed. Options: (a) first squared return; (b) sample variance of initial block; (c) pandas adjust=True. Default when silent: (a) - RiskMetrics 1996.
- V4 garch_mean: GARCH mean. Options: (a) constant mean; (b) zero mean. Default when silent: (a) - Bollerslev 1986 / arch package.
- V5 annualisation: vol scaling. Options: (a) sqrt(252); (b) sqrt(260) or sqrt(365). Default when silent: (a) - Hull.
- V6 vol_target_lag: leverage timing. Options: (a) leverage from sigma through t-1 applied to r_t; (b) same day. Default when silent: (a) - Moreira-Muir 2017.
- V7 capital_form: IMA aggregation. Options: (a) m VaR + m sVaR on current values; (b) max of previous day and m x 60-day average for each. Default when silent: (a) - BCBS 158 when only current VaR and sVaR exist; (b) with a 60-day VaR history.
- V8 svar_window: stress window choice. Options: (a) every rolling 250-day window, refit the same model, take the max window-end VaR; (b) max realised volatility window; (c) fixed crisis window. Default when silent: (a) - BCBS 158 intent, when the window must be identified from the data.
- V9 var_report_horizon: VaR intermediate. Options: (a) holding-period VaR entering the capital; (b) 1-day VaR. Default when silent: (a) - Basel 1996 10-day chain; (b) only when "1-day" is named.
- V10 equity_curve: drawdown wealth path. Options: (a) exp(cumsum) of log returns; (b) cumprod(1 + r) of simple returns. Default when silent: (a) - Magdon-Ismail-Atiya 2004 for log returns; (b) for simple returns (match the return type).

## Pitfalls
- Percent-scale GARCH parameters reported as native (omega ~1e-2 is the tell).
- sqrt(h) applied to variance, or to annual vol instead of daily.
- Using sVaR from a different model/confidence than VaR.
- Look-ahead: unlagged weights, or regime labels shifted twice.
- Annual return: arithmetic mean x252 vs exp(252 mean_log)-1 - follow the stated definition; MDD
  reported negative (min of dd) vs positive magnitude; Calmar uses |MDD|.
- std ddof: numpy default 0, pandas default 1 - state it.
- Stressed VaR from rolling GARCH windows: refit each window from the default start; warm starts can stall the optimizer and move the maximum.
- CTA warm-up: an undefined EMA gives weight 0 and those zero-return rows stay in the series (warmup='zero').
## Sources
BCBS 1996 Amendment and backtesting framework; BCBS 158 (2009, Basel 2.5); BCBS d352 (2016) and d457
(2019) FRTB / MAR33, MAR99; BCBS 128 (2006); BCBS d424 (2017); CESR/10-788 (2010) UCITS risk
measurement; RiskMetrics Technical Document 1996; Bollerslev 1986; Moskowitz-Ooi-Pedersen 2012;
Hurst-Ooi-Pedersen 2017.
