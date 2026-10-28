method: ou-temperature-model
family: weather-derivatives
keywords: ornstein-uhlenbeck temperature, ou temperature model, exact ou transition, temperature anomaly, seasonal mean, day-of-month average, mean reversion speed, ou kappa, ou sigma, ou theta, temp delta, kappa sensitivity, weather monte carlo
---
## Formulas
1. Anomaly x = T_avg - s(day), s = day-of-month mean across years (or a + bt + c sin(2 pi t/365 + phi) when named).
2. OLS on adjacent same-year pairs: dx = x_{t+1} - x_t = alpha + beta x_t + eps; dt = 1/365; kappa = -beta/dt; theta = -alpha/beta; sigma = std(eps, ddof=1)/sqrt dt.
3. Exact step: x' = theta + (x - theta) e^{-kappa dt} + sigma sqrt((1 - e^{-2 kappa dt})/(2 kappa)) Z.
4. MC: x_0 given; each day d: T = s(d) + x, add HDD, then step (if told day 1 uses x_0). Payoff tick x max(I - K, 0); price = DF mean; SE = DF std(payoff, ddof=1)/sqrt N.
5. Greeks by central bump with the same seed: shift s by +-h_T; sigma and kappa by +-1% relative; Greek = (P_up - P_dn)/(2 h), h = absolute change.
## Variants to decide
- V1 ou_regression: estimator. Options: (a) Euler OLS on dx with kappa = -beta/dt; (b) AR(1) phi with kappa = -ln(phi)/dt; (c) Alaton martingale estimator. Default when silent: (a) - Dornier-Queruel 2000 discretisation.
- V2 day_order: index accrual. Options: (a) accrue from x_t then advance; (b) advance then accrue. Default when silent: (b) - first contract day one step after the last state.
- V3 relative_bump_h: relative bump denominator. Options: (a) absolute change 0.01 x parameter; (b) the fraction 0.01. Default when silent: (a) - Glasserman 2003.
## Invariants
- kappa > 0, sigma > 0; HDD call temp delta < 0, vega > 0, kappa sens < 0 for an OTM call.
## Pitfalls
- Cross-year or gap pairs in the regression; Euler instead of the exact step; new seed per bump.
- Misaligned shift(1) after masking rows: build pairs from the full sorted frame, then mask.
## Sources
- Dornier and Queruel (2000) Risk; Alaton, Djehiche, Stillberger (2002); Benth and Saltyte-Benth (2007).
