family: weather-derivatives
category: derivatives-pricing
keywords: weather derivative, heating degree day, cooling degree day, hdd, cdd, degree day, cme weather, temperature, burn analysis, tick value, index value, strike, ornstein-uhlenbeck temperature, alaton, dornier, seasonal mean, daily average temperature, weather option, ghcn-daily, tmax, tmin, exact ou transition, bump-and-revalue, temp delta, kappa sensitivity
---
## Worked frame
DIKETAHUI: daily Tmax, Tmin [n_days], base temperature (same unit), strike (index points), tick (currency per point), r, period. Check: sorted dates, unique days, complete years.
DITANYAKAN: index per year [Y] (degree days), payoffs [Y] (currency), burn price, MC price, SE, Greeks (scalars).
RUMUS: card Core 1-2 -> 3-4 (burn) -> 5-6 (OU, MC) -> 7; toolkit ar1_fit, ou_from_ar1.
JAWAB: 1) daily average [n_days]. 2) degree days [n_days]. 3) sum over period per year -> index [Y]. 4) payoff [Y] -> mean, discount. 5) fit on deseasonalised residuals; simulate [n_paths x n_period_days] -> index [n_paths]. 6) index points x tick -> currency. 7) check: index >= 0 and is a sum, tick in every leg, same draws for bumps; then write.

## Core formulas
1. Daily average temperature T_d = (T_max + T_min)/2 unless the task defines it otherwise; CME base 65 F (18 C).
2. Daily HDD_d = max(65 - T_d, 0); daily CDD_d = max(T_d - 65, 0). The index for a PERIOD (a month, a season) is the SUM over the period's days: HDD = sum_d HDD_d - a monthly HDD is typically hundreds (e.g. ~600-900 in a northern winter month), never the daily average.
3. Option payoff: call = tick * max(Index - K, 0), put = tick * max(K - Index, 0), often capped: min(payoff, cap). Price = e^{-rT} E[payoff] under the chosen measure (burn / MC).
4. Burn analysis: for each historical year y, compute that year's period index I_y (same calendar window), payoff_y = tick * max(I_y - K, 0); expected payoff = mean_y payoff_y; price = e^{-rT} * expected payoff. Detrending (linear in year) optional when stated.
5. Temperature OU (Dornier-Queruel / Alaton): dT = [ds/dt + kappa (s(t) - T)] dt + sigma dW, seasonal mean s(t) = a + b t + c sin(2 pi t/365 + phi); fit kappa from the AR(1) of deseasonalised residuals: e_t = phi e_{t-1} + eps, kappa = -ln(phi) per day (x365 annual), sigma = sd(eps) sqrt(2 kappa/(1 - phi^2)) per sqrt(day).
6. MC: simulate daily T paths with the fitted OU over the contract days, compute the period index per path, payoff with tick, discount; SE = sd(payoff)/sqrt(N) * DF.
7. Greeks by bump-and-revalue with common random numbers: temp_delta = dPrice/d(level shift of the mean temperature, per 1 F); vega = dPrice/dsigma per ABSOLUTE bump (report the bump size as a fraction, e.g. 0.01 = 1%); kappa sensitivity = dPrice/dkappa.
## Conventions and defaults when the task is silent
- Period index = SUM of daily degree days over the calendar days of the period (inclusive of both ends).
- Tick value multiplies EVERY payoff leg (burn and MC alike); report prices in currency.
- Bumps: relative bumps are fractions (1% -> 0.01); the same random numbers for base and bumped runs.
- Temperatures in the unit of the data (F for CME US cities); do not mix C and F.
## Completing missing inputs
- Missing daily max/min: drop the day from the average only if the task says so; otherwise forward-fill single gaps.
- Historical years for burn: every complete year of data with the full period window.
## Invariants to self-check
- Monthly HDD >> daily HDD (sum of ~30 days); HDD >= 0; winter HDD > summer HDD.
- Call price increases with the mean index and decreases in K; HDD call: temp_delta < 0 (warmer -> fewer HDD); vega > 0 for an at-the-money call.
- Burn price and MC price of the same contract within ~20% when the model is calibrated to the same history.
## Variants to decide
- V1 period_index: HDD over a period. Options: (a) sum of daily HDD over the period's days; (b) average daily HDD. Default when silent: (a) - CME HDD index definition.
- V2 daily_temp: daily temperature. Options: (a) (Tmax + Tmin)/2; (b) mean of hourly readings. Default when silent: (a) - CME / NWS convention.
- V3 bump_units: reported bump size. Options: (a) fraction (1% -> 0.01); (b) percent number (1% -> 1.0). Default when silent: (a) - decimal convention for rates and vols.
- V4 ou_time_unit: OU kappa and sigma units. Options: (a) per day; (b) annualised (x365, sqrt(365)). Default when silent: (a) - Alaton-Djehiche-Stillberger 2002 daily model.
- V5 burn_detrend: burn history. Options: (a) raw historical indices; (b) linearly detrended. Default when silent: (a) - Jewson-Brix 2005.
- V6 discount_tenor: tau in DF = exp(-r tau). Options: (a) days in the contract period / 365 (ACT/365); (b) 1/12 per month; (c) to an explicit payment date incl. settlement lag. Default when silent: (a) - ACT/365 accrual (Jewson-Brix 2005).
- V7 ou_regression: OU estimator. Options: (a) Euler OLS dx = alpha + beta x, kappa = -beta/dt, sigma = sd(eps, ddof=1)/sqrt dt; (b) AR(1) kappa = -ln(phi)/dt; (c) Alaton martingale estimator. Default when silent: (a) - Dornier-Queruel 2000 (when dt = 1/365 is given, kappa and sigma are annualised).
- V8 relative_bump_h: denominator of a relative bump. Options: (a) absolute change h = 0.01 x parameter; (b) the fraction 0.01. Default when silent: (a) - Glasserman 2003.
- V9 day_order: MC index accrual. Options: (a) accrue HDD from x_t then advance; (b) advance then accrue. Default when silent: (b) - first contract day one step after the last state (a when day 1 is told to use x_0).
## Pitfalls
- Reporting the average daily HDD as the monthly HDD (an order of magnitude too small).
- Applying the tick value in one leg only (burn vs MC prices then differ ~20x).
- Bump reported as 1.0 for a 1% bump when the fraction 0.01 is asked.
- Different random numbers for base and bumped MC runs: Greeks drown in noise (vega sign flips).
- Fitting OU on raw temperatures instead of deseasonalised residuals.
- Cleaning every month when the contract is one month: filter station, element, contract month and years before the seasonal mean, OU fit and MC (ghcn_month_temperatures); the MC runs over that month's calendar days.
- mean/std of the monthly HDD index are over the YEARLY totals, and the MC HDD mean is the simulated index, not the payoff; cme_hdd_option_bundle writes all seven files.
- Discount an HDD payoff over the contract's days to expiry, not over the number of cleaned observations; report relative bumps as fractions.
## Sources
CME Group weather futures and options specifications; Jewson & Brix (2005) Weather Derivative Valuation; Alaton, Djehiche & Stillberger (2002); Dornier & Queruel (2000).
