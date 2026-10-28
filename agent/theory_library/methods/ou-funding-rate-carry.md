method: ou-funding-rate-carry
family: jump-and-mean-reversion
keywords: funding rate, perpetual swap, perp funding, basis carry, cash and carry, ornstein-uhlenbeck, ou fit, mean reversion, half-life, ar(1) on funding, stationary std, adf test, carry var, funding interval
---
## Formulas
1. Funding every 8h (3/day); annualised = f x 3 x 365; carry = funding received - costs.
2. OLS df_t = alpha + beta f_{t-1} + e (or f_t = a + b f_{t-1} + e, b = 1 + beta).
3. Exact: kappa = -ln(b)/dt, theta = a/(1-b), sigma = sd(e) sqrt(2 kappa/(1-b^2)). Euler: kappa = -beta/dt, sigma = sd(e)/sqrt(dt).
4. Half-life ln2/kappa is in dt's unit (years if dt is a year fraction); periods = ln2/(kappa dt); days = periods/3 for 8h funding.
5. Stationary std sigma/sqrt(2 kappa); MC exact transition, accrual from step 1.
6. VaR = sorted losses[ceil(cN) - 1]; CVaR = mean of losses beyond VaR.
## Variants to decide
- V1 ou_mapping: AR(1) to OU. Options: (a) exact mapping; (b) Euler plug-in. Default when silent: (a) - Vasicek 1977.
- V2 half_life_unit: reported unit. Options: (a) the unit the field names, converted via dt; (b) raw ln2/kappa. Default when silent: (a) - dimensional consistency.
- V3 end_date_inclusive: date-only end bound. Options: (a) whole last day included (< end + 1 day); (b) <= midnight. Default when silent: (a) - calendar-date convention.
- V4 resid_ddof: residual std. Options: (a) ddof=1; (b) OLS s with n-2. Default when silent: (a) - pandas std default.
## Invariants
- 0 < b < 1; half-life positive; stationary mean near the sample mean.
## Pitfalls
- Half-life in years reported as periods; per-path Python loops (vectorise per step).
- Exact -ln(1+beta)/dt used when the task fixes plug-in Euler; end date filtered at midnight (drops the last day's later settlements).
## Sources
- Uhlenbeck-Ornstein 1930; Vasicek 1977; Brigo-Mercurio 2006; He-Manela-Ross-von Wachter 2022.
