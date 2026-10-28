method: capital-protected-note
family: exotic-options
keywords: structured note, capital protected note, principal protected note, zero-coupon bond plus option, participation rate, note fair value, issuer markup, breakeven volatility, note delta, note var, bond pv, option leg, n_options
---
## Formulas
Notation: notional N, participation p, spot S0, strike K, rate r, maturity T, option unit price c(sigma).
1. Bond leg: bond_pv = N e^{-rT} (continuous) or N/(1+r)^T if annual compounding is stated.
2. Option leg: n_options = p N / S0; option_value = n_options x c(sigma).
3. Fair value = bond_pv + option_value; markup_pct = 100 (N - fair)/N (issue price = notional).
4. Breakeven vol: solve bond_pv + n_options c(sigma) = N by brentq on a wide bracket; NaN if no root.
5. Note delta = n_options x option delta; 1-day delta-normal VaR = z_alpha |note_delta| S0 sigma_d; ES = phi(z_alpha)/(1 - alpha) |note_delta| S0 sigma_d, sigma_d = sigma/sqrt 252.
## Variants to decide
- V1 n_options_basis: option count. Options: (a) p N / S0; (b) p N / K. Default when silent: (a) - Hull ch. 26 (participation on spot notional).
- V2 markup_base: markup denominator. Options: (a) issue price (notional); (b) fair value. Default when silent: (a) - issuer quoting practice (Das, Structured Products).
- V3 risk_horizon: VaR horizon. Options: (a) 1 day; (b) remaining life T; (c) 10 days. Default when silent: (a) - RiskMetrics 1996.
- V4 garch_vol: pricing vol. Options: (a) one-step forecast, annualised; (b) long-run. Default when silent: (a) - Engle (2001).
## Invariants
- bond_pv < N; fair value increases in sigma only if the option vega is positive (barrier calls can have negative vega).
- ES > VaR > 0 for the same horizon and confidence.
## Pitfalls
- Using the long-run GARCH vol where the conditional forecast is meant.
- Reporting omega in percent^2 units next to decimal variances without saying so.
- Seeding the forecast with the long-run variance: take sigma2_next from the arch forecast (garch11_forecast).
## Sources
- Hull, Options Futures and Other Derivatives (ch. 26); Das, Structured Products (2001); Jorion (2007).
