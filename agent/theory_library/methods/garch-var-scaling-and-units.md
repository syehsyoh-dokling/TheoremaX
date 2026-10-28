method: garch-var-scaling-and-units
family: volatility-models
keywords: garch var, conditional var, arch_model, percentage returns, rescale=false, omega scaled by 1e4, conditional volatility back by 100, one step ahead forecast, square root of time, holding period var, long run variance, decimal units
---
## Formulas
1. s2_{t+1} = omega + alpha r_t^2 + beta s2_t; P = alpha + beta; V_L = omega/(1-P).
2. Fit on x = 100 r: omega_dec = omega_pct/1e4; sigma_dec = sigma_pct/100; alpha, beta unchanged.
3. z_t = r_t/sigma_dec,t = x_t/sigma_pct,t (unit-free).
4. VaR_1 = z_c sigma_{T+1}; h-day (a) sqrt(h) VaR_1 or (b) z_c sqrt(sum_{k<=h} E s2_{T+k}).
5. Forecast: res.forecast(horizon=1).variance.iloc[-1,0]; in-sample: res.conditional_volatility[-1].
## Variants to decide
- V1 fit_scale: optimiser input. Options: (a) percent then descale; (b) native, rescale=False. Default when silent: (a) - arch docs (Sheppard); (b) when rescaling is forbidden.
- V2 sigma_choice: vol for VaR. Options: (a) forecast sigma_{T+1}; (b) in-sample sigma_T. Default when silent: (a) - Christoffersen 2012; (b) when "last in-sample" is said.
- V3 horizon_rule: multi-day VaR. Options: (a) sqrt(h) x 1-day; (b) GARCH term structure. Default when silent: (a) - Basel 1996.
- V4 garch_mean: mean model. Options: (a) constant; (b) zero. Default when silent: (a) - arch default; (b) when "no mean term".
## Invariants
- V_L near the sample variance; VaR_h/VaR_1 = sqrt(h) under (a).
## Pitfalls
- Decimal r over percent sigma, then rescaled again (off by 100).
- Percent-scale omega (order 1e-2) reported as native.
- h-day VaR reported where 1-day is asked or vice versa.
- Hand-rolled next variance (cond_vol^2/100, unconditional seed): use the arch one-step forecast (garch11_forecast).
## Sources
- Bollerslev 1986; arch docs; Christoffersen 2012; Hull RMFI ch.23; BCBS 1996.
