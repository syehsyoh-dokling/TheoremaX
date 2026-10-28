method: basel-market-risk-capital
family: regulatory-capital
keywords: basel ima capital charge, internal models approach, stressed var, svar, basel 2.5, var multiplier, basel_multiplier, var_component, svar_component, standardised approach, stress window, 250-day rolling window, 10-day var, plus factor
---
## Formulas
1. Basel 2.5 IMA: K = max(VaR_{t-1}, m_c avg60 VaR) + max(sVaR_{t-1}, m_s avg60 sVaR); current-only form K = m VaR + m sVaR.
2. VaR_h = z_c sigma_1 sqrt(h) (zero mean); sigma_1 = one-step GARCH or EWMA forecast.
3. sVaR = same model on a 12-month (250-day) stress window; data-identified: max over rolling windows of the window-end VaR.
4. Standardised approach (pre-FRTB): sum of specific and general risk charges per asset class; no VaR, no multiplier.
## Variants to decide
- V1 capital_form: aggregation. Options: (a) m x VaR + m x sVaR on current values; (b) max with 60-day averages. Default when silent: (a) - BCBS 158 when only current values exist; (b) with a VaR history.
- V2 svar_window_search: stress window. Options: (a) every rolling 250-day window, step one day, refit each; (b) max realised volatility window; (c) supervisor-fixed window. Default when silent: (a) - BCBS 158 intent for a data-identified window.
- V3 var_report_horizon: VaR intermediate. Options: (a) holding-period VaR that enters capital; (b) 1-day VaR. Default when silent: (a) - Basel 1996 10-day chain; (b) only when 1-day is named.
- V4 approach: method. Options: (a) IMA; (b) standardised. Default when silent: (a) - BCBS 1996 when VaR models and a multiplier are given.
## Invariants
- capital = var_component + svar_component; sVaR >= current VaR when the current window is searched.
## Pitfalls
- sVaR from a different model, confidence or horizon than VaR.
- Per-window refits are slow: warm-start and cap iterations.
- VaR from the long-run GARCH variance: the current VaR uses the one-step conditional variance (basel_ima_garch_capital).
## Sources
- BCBS 1996 Amendment; BCBS 158 (2009); BCBS d457 (2019); Hull RMFI ch.16.
