method: residual-momentum
family: factor-models-and-sorts
keywords: residual momentum, idiosyncratic momentum, blitz huij martens, factor-adjusted momentum, residual return, stable residual, residualise signals, residual volatility, ff3 residual, rolling regression residual, sign stability
---
## Formulas
1. Rolling per stock over W months ending at t: r - rf = a + b'F + e (window's own a, b).
2. Score_t = sum of e over t-12..t-2 / std(e over the same months).
3. Signal residualisation (per date): regress signal on controls with intercept (pinv), keep residual, then z-score.
4. Slope stability: median of the last M monthly FM slopes; keep a signal if the sign agrees in at least k of M months.
5. Portfolio: sort on score at t, hold t+1.
## Variants to decide
- V1 resid_intercept: residual definition. Options: (a) e = r - a - b'F; (b) e = r - b'F (alpha kept). Default when silent: (a) - Blitz-Huij-Martens 2011.
- V2 resid_scaling: score scaling. Options: (a) sum / std(formation residuals, ddof=1); (b) sum / std(all window residuals); (c) raw sum. Default when silent: (a) - Blitz-Huij-Martens 2011.
- V3 formation_window: momentum months. Options: (a) t-12..t-2, skip latest; (b) t-12..t-1. Default when silent: (a) - Jegadeesh-Titman 1993.
- V4 resid_universe: rows used in per-date residualisation. Options: (a) eligible universe only, winsorised first; (b) all rows of the month. Default when silent: (a) - Bali-Engle-Murray 2016.
## Invariants
- Window residuals sum ~0; no return at or after t+1 enters a score dated t.
## Pitfalls
- Factors in percent with decimal returns; one full-sample regression (look-ahead).
- Python loops over the full panel per month (timeouts): pre-group by date once.
## Sources
- Blitz, Huij, Martens 2011; Gutierrez-Pirinsky 2007; Jegadeesh-Titman 1993; Bali-Engle-Murray 2016.
