method: event-study-tests
family: event-studies-and-filings
keywords: event study, market model, abnormal return, cumulative abnormal return, caar, patell test, standardized residual, bmp test, boehmer musumeci poulsen, corrado rank test, generalized rank test, kolari pynnonen, estimation window, event window
---
## Formulas
1. Estimation window of L1 trading-day positions before the event window: AR = R - a - b Rm; s^2 = SSR/(L1-2).
2. Patell SAR_t = AR_t/(s sqrt(C_t)), C_t = 1 + 1/L1 + (Rm_t - mean Rm)^2/sum(Rm - mean Rm)^2; SCAR = sum SAR/sqrt(L2).
3. BMP t = mean(SCAR)/(std(SCAR, ddof=1)/sqrt N).
4. KP: t_BMP sqrt((1-r)/(1+(N-1)r)), r = mean pairwise residual corr on common dates.
5. Corrado: U = rank(AR)/(T_i+1) - 0.5 over estimation+event days; S = sqrt(mean_t Ubar_t^2); z = sum_window Ubar/(sqrt(L2) S). Generalized rank (KP 2011) ranks standardized AR with the window CAR as one observation.
## Variants to decide
- V1 short_estimation: events lacking a full window. Options: (a) keep with the available window if >= the stated minimum; (b) drop. Default when silent: (a) - MacKinlay 1997.
- V2 event_day_roll: non-trading event date. Options: (a) next trading day; (b) previous day; (c) drop. Default when silent: (a) - Campbell-Lo-MacKinlay 1997.
- V3 kp_form: correlation correction. Options: (a) multiply by sqrt((1-r)/(1+(N-1)r)); (b) divide by sqrt(1+(N-1)r). Default when silent: (a) - Kolari-Pynnonen 2010.
- V4 rank_test_form: rank test version. Options: (a) Corrado ranks of AR; (b) generalized rank on standardized AR. Default when silent: (a) - Corrado 1989.
## Invariants
- CAR = sum AR; windows disjoint; relative days are positions.
## Pitfalls
- Calendar Timedelta windows; dropping events that still meet the minimum.
- Corrado given date-indexed or gap-less arrays: label residuals by relative day so the span denominator counts the gap (event_study_tables does it).
## Sources
- Patell 1976; Boehmer-Musumeci-Poulsen 1991; Corrado 1989; Kolari-Pynnonen 2010, 2011; MacKinlay 1997.
