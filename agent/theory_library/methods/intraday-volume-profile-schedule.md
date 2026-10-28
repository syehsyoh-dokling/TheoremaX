method: intraday-volume-profile-schedule
family: execution-and-microstructure
keywords: intraday volume profile, u-shaped volume, volume curve, vwap schedule, volume fitting, ewma profile, half-life, winsorized profile, median profile, largest remainder, execution schedule, complete days, walk-forward profile
---
## Formulas
1. Bars: resample label=left, closed=left; complete day = all session bars. Share s_{d,b} = V_{d,b}/sum_b V_{d,b}; pivot on time-of-day.
2. Profiles over W complete days before the target: mean; per-bar median; EWMA w_L = exp(-ln2 L/H), L = 0 most recent; winsorised mean (clip each bar across days, numpy linear percentiles). Renormalise to 1.
3. Daily R2 = 1 - sum(s - shat)^2/sum(s - mean s)^2; score = mean daily R2; best = argmax.
4. Schedule on the order day's session grid, bars starting after the order time: renormalise; q_b = floor(Q w_b) + largest remainder, ties to the earlier bar.
## Variants to decide
- V1 ewma_lag_order: EWMA weight order. Options: (a) lag 0 = most recent day, largest weight; (b) oldest first. Default when silent: (a) - RiskMetrics 1996.
- V2 schedule_grid: order-day bars. Options: (a) full session grid regardless of completeness; (b) complete-day bars only. Default when silent: (a) - Kissell 2013.
- V3 rounding: integer allocation. Options: (a) largest remainder, ties earlier bar; (b) round and fix the last bar. Default when silent: (a) - Hamilton apportionment.
- V4 profile_score: fit score. Options: (a) mean of daily R2; (b) pooled R2. Default when silent: (a) - Bialkowski-Darolles-Le Fol 2008.
## Invariants
- Profiles sum to 1; schedule integers sum to Q; cumulative ends at Q.
## Pitfalls
- Weights applied in oldest-to-newest row order; winsorising across bars.
- Empty schedule because the order day is incomplete.
- Hand loops reversed the EWMA lags (lag 0 = OLDEST): call volume_share_profile / walk_forward_profile_scores.
## Sources
- Bialkowski-Darolles-Le Fol 2008; Kissell 2013; Konishi 2002.
