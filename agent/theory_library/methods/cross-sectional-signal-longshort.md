method: cross-sectional-signal-longshort
family: factor-models-and-sorts
keywords: cross-sectional z-score, signal fusion, composite score, sentiment factor, alpha signal, long-short portfolio, top-n bottom-n, information coefficient, spearman ic, rank ic, turnover, market neutral, alpha hedge, ff3 alpha
---
## Formulas
1. Per date: fill missing with the cross-sectional median, z = (x - mean)/std, clip +-c; composite = sum w_k z_k.
2. Lookback signal: compounded prod(1+r) - 1 over the lookback through t.
3. Book: long top N, short bottom N, equal weight; legs +g/2 and -g/2; per-name caps.
4. r_{p,t+1} = sum w_t r_{t+1} - cost x sum|w_t - w_{t-1}| (first book from zero).
5. IC_t = spearman(signal_t, r_{t+1}); t = mean/(std(ddof=1)/sqrt(T)).
6. OLS r_p - rf = a + b'F; alpha x P; TE = std(e) sqrt(P); IR = a P/TE.
## Variants to decide
- V1 cs_std_ddof: cross-sectional std. Options: (a) ddof=0 population; (b) ddof=1. Default when silent: (a) - Grinold-Kahn 2000.
- V2 signal_return_alignment: lag placement. Options: (a) w_t x r_{t+1} with one lag only; (b) w_{t-1} x r_t, the same shift. Default when silent: (a) - Grinold-Kahn 2000.
- V3 eval_start: metric start date. Options: (a) first traded day; (b) first signal day. Default when silent: (a) - Bacon 2008.
- V4 tie_break: ranking ties. Options: (a) ticker ascending; (b) file order. Default when silent: (a) - deterministic sort practice.
## Invariants
- Long legs = -short legs; gross = g; IC in [-1, 1]; z mean 0 per date.
## Pitfalls
- Normalising across the pooled panel instead of per date; same-day IC.
- Lagging both weights and returns; whitespace tokenising (use lowercase regex words).
## Sources
- Grinold-Kahn 2000; Qian-Hua-Sorensen 2007; Fama-French 1993; Loughran-McDonald 2011.
