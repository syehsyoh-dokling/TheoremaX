family: var-backtesting-and-data-prep
category: risk-management
keywords: backtest, exceedance, violation, kupiec, proportion of failures, pof, christoffersen, independence, conditional coverage, traffic light, basel, rolling window, expanding window, data cleaning, trading calendar, duplicates, outliers, missing values, mixed date formats, stale prices, adjusted close, portfolio p&l, log returns, simple returns, equal weight, rolling one day ahead, lr_ind, lr_cc, survival function, polars, api migration, group_by, deprecationwarning
---
## Worked frame
DIKETAHUI: raw rows [n_raw] (date, symbol, price or return), calendar [n_cal], window W (rows), level. Check: parse every date, de-duplicate before sorting.
DITANYAKAN: audit (rows removed per step), VaR and hits [T - W], tests, p-values, zone.
RUMUS: card Core 7 -> 1 -> 2-4 -> 6; toolkit historical_var_es, exceedances, kupiec_pof, christoffersen_independence, christoffersen_cc, basel_traffic_light.
JAWAB: 1) clean in the stated order, count per step. 2) pivot, align -> prices [T+1 x N]. 3) returns [T x N] -> portfolio [T]. 4) VaR for day t from rows before t [T - W]. 5) hits [T - W]; pairs [T - W - 1]. 6) percent -> decimal; VaR positive. 7) check: final = raw - removed, LR >= 0, LR_cc = LR_uc + LR_ind; then write.

## Core formulas
1. Violation I_t = 1{L_t > VaR_t} (equivalently r_t < -VaR_t), VaR_t built ONLY from data before t. x = sum I, T = number of forecast days, p = 1 - c.
2. Kupiec POF: LR_uc = -2[(T-x) ln(1-p) + x ln p] + 2[(T-x) ln(1-x/T) + x ln(x/T)] ~ chi2(1); p-value = chi2.sf(LR, 1). Use 0 ln 0 = 0 (x = 0 or x = T).
3. Christoffersen independence: count transitions over consecutive pairs (t-1, t), T-1 pairs: n00, n01, n10, n11. pi01 = n01/(n00+n01), pi11 = n11/(n10+n11), pi = (n01+n11)/(n00+n01+n10+n11). LR_ind = -2[(n00+n10) ln(1-pi) + (n01+n11) ln pi - n00 ln(1-pi01) - n01 ln pi01 - n10 ln(1-pi11) - n11 ln pi11] ~ chi2(1).
4. Conditional coverage LR_cc = LR_uc + LR_ind ~ chi2(2) (uc on all T days, ind on T-1 pairs: standard practice).
5. Binomial z: (x - pT)/sqrt(p(1-p)T).
6. Basel traffic light (250 days, 99% VaR): green 0-4, yellow 5-9, red >= 10 exceptions. Plus factor k: 0-4 -> 0; 5 .40; 6 .50; 7 .65; 8 .75; 9 .85; >=10 1.0. Capital = max(VaR_{t-1}, (3+k) mean(VaR over last 60 days)). Zones = cumulative binomial prob < 95% green, >= 99.99% red.
7. Windows: rolling W -> forecast day t uses [t-W, t-1], T = N - W forecasts. Expanding with minimum m -> t = m..N-1 uses r[0:t], T = N - m.
8. Portfolio: daily-rebalanced fixed weights r_p = sum w_i r_i (simple returns). Buy-and-hold: V_t = sum n_i P_{i,t}, weights drift. HS on current holdings: P&L_s = sum_i position_i r_{i,s} (today's positions, historical returns).
9. Returns: simple P_t/P_{t-1} - 1; log ln(P_t/P_{t-1}); log portfolio return = ln(sum w e^{r_i}), not the weighted log average (unless the text says average).

## Conventions and defaults when the task is silent
- Sort by date ascending before any return or window. Prices -> returns from ADJUSTED close (splits/dividends) if both exist.
- Long format (date, symbol, ...): pivot to wide (pivot_table if duplicates), align on common dates (inner join, listwise), then difference. Counts: per-symbol rows, aligned dates, returns = aligned - 1.
- pandas pct_change: pass fill_method=None (older default padded NaN, fabricating zero returns).
- Missing values: listwise deletion; never forward-fill returns; forward-filling prices only if stated (creates zero returns = stale).
- Non-trading days: filter against the SUPPLIED calendar (string-compare canonical YYYY-MM-DD); do not generate holidays yourself; weekends are non-trading.
- Duplicate dates: keep LAST occurrence in original file order (vendor correction) unless stated; de-duplicate BEFORE sorting.
- Mixed date formats: parse per value with an explicit format list (%Y-%m-%d, %m/%d/%Y, %d-%b-%y, %Y%m%d); pandas >= 2 infers one format from the first value and silently NaTs the rest. Slash dates are US month-first unless day > 12 or the text says otherwise. Assert zero NaT.
- Outliers: NaN-safe absolute threshold (abs(NaN) > x is False - NaN is not an outlier and is removed later by the missing step). Robust alternative: |r - median| > k * 1.4826 MAD.
- Stale prices: runs of identical closes (>= a stated length) -> treat as missing, not zero returns.
- Invalid prices (<= 0) dropped before logs. Percent-looking returns (|r| > 1 routinely, or "1.2%" strings) -> divide by 100.
- Cleaning audit: record raw count from the file itself; each step counts rows removed among rows still present; follow the stated step order.
- Kupiec p = 1 - c, not c. Violation strict ">" unless the text says ">=".

## Completing missing inputs
- No window given: 250 (Basel) or 252 trading days; expanding start = 1 year.
- Test level: 5% significance -> reject if p-value < 0.05; chi2(1) 5% critical 3.841, chi2(2) 5.991.
- Expected violations = pT.

## Invariants to self-check
- final rows = raw - sum(removed per step), exactly; after cleaning: unique dates, all in calendar, no NaN, no |r| above threshold.
- LR >= 0; p-values in [0,1]; LR_cc = LR_uc + LR_ind; x = 0 still gives finite LR_uc = -2T ln(1-p).
- sum_ij n_ij = T - 1; violation rate = x/T.
- Worst day return <= -VaR_99 <= -VaR_95; worst date as zero-padded ISO string.
- Stress P&L = sum position_i * scenario_return_i.

## Variants to decide
- V1 violation_rule: exceedance. Options: (a) loss > VaR strict; (b) loss >= VaR. Default when silent: (a) - Kupiec 1995.
- V2 window_type: estimation window. Options: (a) rolling fixed; (b) expanding. Default when silent: (a) - Basel 1996 (250 days).
- V3 return_type: portfolio returns. Options: (a) simple, weighted; (b) average of the asset log-returns (kind="log_avg"); (c) log of portfolio gross return (kind="log_portfolio"). Default when silent: (a) - Jorion 2007.
- V4 missing_policy: gaps. Options: (a) listwise drop; (b) forward-fill prices. Default when silent: (a) - Jorion 2007.
- V5 duplicate_keep: duplicate dates. Options: (a) keep last; (b) keep first. Default when silent: (a) - vendor correction practice.
- V6 kupiec_test: POF test. Options: (a) LR chi2(1); (b) exact binomial; (c) normal z. Default when silent: (a) - Kupiec 1995.
- V7 forecast_alignment: out-of-sample days. Options: (a) VaR for day t uses returns t-W..t-1, so test days = N_returns - W; (b) window includes day t (look-ahead, one test day more). Default when silent: (a) - Christoffersen 2012.
- V8 ind_pooled_rate: pooled hit rate in LR_ind. Options: (a) (n01+n11)/(T-1) from transition counts; (b) x/T. Default when silent: (a) - Christoffersen 1998.
- V9 rolling_refit: model in the rolling loop. Options: (a) refit on each window; (b) fit once, filter forward. Default when silent: (a) - Christoffersen 2012; (b) only when stated.
- V10 hs_input_sign: series fed to an HS helper. Options: (a) returns (helper negates the left tail); (b) losses with a right-tail quantile computed directly. Default when silent: (a) - Jorion 2007; never pass losses to a returns helper.
- V11 library_migration: old-API code. Options: (a) translate statement by statement, preserving outputs and names; (b) rewrite. Default when silent: (a) - Polars 1.0 upgrade guide.

## Pitfalls
- Look-ahead: including day t in its own VaR window (expanding window r[:t+1]).
- Off-by-one: N - W + 1 forecasts, or starting the expanding test at m-1.
- Kupiec with ln(0) -> NaN; using chi2.cdf instead of sf; p-values reported as 1 - p.
- Christoffersen transitions computed on T (wrapping) instead of T-1 pairs; pi11 undefined when n10+n11 = 0 (0 ln 0 = 0 makes the term vanish).
- Single pd.to_datetime on mixed formats; comparing Timestamp to string calendar.
- fillna(0) before the outlier/missing steps (hides missing rows, shifts counts).
- keep='first' instead of last; sorting before de-duplicating changes which row survives.
- Per-asset dropna (misaligned series) instead of joint.
- Returning numpy int64/bool in JSON (cast to int/bool/float).

- Compare calendar dates as 'YYYY-MM-DD' strings on both sides (INPUTS may parse the calendar to datetime64 and nothing matches): clean_returns_with_calendar.
- A date-only end bound includes the whole day: t < end + 1 day (between_inclusive); t <= end at midnight drops that day's intraday rows.
- Code-migration deliverables: RUN the migrated function so every file it writes exists (polars_migrate_and_run), Polars warnings treated as errors.
## Sources
Kupiec (1995); Christoffersen (1998); BCBS (1996) Supervisory framework for backtesting; Jorion, Value at Risk ch.6; Campbell (2005) review of backtesting; McNeil-Frey-Embrechts (2015) ch.9; pandas documentation (to_datetime, pct_change).
