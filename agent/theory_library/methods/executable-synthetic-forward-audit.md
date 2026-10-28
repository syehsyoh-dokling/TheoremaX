method: executable-synthetic-forward-audit
family: implied-vol-and-parity
keywords: put-call parity audit, synthetic forward bid, synthetic forward ask, executable forward, implied borrow rate, forward too high, forward too low, parity violation, violation amount, stale quote, crossed quote, one-sided quote, missing pair, quote cleaning
---
## Formulas
1. T = (expiry_ts - quote_ts) in seconds / (365 x 86400) (ACT/365 on exact UTC timestamps); DF = exp(-r T).
2. F_bid = K + (C_bid - P_ask)/DF; F_ask = K + (C_ask - P_bid)/DF; F_mid = (F_bid + F_ask)/2.
3. Theoretical forward F_th = S exp((r - q - b) T), b = reference borrow.
4. Implied borrow b_impl = r - q - ln(F_mid / S)/T.
5. violation = max(F_bid - F_th, F_th - F_ask, 0); label too_high when F_th < F_bid, too_low when F_th > F_ask, only if violation > threshold.
## Variants to decide
- V1 flag_order: overlapping flags. Options: (a) one_sided, then crossed, then stale, each row counted once; (b) count every flag independently. Default when silent: (a) - task rule order as listed.
- V2 missing_pair_unit: unmatched count. Options: (a) clean quotes without a clean opposite type at the same expiry and strike; (b) (expiry, strike) keys. Default when silent: (a) - quote-level audit wording.
- V3 threshold_test: violation threshold. Options: (a) strict >; (b) >=. Default when silent: (a) - written formula.
- V4 ttm_clock: T. Options: (a) exact seconds between timestamps /(365 x 86400); (b) calendar days/365. Default when silent: (a) - ACT/365 between timestamps.
## Invariants
- F_bid <= F_mid <= F_ask; violation 0 iff F_th inside the band; violation rows subset of audit rows.
- excluded counts + clean = total rows; worst violation = max of audit column.
## Pitfalls
- Toolkit year_fraction on a pandas Series of tz-aware timestamps crashes: compute T column-wise with (exp - quote).dt.total_seconds().
- Suffix columns after merge (_call/_put): keep one expiry_ts and strike key column before sorting.
- Mixing legs: bid bound uses call bid and put ask.
## Sources
- Hull, Options Futures and Other Derivatives (ch. 11); Cboe put-call parity / synthetic forward practice.
