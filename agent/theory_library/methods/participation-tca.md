method: participation-tca
family: execution-and-microstructure
keywords: participation rate, percentage of volume, pov algorithm, child order, bucket, transaction cost analysis, tca, implementation shortfall, arrival mid, effective spread, realized spread, temporary impact, market vwap, epoch ms
---
## Formulas
1. Buckets [start + k dt, start + (k+1) dt), n = ceil((end - start)/dt); trades inside are eligible.
2. Child q_b = min(remaining, rho V_b, max_child); remaining after each bucket = target - cumulative fills.
3. Fill at the touch of the benchmark quote (buy ask, sell bid); market VWAP over eligible trades.
4. IS bps = s(avg fill - arrival mid)/arrival mid x 1e4; effective = 2s(P - M)/M x 1e4; realized = 2s(P - M_{t+h})/M x 1e4; impact = effective - realized.
5. Future reference time = end + horizon; quote prevailing at that instant.
## Variants to decide
- V1 quote_match: benchmark quote. Options: (a) last quote at or before t; (b) first quote at or after t. Default when silent: (a) - SEC Rule 605 / Bessembinder 2003.
- V2 quote_clock: timestamp used. Options: (a) transaction/exchange time; (b) event/receipt time. Default when silent: (a) - Bessembinder 2003.
- V3 empty_bucket: empty definition. Options: (a) zero eligible trades; (b) zero fill. Default when silent: (a) - Kissell 2013.
- V4 spread_multiplier: spread scaling. Options: (a) full spread 2 x side x (P-M)/M; (b) half-spread. Default when silent: (a) - Bessembinder 2003.
## Invariants
- sum fills <= target; remaining non-increasing; child <= every cap; effective = realized + impact.
## Pitfalls
- int64 ms converted to datetimes then compared with ints; empty strings in int columns.
- Closed bucket ends double-counting boundary trades.
- INPUTS turns every int column whose NAME holds 'time'/'date' (update_id too) into datetime64: call participation_bucket_fills (or epoch_ms_int on each column) first.
## Sources
- Perold 1988; Kissell 2013; Bessembinder 2003; SEC Rule 605.
