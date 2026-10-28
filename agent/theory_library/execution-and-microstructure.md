family: execution-and-microstructure
category: microstructure
keywords: vwap, twap, participation, percentage of volume, pov, child order, bucket, implementation shortfall, arrival price, slippage, bps, effective spread, realized spread, price impact, tca, almgren-chriss, square-root impact, volume profile, u-shaped, intraday, order book, ofi, order imbalance, microprice, schedule, largest remainder, ewma profile, winsorized profile, realized spread, temporary impact, market vwap, binance
---
## Worked frame
DIKETAHUI: trades [n_tr] (time, price, size), quotes [n_q] (time, bid, ask), volume bars [D x B], order (side, quantity, window). Check: stable time sort, bid <= ask, half-open windows.
DITANYAKAN: fills or schedule one row per bucket [B] (integer shares); costs in bps [n_tr] or scalar; profile [B].
RUMUS: card Core 1-2 (VWAP, POV), 3-4 (costs), 6 (Almgren-Chriss), 8-10 (profile, schedule), 11-12 (book).
JAWAB: 1) filter to window, bucket id [n_tr]. 2) aggregate [B]. 3) prevailing mid per trade [n_tr]. 4) child sizes [B], cumulative [B]. 5) signed costs [n_tr] -> scalars. 6) fraction -> bps; shares -> integers. 7) check: fills sum to executed <= target, profile sums to 1, effective = realized + impact; then write.

## Core formulas
1. VWAP = sum(p q)/sum(q) over trades in the window; TWAP = equal child sizes per equal time slice (or mean price over slices).
2. POV child_b = min(remaining, rho*V_b, max_child); remaining_b = target - cumulative executed.
3. Side s=+1 buy, -1 sell. Arrival slippage / implementation shortfall bps = s*(avg_fill - P_arr)/P_arr*1e4 (positive = cost). Full Perold IS adds opportunity cost s*(P_end-P_arr)*unfilled.
4. Spreads (Rule 605 / Bessembinder): effective = 2s(P-M)/M; realized = 2s(P-M_{t+D})/M; impact = 2s(M_{t+D}-M)/M; effective = realized + impact. If "temporary impact" is defined without the 2, effective = realized + 2*impact. Denominator = mid at trade/arrival.
5. Quoted spread = ask-bid; relative = (ask-bid)/mid. Roll: s = 2 sqrt(-cov(dp_t, dp_{t-1})) when cov<0.
6. Almgren-Chriss (N slices, tau=T/N): eta~ = eta - gamma tau/2; (2/tau^2)(cosh(k tau)-1) = lambda sigma^2/eta~ (continuous: k=sqrt(lambda sigma^2/eta)); holdings x_j = X sinh(k(T-t_j))/sinh(kT); trades n_j = x_{j-1}-x_j. E = gamma X^2/2 + eps sum|n_j| + (eta~/tau) sum n_j^2; Var = sigma^2 tau sum x_j^2. lambda=0 -> linear TWAP.
7. Square-root impact: I = Y sigma_daily sqrt(Q/ADV), Y ~ 0.5-1.
8. Volume share s_{d,b} = V_{d,b}/sum_b V_{d,b}. Profiles: mean, per-bar median, EWMA w_L = exp(-ln2 L/H) with L=0 most recent, winsorised mean (per bar across days at P5/P95, numpy linear). Renormalise every profile to sum 1.
9. Profile R2 per day = 1 - sum(s-shat)^2/sum(s-mean s)^2; model score = mean of daily R2 (not pooled).
10. Integer schedule (largest remainder): floor(Q w_b), give leftover units to largest fractional parts, ties to the earlier bar (stable argsort of -frac).
11. OFI (Cont-Kukanov-Stoikov 2014): e = 1{Pb>=Pb'}qb - 1{Pb<=Pb'}qb' - 1{Pa<=Pa'}qa + 1{Pa>=Pa'}qa' (' = previous). Flow form per side: limit - cancel - market; OFI = bid flow - ask flow; depth-weighted by 1/(1+|distance|) if asked.
12. Imbalance = (Vb-Va)/(Vb+Va) in [-1,1]; microprice = (Pa qb + Pb qa)/(qb+qa); mid = (bid+ask)/2.
13. Trade sign (Lee-Ready): P>mid buy, P<mid sell, P=mid tick rule.

## Conventions and defaults when the task is silent
- Time filters half-open [start, end). Benchmark quote: TCA default is the prevailing quote (last at or before t); if text says "first quote at or after", use that. Look up on exchange/transaction time, not receipt/event time.
- A "reference time" field is the target instant (e.g. end + horizon); the matched quote's own timestamps are separate fields.
- Marketable fills: buy at ask, sell at bid; child fully filled at the touch unless a book walk is specified.
- Empty bucket = zero eligible trades; a bucket with trades but zero child is NOT empty.
- US RTH 09:30-16:00; 5-min bars labelled at bar START (resample label=left, closed=left) = 78 bars; include the 16:00 print only if the last bar needs it; drop empty resample buckets before counting complete days; exclude half-days.
- Walk-forward over complete days; estimation window = W complete days strictly before the test/order day.
- Schedule only bars with start strictly after the decision time; renormalise shares over them.
- Rolling signal models: fit scaler/PCA/OLS on rows [t-W, t-1] only; z-score with training mean and population std (StandardScaler, ddof=0); apply the same transform and sign flip to row t.
- Target = forward log mid return ln(m_{t+1}/m_t); last row NaN; never drop rows up front (indices shift).
- Stable sort by time so same-timestamp events keep file order.
- Output: integer epoch ms, fixed-decimal floats, empty string for missing, lowercase true/false; hand-format CSV instead of pandas defaults.

## Completing missing inputs
- No mid column: mid=(bid+ask)/2. No side: infer from order spec; sign conventions from the rules file win.
- Missing target inside a training window: skip that step, do not impute.
- Zero-variance feature in a window: std=1 (column becomes 0).
- Remaining-bar profile sums to 0: uniform weights.

## Invariants to self-check
- sum fills = executed; fill_rate = executed/target in [0,1]; remaining non-increasing and >=0; child <= every cap.
- Sum of bucket trade counts/volumes = window totals; VWAP within [low, high]; bid <= ask.
- effective - realized = impact (or 2*temp impact) exactly.
- Schedule non-negative integers summing to Q; cumulative ends at Q; profiles sum to 1; daily R2 <= 1.
- Liquidation trajectory monotone (no re-buy); x_0=X, x_N=0.
- corr(PC1_train, anchor) >= 0 after flip; predictions invariant to PC sign.

## Variants to decide
- V1 vwap_benchmark: VWAP window. Options: (a) interval VWAP over the order horizon; (b) full-day VWAP. Default when silent: (a) - Kissell 2013.
- V2 arrival_price: IS benchmark. Options: (a) prevailing mid at decision time; (b) first quote/trade after decision. Default when silent: (a) - Perold 1988.
- V3 is_sign: shortfall sign. Options: (a) side x (exec - arrival)/arrival x 1e4, positive = cost; (b) opposite sign. Default when silent: (a) - Perold 1988 / Kissell 2013.
- V4 effective_spread: spread measure. Options: (a) 2 x side x (P - M)/M; (b) side x (P - M)/M (half-spread). Default when silent: (a) - Bessembinder 2003.
- V5 realized_horizon: realized-spread / impact horizon. Options: (a) 5 minutes; (b) 1 minute; (c) as stated. Default when silent: (a) - Bessembinder 2003.
- V6 zscore_ddof: rolling feature standardisation. Options: (a) population std ddof=0 (StandardScaler); (b) ddof=1. Default when silent: (a) - scikit-learn StandardScaler.
- V7 ewma_profile_order: EWMA profile weights. Options: (a) lag 0 = most recent day gets the largest weight, reversed onto oldest-first rows; (b) weights in row order. Default when silent: (a) - RiskMetrics 1996.
- V8 order_day_grid: bars eligible for the schedule. Options: (a) the order day's full session grid, bars starting after the order time; (b) only bars of complete days. Default when silent: (a) - Kissell 2013.
- V9 quote_match: benchmark quote lookup. Options: (a) prevailing quote at or before t; (b) first quote at or after t. Default when silent: (a) - SEC Rule 605 / Bessembinder 2003.
- V10 bucket_count: number of participation buckets. Options: (a) ceil((end - start)/bucket); (b) floor. Default when silent: (a) - half-open buckets cover the whole window.

## Pitfalls
- Closed bucket ends (double count boundary trades); right-labelled bars shift every bar.
- event_time vs transaction_time; prevailing vs next quote.
- Sell-side sign not flipped (or flipped when the rules write raw formulas).
- Realized spread divided by future mid; 1e4 vs 2e4 multipliers swapped.
- EWMA weights reversed (most weight on the oldest day); winsorising across bars instead of across days.
- round() instead of largest remainder (sum mismatch); ties to later bar.
- Re-testing the sign on the already-flipped series so the test row is never flipped.
- IC computed with the NaN last forward return gives NaN (JSON null): nan_safe_ic masks pairs; rolling_pc1_ols_forecast for walk-forward PCA+OLS.
- Locked-edge allocation sizes qty on trading cash; daily funding is charged in equity, not in the cash later trades use; signal counts exclude below-threshold rows.

## Sources
Perold 1988; Almgren-Chriss 2000; Almgren et al. 2005; Kissell-Glantz 2003; Bessembinder 2003; SEC Rule 605; Roll 1984; Lee-Ready 1991; Cont-Kukanov-Stoikov 2014; Stoikov 2018 (microprice); Bialkowski-Darolles-Le Fol 2008.
