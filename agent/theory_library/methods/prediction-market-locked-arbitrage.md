method: prediction-market-locked-arbitrage
family: event-studies-and-filings
keywords: prediction market, cross-venue, dislocation, locked edge, binary contract, yes no, kalshi, polymarket, arbitrage, resolution, capital reservation, funding charge, source sequence
---
## Formulas
1. Price p in (0,1) per $1 payout. Direction A: YES on venue X + NO on Y; cost C = p_yes_X + p_no_Y; B swaps venues.
2. Gross edge per unit = 1 - C; net = gross - fees (per-leg schedule, e.g. k C P(1-P) rounded up to the cent).
3. Currency fields scale by qty: gross_cost = qty C, fees = qty fee, gross_pnl = qty(1 - C), net = gross - fees - funding.
4. Reserved capital qty C until resolution; funding = rate x reserved x days/365.
5. Daily equity carried forward to the last resolution; r = E_t/E_{t-1} - 1; std ddof=1.
## Variants to decide
- V1 clean_order: cleaning sequence. Options: (a) dedup on max sequence, then status/active/price filters; (b) filter then dedup. Default when silent: (a) - event-sourcing latest-revision rule.
- V2 price_bounds: valid price range. Options: (a) open interval 0 < p < 1; (b) closed. Default when silent: (a) - Wolfers-Zitzewitz 2004.
- V3 direction_tie: equal-edge directions. Options: (a) label sorting first; (b) larger capacity. Default when silent: (a) - deterministic tie rule.
- V4 release_timing: capital release. Options: (a) release resolutions at or before the snapshot first; (b) after allocation. Default when silent: (a) - cash-accounting practice.
## Invariants
- Payoff 1 per paired unit; net <= gross; cash never negative.
## Pitfalls
- Per-unit edge written where qty-scaled currency is asked; pandas between(inclusive="neither").
- Rewriting the event loop under a short budget: locked_edge_backtest after clean_venue_snapshots.
## Sources
- Wolfers-Zitzewitz 2004; Kalshi fee schedule; Manski 2006.
