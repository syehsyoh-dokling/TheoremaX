method: fx-forward-mtm-attribution
family: fx-forwards-carry-xccy
keywords: fx forward mtm, mark-to-market, forward portfolio valuation, pnl attribution, spot delta, carry, residual pnl, net delta exposure, delta per currency, pnl in quote currency, pnl usd, historical cross rate, fair forward rate
---
## Formulas
1. Fair forward for the trade: F_now = CIP forward on today's mid for the trade tenor (or remaining days to delivery if dates exist).
2. MTM quote ccy (long base = +1, short = -1): V_Y = sign N_base (F_now - K); discounted form V_Y DF_Y(T).
3. To USD: Y = USD -> V; USD/Y quoted -> V / S(USD/Y); Y/USD quoted -> V x S(Y/USD); crosses via triangulated spot.
4. Attribution (USD): spot = sign N (S_now - S_hist) to USD; carry = sign N [(F_now - S_now) - (K - S_hist)]; residual = total - spot - carry. Historical crosses triangulated from historical USD legs.
5. Net delta per ccy (USD): base leg +sign N_base in base ccy, quote leg -sign N_base K in quote ccy, each to USD at spot; sum per ccy.
## Variants to decide
- V1 mtm_discounting: forward MTM. Options: (a) discounted (F - K) N DF_Y; (b) undiscounted (F - K) N. Default when silent: (a) - Hull OFOD ch. 5; (b) when no discount curve is given and PnL is stated in quote ccy.
- V2 forward_horizon: tenor for F_now. Options: (a) remaining time to delivery; (b) full original tenor (no ageing). Default when silent: (a) - Hull OFOD ch. 5.
- V3 carry_definition: carry part. Options: (a) change in forward points; (b) rate-differential accrual N S (r_Y - r_X) dt. Default when silent: (a) - desk practice (Clark 2011).
- V4 delta_basis: exposure. Options: (a) undiscounted notional legs at spot; (b) PV-discounted legs. Default when silent: (a) - cash-flow delta convention.
## Invariants
- spot + carry + residual = total exactly; long base gains when spot rises.
## Pitfalls
- Inverted quote direction flips PnL and delta signs; multiply vs divide in USD conversion.
## Sources
- Hull OFOD ch. 5; Clark (2011) ch. 1.
