method: fx-forward-points-cross-triangulation
family: fx-forwards-carry-xccy
keywords: forward points, outright forward, pip multiplier, covered interest parity, cross rate, synthetic cross, triangulation, bid ask cross, forward cross rate, implied yield, cip deviation, deposit bid offer, stale quote, usd legs
---
## Formulas
1. Pair X/Y (Y per X): F = S (1 + r_Y t_Y)/(1 + r_X t_X), t_c = days/basis_c (ACT/360 or ACT/365 per ccy), days = spot->delivery or the stated tenor days.
2. Dealer sides: F_bid = S_bid (1 + r_Y,bid t)/(1 + r_X,ask t); F_ask = S_ask (1 + r_Y,ask t)/(1 + r_X,bid t). Mid-only rates: same factor on S_bid and S_ask.
3. Points = F - S (quote units); pips = points x multiplier (1e4; 1e2 for JPY terms).
4. Crosses: X/USD, Y/USD -> X/Y bid = XUSD_bid/YUSD_ask, ask = XUSD_ask/YUSD_bid. X/USD, USD/Y -> bid = XUSD_bid x USDY_bid, ask = ask x ask. USD/X, USD/Y -> bid = USDY_bid/USDX_ask, ask = USDY_ask/USDX_bid. Reciprocal (1/ask, 1/bid).
5. Forward cross: same rule on forward legs of the same tenor.
6. CIP deviation (bps) = (F_tri/F_cip - 1) 1e4, F_cip from cross spot and the two ccy rates.
7. Implied yield: r_X = [(S/F)(1 + r_Y t_Y) - 1]/t_X with sides as stated.
## Variants to decide
- V1 deposit_sides: dealer forwards. Options: (a) opposite deposit sides; (b) mid rates on spot bid and ask. Default when silent: (a) - Clark 2011 when bid/offer rates exist.
- V2 tenor_time: year fraction. Options: (a) actual days spot->delivery over ccy basis; (b) stated tenor days over ccy basis; (c) months/12. Default when silent: (a) - Hull OFOD ch. 5.
- V3 cross_mid: cross mid. Options: (a) product or ratio of leg mids; (b) average of cross bid and ask. Default when silent: (a) - Clark 2011.
## Invariants
- bid < mid < ask on every leg and cross; cross spread wider than each leg.
- CIP-built crosses give deviation near 0; sign(points) = sign(r_Y - r_X).
## Pitfalls
- Drop stale rows (bid >= ask) first; normalise labels (EUR/USD vs EURUSD).
- Leg missing as USD/Y: invert Y/USD; no mid column: mid = (bid + ask)/2.
## Sources
- Hull OFOD ch. 5; Clark (2011) FX Option Pricing ch. 1; Du-Tepper-Verdelhan (2018).
