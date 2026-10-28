family: fx-forwards-carry-xccy
category: fx
keywords: fx forward, covered interest parity, cip, forward points, pips, outright, cross rate, triangulation, bid, ask, offer, implied yield, cross-currency basis, xccy, mtm reset, resettable, carry trade, forward hedge, fx swap, roll, ndf, garman-kohlhagen, risk reversal, butterfly, 25 delta, funding rate, synthetic cross, pnl attribution, net delta, implied basis, par basis, notional reset, broken date, implied yield, forward hedge overlay
---
## Worked frame
DIKETAHUI: spot per pair [P] (quote ccy per 1 base), rates per ccy (percent -> decimal), days, trades one row each [M] (pair, direction, base notional, strike). Check: quote direction, pair labels.
DITANYAKAN: per trade or tenor [M]: outright, points (pips), basis (bps), MTM (quote ccy); exposure one key per ccy.
RUMUS: card Core 1-6; toolkit money_market_year_fraction, cip_forward, forward_points, cross_rate, cip_deviation_bps, forward_mtm.
JAWAB: 1) split pair -> base, quote [M]. 2) year fraction per ccy [M]. 3) forward [M]. 4) points -> pips by pair multiplier [M]. 5) basis or MTM [M]. 6) convert to report ccy; sum per ccy. 7) check: bid < ask, sign(points) = sign(r_quote - r_base), parts sum to total; then write.

## Core formulas
Pair X/Y = units of Y (terms/quote) per 1 X (base). d = terms (domestic) ccy, f = base (foreign).
1. CIP, money-market: F = S (1 + r_Y t_Y)/(1 + r_X t_X), t_c = days(spot -> delivery)/basis_c (per-ccy day count). Continuous: F = S exp((r_d - r_f)T). Points = F - S; pips = points * 1e4 (1e2 for JPY-terms pairs); outright = spot + pips/multiplier.
2. Dealer sides: F_bid = S_bid (1 + r_Y,bid t)/(1 + r_X,offer t); F_offer = S_offer (1 + r_Y,offer t)/(1 + r_X,bid t). No deposit bid/offer -> same mid factor on each spot side.
3. Crosses via USD: X/USD and Y/USD -> X/Y bid = XUSD_bid / YUSD_offer, offer = XUSD_offer / YUSD_bid. X/USD and USD/Y -> bid = bid*bid, offer = offer*offer. USD/X and USD/Y -> X/Y bid = USDY_bid / USDX_offer, offer = USDY_offer / USDX_bid. Inverse quote: (1/offer, 1/bid). Mid cross = product/ratio of mids.
4. Implied yield (invert CIP, spot mid): r_X = [(S/F)(1 + r_Y t_Y) - 1]/t_X; r_Y = [(F/S)(1 + r_X t_X) - 1]/t_Y; pair F side with the opposite USD rate side as the task states.
5. Basis: F_mkt = S (1 + r_Y t_Y)/(1 + (r_X + b) t_X) or continuous b = [ln(F_cip/F_mkt)]/tau; CIP deviation bps = (F_mkt - F_cip)/F_cip * 1e4 (sign as specified). Negative EUR/JPY basis = synthetic USD via FX swap costs more than USD OIS.
6. Forward MTM (long N base at K): V_Y = N (F_t(T) - K) DF_Y(t,T) (desk) or undiscounted if the task says so; to USD at spot. Attribution: spot = N(S_now - S_then), carry = N[(F_now - S_now) - (F_then - S_then)] or as defined, residual = total - spot - carry (sums exactly).
7. FX swap roll: near leg at spot +/- near points, far at spot +/- far points, SAME side (bid or offer) on both legs, chosen by the far leg; swap points = far - near.
8. NDF (USD/LOC, USD notional N, rate K): USD settlement = N (Fix - K)/Fix to the USD buyer; fixing = settlement - 2 business days.
9. MTM (resettable) xccy swap, USD notional fixed: non-USD notional N_i = N_USD / FX(reset_i) (FX = USD per non-USD); reset flow N_i - N_{i-1} paid in non-USD at reset (sign by holder perspective); final exchange N_last; non-USD flows -> USD at FX forward of payment date -> discount on USD OIS (USD collateral). Par basis = all-in spread s: PV = 0, s = s_contract - PV/(sum N_i tau_i FX_i DF_i).
10. Garman-Kohlhagen: C = S e^{-r_f T} N(d1) - K e^{-r_d T} N(d2); d1 = [ln(S/K) + (r_d - r_f + s^2/2)T]/(s sqrt T); C - P = S e^{-r_f T} - K e^{-r_d T}. Deltas: spot e^{-r_f T}N(d1); forward N(d1); premium-adjusted spot e^{-r_f T}(K/F)N(d2) (put: -e^{-r_f T}(K/F)N(-d2)); pi-delta = delta - V/S. Vega per 1% = vega/100.
11. Smile from quotes: s_25C = ATM + BF + RR/2, s_25P = ATM + BF - RR/2 (smile-strangle approx); strike from target delta by brentq.
12. Carry return (long foreign, USD investor): R = S_t/S_{t-1} - 1 (S = USD per foreign) + (r_f - r_USD) * days/basis; forward-hedged ~ R_local + (F/S - 1) with F/S - 1 ~ (r_USD - r_f)t.
## Conventions and defaults when the task is silent
- Money-market basis: ACT/360 USD EUR JPY CHF SEK NOK DKK; ACT/365 GBP AUD NZD CAD. Spot T+2 (USD/CAD T+1), joint calendar of both ccys; delivery = spot + n months, Modified Following (Following if stated). Year fractions from SPOT date, not trade date.
- Rates in percent -> /100; points quoted in pips -> /multiplier.
- Premium-adjusted delta when premium is paid in BASE ccy (e.g. USDJPY with USD premium); unadjusted when premium in terms ccy (EURUSD, USD premium). Option T = calendar days/365. ATM = delta-neutral straddle K = F exp(s^2 T/2) (unadjusted) unless ATMF stated.
- Premium units (Clark): terms pips V; base % = V/S; terms % = V/K; base pips = V/(S K). Follow the schema if it redefines them.
- Carry portfolio: signal at close of date t, trade from t+1 (lag); long top-k high-rate, short bottom-k; monthly rates forward-filled; hedge book stays stale until its own roll.
- Perpetual funding: positive rate -> longs pay shorts; long spot + short perp receives it; 3 settlements/day -> annualise x1095 (x365*3); funding 0.0001 = 1 bp.
## Completing missing inputs
- Broken date: interpolate rates linearly in days (flat before first tenor) or ln F linearly in time from spot (at spot date) to tenor nodes.
- Initial non-USD notional = N_USD / inception spot. Historical fixings for coupons fixed on/before valuation; projected forwards after.
- Historical cross = triangulate historical USD legs exactly like today's.
## Invariants to self-check
- bid < mid < offer; cross spread wider than each leg; triangle EURUSD*USDJPY/EURJPY ~ 1 within spreads.
- CIP-built forwards give zero CIP deviation; sign(points) = sign(r_terms - r_base).
- GK parity to 1e-10; call delta in (0, e^{-r_f T}); PA delta < spot delta for calls.
- PnL components sum to total; leg PVs sum to trade PV; stress PnL = stressed PV - base PV.
- Carry: long leg = highest-rate ccys (sign check); VaR/CVaR sign per convention (return quantile negative, or loss positive), CVaR beyond VaR.
## Variants to decide
- V1 cip_compounding: forward from rates. Options: (a) simple money-market F=S(1+r_d t_d)/(1+r_f t_f), each ccy's basis; (b) continuous F=S e^{(r_d-r_f)T}. Default when silent: (a) - Clark 2011.
- V2 delta_type: FX option delta. Options: (a) spot delta, premium-adjusted only if premium in base ccy; (b) forward delta. Default when silent: (a) - Reiswich-Wystup 2010 (forward delta beyond 1Y).
- V3 atm_def: ATM strike. Options: (a) delta-neutral straddle K=F e^{0.5 s^2 T}; (b) ATMF K=F; (c) ATM spot. Default when silent: (a) - Reiswich-Wystup 2010 / Clark 2011.
- V4 rr_bf: smile quotes to vols. Options: (a) RR=s25C-s25P, s25=ATM+BF+-RR/2 (smile strangle); (b) broker market strangle. Default when silent: (a) - Clark 2011.
- V5 carry_lag: carry portfolio timing. Options: (a) signal at t, trade from t+1; (b) same day. Default when silent: (a) - Lustig-Verdelhan 2007 practice.
- V6 broken_date_fx: forward between tenor nodes. Options: (a) interpolate forward points linearly in days from spot (0 at spot); (b) interpolate deposit rates, then CIP; (c) ln F linear in time. Default when silent: (a) - Clark 2011 ch. 1; follow a stated interpolation convention.
- V7 forward_mtm_discount: forward MTM. Options: (a) N (F_now - K) DF_quote(T); (b) N (F_now - K) undiscounted. Default when silent: (a) - Hull OFOD ch. 5; (b) when no discount curve is given.
- V8 xccy_basis_form: implied basis. Options: (a) continuous spread on the non-USD OIS curve, b = -ln(F_mkt/F_cip)/T; (b) simple spread on the non-USD deposit rate. Default when silent: (a) - Fujii-Shimada-Takahashi 2010.
- V9 mtm_reset_flow: GBP-type receiver at a notional reset. Options: (a) flow N_{i-1} - N_i at the reset date, final + N_last; (b) no intermediate exchange (non-MTM swap). Default when silent: (a) - standard MTM CCS (Tuckman-Serrat ch. 18).

## Pitfalls
- Inverted quote direction flips every sign; JPY pips at 1e2.
- One day-count for all ccys; t from trade date; spot date skipping only one calendar.
- Wrong bid/offer pairing in crosses/implied yields; mids used for dealer sides.
- Swap points reported in quote units when pips are asked (or vice versa).
- No e^{-r_f T} in GK delta.
- MTM reset sign flipped (flips FX delta); spread added to USD leg; past flows not excluded.
- Look-ahead (signal-date return); rolling the hedge with the monthly basket.
- Quote units by exact name: 'spot_rate' contains 'rate' (a substring test divides spot by 100) - normalize_quote_book. Crosses: cross_via_usd; spot/broken dates: fx_spot_date, fx_add_tenor; JSON inputs nest real data under a key next to a '_comment' string.
- Snapshot desks in one call: fx_snapshot_desk on the raw JSON (joint-calendar spot, CIP, implied yields, NDF); forward books: fx_forward_book (cross trades marked at the rounded cross forward).
- Pass RAW quote tables to fx_forward_book; read snapshot dates with first_iso_date, never parse a '_comment' sentence.
- MTM xccy: the MTM-leg receiver is paid each notional increase (N_k - N_{k-1}); see mtm_xccy_swap_bundle.
## Sources
Hull OFOD ch. 5, 17; Clark (2011) FX Option Pricing; Reiswich-Wystup (2010) FX smile; Du-Tepper-Verdelhan (2018) CIP deviations; Lustig-Verdelhan (2007) carry; Fujii-Shimada-Takahashi (2010) xccy collateral.
