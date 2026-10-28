family: day-count-and-calendars
category: fixed-income
keywords: day count, act/360, act/365, act/365f, act/act, 30/360, 30e/360, bond basis, money market basis, business day, modified following, following, spot lag, spot date, holiday calendar, value date, settlement, t+2, stub, roll date, end of month, accrual, year fraction, payment schedule, unadjusted, act/365.25, 30/360 us, 30_360_us, 30u/360, broken date, delivery date, calendar months
---
## Worked frame
DIKETAHUI: one row per date pair or trade [M] (ISO dates, basis, tenor, frequency), holidays [H], spot lag (business days). Check: every date parsed, holidays as dates, start < end.
DITANYAKAN: per row [M]: days, year fraction (years), adjusted dates (ISO); schedule one row per period [n]; accrued (currency).
RUMUS: card Core 1-4 (day counts), 5-6 (rolls), 7 (spot); toolkit year_fraction.
JAWAB: 1) parse dates [M]. 2) calendar set from [H]. 3) spot date [M]. 4) schedule: anchor + k months, then roll -> [n+1] dates, [n] periods. 5) year fractions [n] or [M]. 6) amount = notional x decimal rate x fraction -> currency; dates -> ISO. 7) check: dates are business days, sorted, unique; fractions sum ~ tenor; then write.

## Core formulas
Days = date2 - date1 (start inclusive, end exclusive). d1,d2 = day-of-month; Y,M = year, month.
1. ACT/360: yf = days/360. ACT/365F (Fixed): days/365 even in leap years. ACT/365.25 only if stated.
2. ACT/ACT ISDA: days falling in leap years/366 + days in non-leap years/365 (split at Jan 1).
3. ACT/ACT ICMA (bonds): yf = days_accrued / (freq * days_in_coupon_period); a full regular period = 1/freq exactly.
4. 30/360 generic: yf = [360(Y2-Y1) + 30(M2-M1) + (D2-D1)] / 360 with adjusted D1, D2:
   - 30/360 US bond basis (ISDA 2006 "30/360"): D1 = min(d1,30); if D1 == 30 and d2 == 31 then D2 = 30.
   - 30/360 US SIA (US corporates/agencies): as above plus: if d1 is last day of Feb, D1 = 30; if both d1 and d2 are last day of Feb, D2 = 30.
   - 30E/360 (Eurobond): D1 = min(d1,30), D2 = min(d2,30) unconditionally.
   - 30E/360 ISDA: also last day of Feb -> 30 (except D2 on the maturity date).
5. Business-day rules on a date that is not a good day: Following = next good day; Modified Following = following unless that crosses into the next month, then preceding; Preceding; Unadjusted = keep date (accrual may still use unadjusted dates).
6. End-of-month rule: if the start/anchor date is the last business day of its month, every rolled date is month-end (Feb 28 -> Mar 31, not Mar 28). Jan 31 + 1M = Feb 28/29 (clip, no overflow).
7. FX spot date: add N good business days (N = spot lag) to trade date, a good day being open in BOTH currencies of the pair (and USD for crosses at settlement). Forward delivery = spot date + n months, then roll (Modified Following + EOM is market standard; use Following if the task says so). Broken date: days_from_spot = delivery - spot.
## Conventions and defaults when the task is silent
- Money-market (deposit/LIBOR/OIS float) basis: ACT/360 for USD, EUR, JPY, CHF, SEK, NOK, DKK; ACT/365F for GBP, AUD, NZD, CAD, HKD, SGD, ZAR, PLN. A file column with per-ccy convention ALWAYS overrides (ISDA 2006; OpenGamma "Interest rate instruments and market conventions").
- Swap fixed legs: legacy USD LIBOR IRS semi 30/360 vs 3M ACT/360; USD SOFR OIS annual ACT/360 both legs; EUR annual 30/360 vs 6M Euribor ACT/360 (1Y: 3M); GBP SONIA annual ACT/365F; JPY semi ACT/365F; CHF annual 30/360.
- Government bonds: UST, Gilts semi ACT/ACT ICMA; Bunds/OATs annual ACT/ACT ICMA; JGB semi ACT/365F; US corporates semi 30/360 SIA. UST settle T+1.
- Spot lags: FX T+2; USD/CAD, USD/TRY, USD/RUB, USD/PHP T+1. GBP SONIA and GBP deposits T+0; USD SOFR, EUR, JPY, CHF swaps T+2.
- Discount-curve time vs accrual: accrual fractions use the leg's day count; curve time t (for exp(-z t)) uses the curve's own basis (commonly ACT/365F). Never mix them unless the task defines one time axis.
- Schedules: generate BACKWARD from maturity in frequency steps (front stub, short by default); forward generation gives a back stub. Accrual start of current period = last schedule date <= settlement.
- Cues: "bond basis" = 30/360; "money-market basis" = ACT/360; "A/365" = ACT/365F; "actual/actual" for Treasuries = ICMA; "ISDA" = calendar-year split; percent-looking rates (5.25) -> divide by 100.
## Completing missing inputs
- No holiday file: weekends only (Mon-Fri) calendar; state it. Holidays given as strings: parse to date before comparing.
- Missing spot lag: FX 2 (1 for CAD vs USD); deposits: currency's own lag above.
- Missing frequency for a USD bond: semi-annual; EUR govt: annual.
- Year fraction for a tenor label with no dates: 1M = 1/12, 3M = 0.25, 6M = 0.5, 1W = 7/365 (or /360 for ACT/360 ccy), ON = 1 day.
- Stub coupon = full coupon * (stub days / regular period days) under ICMA; under 30/360 or ACT/360 just compute yf of the actual stub.
## Invariants to self-check
- Sum of accrual fractions over a regular schedule ~ tenor (30/360 exact: 0.5 per semi period; ACT/360 slightly > tenor).
- ACT/360 yf > ACT/365F yf for the same dates (ratio 365/360).
- Every payment/spot date is a good business day in the relevant calendar(s); no rolled date crosses month end under Modified Following.
- Accrued interest = 0 on a coupon date (not a full coupon); accrued grows linearly within the period.
- Dates written as ISO YYYY-MM-DD; schedules sorted ascending, no duplicates.
## Variants to decide
- V1 act_act: ACT/ACT flavour. Options: (a) ICMA (per coupon period) for bonds; (b) ISDA (calendar-year split) for swaps. Default when silent: (a) - ICMA Rule 251 for bonds; ISDA 2006 4.16(b) when "ISDA" is said.
- V2 thirty_360: 30/360 flavour. Options: (a) 30/360 US bond basis; (b) 30E/360 Eurobond; (c) 30E/360 ISDA. Default when silent: (a) - ISDA 2006 4.16(f) for USD; (b) for EUR bonds.
- V3 bdc: business day convention. Options: (a) Modified Following; (b) Following; (c) Preceding. Default when silent: (a) - ISDA 2006.
- V4 schedule_direction: schedule generation. Options: (a) backward from maturity, short front stub; (b) forward from start, back stub. Default when silent: (a) - ISDA practice.
- V5 eom_rule: roll rule for month-end starts. Options: (a) end-of-month roll; (b) same day-of-month. Default when silent: (a) - ISDA 2006 EOM convention.
- V6 accrual_dates: accrual period endpoints. Options: (a) adjusted dates (swap legs); (b) unadjusted dates (bonds). Default when silent: (a) - ISDA 2006 for swaps; bonds accrue unadjusted.
- V7 act365_flavour: ACT/365 label. Options: (a) ACT/365F (fixed 365); (b) ACT/365.25 (only when stated, often a curve time axis); (c) ACT/365L (leap-aware). Default when silent: (a) - ISDA 2006 4.16(d); implement any other stated basis by hand if a helper rejects it.
- V8 month_addition: tenor dates. Options: (a) anchor + k calendar months with month-end clip, then the business-day rule; (b) fixed 30 days per month. Default when silent: (a) - ISDA 2006.
- V9 settlement_calendar: good day for FX value dates. Options: (a) open in both currencies of the pair (plus USD for crosses); (b) terms-currency calendar only. Default when silent: (a) - Clark 2011 ch. 1.

## Pitfalls
- Counting both endpoints (days + 1) or using business days instead of calendar days in a day count.
- 30/360 US: applying D2 = 30 when d2 = 31 but D1 < 30 (wrong; D2 stays 31 then).
- Using 365 for USD/EUR money-market rates, or 360 for GBP.
- Measuring yf from trade date when the curve/forward is anchored at spot date (and vice versa).
- Rolling with Following at month end instead of Modified Following (or ignoring the task's stated rule).
- Adding months by adding 30 days; numpy/pandas DateOffset vs dateutil relativedelta both clip month-end, but repeated addition from a clipped date drifts (Feb 28 + 1M = Mar 28): always roll from the anchor (anchor + k months).
- Spot date that skips a holiday in only one of the two currencies.
- Leap-year handling in ACT/ACT ISDA (split at Jan 1, not by the period's year).
## Sources
ISDA 2006 Definitions s4.16; ICMA Rule 251; SIFMA (SIA) Standard Securities Calculation Methods; OpenGamma (Henrard) Interest Rate Instruments and Market Conventions Guide 2013; Hull, Options Futures and Other Derivatives ch. 4, 6; Clark, Foreign Exchange Option Pricing ch. 1 (FX dates).
