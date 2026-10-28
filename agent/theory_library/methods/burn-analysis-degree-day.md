method: burn-analysis-degree-day
family: weather-derivatives
keywords: burn analysis, historical burn, hdd option, cdd option, degree day index, monthly hdd, yearly hdd, tick value, cme hdd, ghcn-daily, tmax tmin, base temperature, burn price, expected payoff
---
## Formulas
1. Data: GHCN-Daily CSV is headerless (station, date YYYYMMDD, element, value, m_flag, q_flag, s_flag); read with header=None and names; TMAX/TMIN in tenths of C (/10).
2. Filter station, contract month, year range; pivot TMAX and TMIN per date; drop days missing either (or with a q_flag if told).
3. T_avg = (TMAX + TMIN)/2; HDD_d = max(base - T_avg, 0); base 65 F -> (65 - 32) x 5/9 C when working in C.
4. Yearly index I_y = sum of HDD_d over the month; payoff_y = tick x max(I_y - K, 0) (put: K - I_y; cap if given).
5. Burn price = DF x mean_y payoff_y, DF = exp(-r tau).
## Variants to decide
- V1 discount_tenor: tau for DF. Options: (a) days in the contract month / 365 (period start to period end); (b) 1/12; (c) to an explicit payment date ACT/365. Default when silent: (a) - ACT/365 on the accrual period (Jewson-Brix 2005).
- V2 year_weighting: history. Options: (a) equal weight every complete year; (b) detrended / recent-weighted. Default when silent: (a) - Jewson-Brix 2005.
- V3 temp_unit: computation unit. Options: (a) convert base to the data unit (C for GHCN); (b) convert data to F. Default when silent: (a) - task data unit.
## Invariants
- HDD >= 0; monthly index is a sum (hundreds in winter), clean days <= years x days in month.
- burn_price = DF x mean payoff; HDD call payoff increases when colder.
## Pitfalls
- Reading the first data row as header: columns get named by values.
- DF from the number of sample years instead of the contract tenor.
## Sources
- CME Group weather futures and options specs; Jewson and Brix (2005); NOAA GHCN-Daily readme.
