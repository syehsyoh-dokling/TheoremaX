family: event-studies-and-filings
category: nlp-on-finance
keywords: event study, abnormal return, car, caar, bhar, market model, estimation window, patell, bmp, corrado, earnings surprise, sue, 13f, amendment, form 4, insider, 8-k, 10-k, xbrl, edgar, sentiment, loughran-mcdonald, tone, fomc, tf-idf, etf holdings, prediction market, funding rate, basis, kolari-pynnonen, generalized rank, restatement, natureofownership, 10b5-1, inline xbrl, guidance, locked edge, redemption, crowding
---
## Worked frame
DIKETAHUI: stock returns [T x N], market returns [T], events one row each [E] (id, date); filings branch: one row per filing. Check: sorted dates, unique (id, date), event date -> row position.
DITANYAKAN: per event [E]: alpha, beta, CAR (decimal); per relative day [L2]: AAR, CAAR; test statistics (scalars).
RUMUS: card Core 1 -> 2 or 3; toolkit market_model, abnormal_returns, car, patell_standardized, bmp_test.
JAWAB: 1) align stock and market [T]. 2) per event: estimation [L1] and event [L2] slices by row position. 3) fit -> alpha, beta, sigma. 4) AR [E x L2]. 5) CAR [E], AAR and CAAR [L2]. 6) tests -> scalars; decimal -> percent if asked. 7) check: CAR = sum AR, windows disjoint, no padded days; then write.

## Core formulas
1. Market model OLS on estimation window: AR_t = R_t-(a+b Rm_t); CAR=sum AR; CAAR_t=mean_i AR_it; BHAR=prod(1+R)-prod(1+R_bench).
2. s_i^2=SSR/(L-2). Patell SAR=AR/(s_i sqrt(C_t)), C_t=1+1/L+(Rm_t-mean Rm_est)^2/sum(Rm_est-mean)^2. Window SCAR=sum SAR/sqrt(L2). Patell Z=sum_i SAR/sqrt(sum_i (L_i-2)/(L_i-4)). BMP t=mean(SCAR)/(std(SCAR,ddof=1)/sqrt N). Kolari-Pynnonen: t_BMP*sqrt((1-rho)/(1+(N-1)rho)), rho=mean pairwise corr of estimation residuals. Plain t=CAAR/(std(CAR,ddof=1)/sqrt N).
3. Corrado rank: rank AR over estimation+event days jointly (average ties), U=K/(T+1)-0.5; S=sqrt(sum_t Ubar_t^2/D) with D = days in the full span (gap days count, contribute 0); z0=Ubar_0/S; window z=sum Ubar/(sqrt(L2) S).
4. SUE: time-series (Bernard-Thomas) = (E_q-E_{q-4})/std of prior seasonal differences (4-8 qtrs); analyst = (A-consensus)/std(estimates) (or /price). Surprise % = (A-C)/|C|*100; beat/miss thresholds strict.
5. Lexicon tone: lowercase, non-letters -> space, split; (pos-neg)/(pos+neg) (LM) or scale*(pos-neg)/total_tokens. Surprise = tone - mean(prior K), current excluded.
6. TF-IDF (sklearn smooth): idf=ln((1+N)/(1+df))+1; tf=count/total tokens (all tokens); novelty=1-cosine(current, mean of prior vectors).
7. Prediction market: price in (0,1) = risk-neutral prob per $1 payout; YES+NO~1. Locked edge=1-(p_yes_X+p_no_Y)-fees; payoff exactly 1 per paired unit. Kalshi schedule fee=0.07 C P(1-P) $ rounded up to the cent; else fee bps x leg price as given.
8. Perp funding: paid every 8h (3/day); rate>0 longs pay shorts; annualised=rate*3*365. Basis=(F-S)/S, annualised *365/days.
9. ETF: holding $=TNA*weight; implied price=TNA*w/shares; overlap=sum min(w_i,w_j); HHI=sum w^2, effective N=1/HHI; premium=(price-NAV)/NAV; redemption sell=pct*shares held.
10. Holdings turnover=0.5 sum|w_t-w_{t-1}| over union of names.

## Conventions and defaults when the task is silent
- Relative days = positions in the merged trading-day index, not calendar days; non-trading event date -> next trading day (after-close news -> day 0 next day if timestamps given).
- Defaults: estimation [-250,-11] or [-120,-11], >=60 obs; windows [-1,+1],[-5,+5]; log or simple returns as stated; never pad missing event days.
- 13F: filed <=45 days after quarter end. Eligible = 13F-HR and 13F-HR/A with report type "13F HOLDINGS REPORT" (NOTICE has no table; COMBINATION partial). Order filings by filing date, amendment no, accession. AMENDMENTTYPE RESTATEMENT replaces the whole report (even if short or confidential treatment expired); NEW HOLDINGS appends rows. VALUE: thousands before 2023-01-03 filings, whole dollars after. Keep SSHPRNAMTTYPE=SH, blank PUTCALL, VALUE>0; aggregate by CUSIP; manager = filer CIK (leading zeros), never accession prefix.
- Form 4 (due 2 business days): Table I non-derivative, Table II derivative. Codes: P open-market buy, S open-market sale, A grant, D to issuer, F tax/exercise-price withholding, M exempt option exercise, X in/at-money exercise, C conversion, G gift, J other, W will. Sale ledger = Table I, code S, disposed D. Inventory before a sale = shares + shares-following. Ownership D direct / I indirect (natureOfOwnership or footnote). Price footnotes give weighted-average ranges "$lo to $hi". 10b5-1 checkbox since 2023.
- 8-K items (due 4 business days): 1.01 material agreement, 2.01 acquisition, 2.02 results, 2.03 debt obligation, 2.05 exit costs, 2.06 impairment, 4.01 auditor change, 4.02 non-reliance, 5.02 officer departure/appointment, 5.07 vote, 7.01 Reg FD, 8.01 other, 9.01 exhibits. Date of Report = event date. Debt principal before over-allotment.
- 10-K: Item 1 Business, 1A Risk Factors, 1C Cybersecurity, 2 Properties, 5 equity/repurchases, 7 MD&A, 7A market risk, 8 statements. Headings also appear in the table of contents: anchor on the body (last) occurrence. Inline XBRL: ix:nonFraction scale=6 (millions), sign="-"; dei tags on cover (EntityPublicFloat = non-affiliate value); "(1,234)" negative; first number = latest year.

## Completing missing inputs
- Yields in percent: change*100 = bps. Offsets count valid data rows.
- 7d/30d style ratios: denominator max(x,1). Missing modality: CS median fill, all-null -> 0.
- Units: "$653.2 M" -> 653.2e6; "0.02%" -> 0.0002.

## Invariants to self-check
- CAR = sum of its ARs; CAAR = mean CAR; residual mean ~0 in estimation window; Corrado U sums ~0 per event.
- 13F weights sum to 1 per book; summary-page rows exact, value within rounding; HHI in (0,1].
- Form 4 rollforward: start - sold = end per bucket; shares >= 0.
- Tone in declared range; event date >= announcement; no future filing used.
- Prediction legs: net edge <= gross edge; pnl = qty*edge.

## Variants to decide
- V1 mm_sigma: market-model estimation window and residual variance. Options: (a) [-250,-11] disjoint from event window, >=60 obs, s^2=SSR/(L-2); (b) [-120,-11], SSR/(L-2); (c) std(resid, ddof=1)=SSR/(L-1). Default when silent: (a) - MacKinlay 1997.
- V2 patell_scar: multi-day Patell standardisation. Options: (a) SCAR=sum_t SAR_t/sqrt(L2), SAR_t with own C_t; (b) CAR/(s sqrt(sum_t C_t)); (c) CAR/(s sqrt(L2+L2^2/L1+(sum Rm_ev-L2 mean Rm)^2/SS_m)). Default when silent: (a) - Patell 1976.
- V3 bmp_ddof: BMP cross-sectional sd of SCAR/SAR. Options: (a) std(ddof=1); (b) std(ddof=0). Default when silent: (a) - Boehmer-Musumeci-Poulsen 1991.
- V4 kp_adjust: Kolari-Pynnonen correction form. Options: (a) t_BMP*sqrt((1-r)/(1+(n-1)r)); (b) t_BMP/sqrt(1+(n-1)r); r = mean pairwise corr of estimation residuals. Default when silent: (a) - Kolari-Pynnonen 2010.
- V5 corrado_scale: rank scaling. Options: (a) standardized U=K/(T_i+1)-0.5 (T_i = firm's non-missing days); (b) raw K-(T+1)/2 (same z when no returns missing). Default when silent: (a) - Corrado-Zivney 1992 (card formula 3).
- V6 corrado_sk_days: days entering s(K). Options: (a) all estimation+event days, ties averaged; (b) estimation days only. Default when silent: (a) - Corrado 1989.
- V7 window_stat: multi-day window test form. Options: (a) sum of daily stats/(sqrt(L2) x daily sd) (rank, Patell SCAR); (b) cross-sectional t on CARs; (c) mean of daily z. Default when silent: (a) - Campbell-Wasley 1993.
- V8 short_history: events with fewer than the full estimation days. Options: (a) keep with the shorter window if >= the stated minimum; (b) drop. Default when silent: (a) - MacKinlay 1997.
- V9 amendment_seq: same-day 13F amendments. Options: (a) order by filing date, numeric AMENDMENTNO, accession; (b) accession only. Default when silent: (a) - SEC 13F data set documentation.
- V10 qty_scaling: currency fields of a position. Options: (a) per-unit price or edge times qty; (b) per-unit values. Default when silent: (a) - currency units mean position totals.
- V11 xbrl_value: inline XBRL numbers. Options: (a) apply scale and sign="-", current fiscal-year context without dimensions; (b) displayed text as is. Default when silent: (a) - SEC EDGAR Filer Manual.

## Pitfalls
- Calendar-day windows; estimation window overlapping the event window.
- Forgetting the (1-rho) numerator in the KP factor; dropping gap days from the Corrado denominator.
- Treating restatements as appends; mixing 13F thousands and dollars across the 2023 change.
- Counting F/M/C/derivative rows as insider sales.
- Splitting sentences on '.' breaks "$13.3 billion" - parse line by line.
- tf over kept tokens only; including the current statement in its prior mean.
- Tie-prone rankings: break ties by the economic size measure, then ticker.
- Helper return SHAPES: sec13f_clean_rows keeps ALL rows (+reject_reason), clean_venue_snapshots is a 3-tuple, guidance/debt parsers return dicts, caar_by_day is a str-keyed dict. Prefer the *_bundle call that returns every file.
- A *_bundle returns {file: obj}: emit every file it returns; never tuple-unpack it or rebuild its JSONs from its CSV.

## Sources
MacKinlay 1997; Campbell-Lo-MacKinlay 1997; Patell 1976; Boehmer-Musumeci-Poulsen 1991; Corrado 1989; Kolari-Pynnonen 2010; Bernard-Thomas 1989; Loughran-McDonald 2011; SEC Form 13F/Form 4 instructions and EDGAR 13F data docs; Form 8-K and 10-K General Instructions; Wolfers-Zitzewitz 2004.
