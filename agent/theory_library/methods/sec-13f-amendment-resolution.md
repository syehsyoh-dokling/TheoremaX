method: sec-13f-amendment-resolution
family: event-studies-and-filings
keywords: 13f, form 13f, 13f-hr, 13f-hr/a, amendment, restatement, new holdings, infotable, coverpage, summarypage, accession number, amendmentno, institutional holdings, crowding, holdings overlap
---
## Formulas
1. Eligible: 13F-HR or 13F-HR/A, report type "13F HOLDINGS REPORT", PERIODOFREPORT consistent with REPORTCALENDARORQUARTER.
2. Order per (manager CIK, period): FILING_DATE, numeric AMENDMENTNO, accession. First = initial book; RESTATEMENT replaces; NEW HOLDINGS appends.
3. Row filters in fixed priority, each reject counted once: blank CUSIP, then type != SH, then PUTCALL set, then value or shares <= 0; then sum duplicate CUSIPs.
4. VALUE in thousands for older filings, dollars after the 2023 change; reconcile with the summary page.
5. w = value/sum; HHI = sum w^2; effective N = 1/HHI; overlap = sum min(w_i, w_j); turnover = 0.5 sum|dw| over the union.
## Variants to decide
- V1 amendment_order: same-day amendments. Options: (a) numeric AMENDMENTNO then accession; (b) accession only. Default when silent: (a) - SEC 13F data set documentation.
- V2 value_units: VALUE scale. Options: (a) era-dependent thousands vs dollars; (b) dollars throughout. Default when silent: (a) - SEC 13F FAQ.
- V3 crowding_count: crowding measure. Options: (a) count of managers holding with w > 0; (b) value share of all managers. Default when silent: (a) - Sias 2004 style breadth.
- V4 restatement_scope: restatement effect. Options: (a) replaces the full book; (b) merges by CUSIP. Default when silent: (a) - Form 13F General Instructions.
## Invariants
- One resolved book per manager-period; weights sum to 1; joins never duplicate rows.
## Pitfalls
- DD-MON-YYYY vs ISO dates; CIK int vs padded string; string AMENDMENTNO order.
- Many-to-many joins of the filing tables (join on accession).
- CIK lives in SUBMISSION, not COVERPAGE; type filter on SSHPRNAMTTYPE, not TITLEOFCLASS (sec13f_* helpers).
## Sources
- SEC Form 13F General Instructions; SEC 13F data sets readme and FAQ; Sias 2004.
