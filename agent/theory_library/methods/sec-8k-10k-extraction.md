method: sec-8k-10k-extraction
family: event-studies-and-filings
keywords: 8-k, form 8-k, item 2.02, item 5.02, item 2.03, date of report, guidance midpoint, 10-k, inline xbrl, ixbrl, ix:nonfraction, contextref, dei, scale attribute, edgar parsing
---
## Formulas
1. 8-K items from the cover/item headings only; Date of Report (earliest event) = event date.
2. Guidance: range -> (lo+hi)/2; "X +- p%" -> X; unify units (B -> M); surprise % = (guide - consensus)/|consensus| x 100.
3. Debt (2.03): initial principal, excluding over-allotment.
4. Officer change (5.02): role of the departing/appointed person, word-boundary match (CEO vs CFO).
5. iXBRL: value = number x 10^scale, negated if sign="-"; tags are name="dei:..." attributes on ix:nonFraction / ix:nonNumeric.
6. Signal long if score >= +thr, short if <= -thr.
## Variants to decide
- V1 xbrl_context: which context. Options: (a) current fiscal-year context without dimensions; (b) first occurrence. Default when silent: (a) - SEC EDGAR Filer Manual.
- V2 cashflow_sign: "used in" amounts. Options: (a) signed as in the statement; (b) positive magnitude. Default when silent: (a) - US GAAP ASC 230 presentation.
- V3 guidance_point: guidance estimate. Options: (a) midpoint of range; (b) low end. Default when silent: (a) - Ciconte-Kirk-Tucker 2014.
- V4 threshold_inclusive: signal boundaries. Options: (a) >= and <=; (b) strict. Default when silent: (a) - follow written operators.
## Invariants
- Ticker resolved from the filing CIK, never guessed; JSON values native floats/ints.
## Pitfalls
- Ignoring scale; the first "$X billion" (current results) taken as the outlook.
- Items gathered from the whole body; substring tests for roles.
- Zip handed over as text (null byte / bytes errors): open by path or bytes (ixbrl_main_html); nested ix tags duplicate text.
- Event type from item numbers first (5.02 departure, 1.01/2.03 debt), not keyword order.
## Sources
- SEC Form 8-K and 10-K General Instructions; EDGAR Filer Manual (Inline XBRL); Loughran-McDonald 2016.
