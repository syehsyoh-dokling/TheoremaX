method: form4-insider-sale-ledger
family: event-studies-and-filings
keywords: form 4, insider sale, insider selling, transaction code, nonderivativetransaction, sharesownedfollowingtransaction, natureofownership, footnoteid, weighted average price, price range footnote, 10b5-1, sale pressure, seller hhi, adv20
---
## Formulas
1. Ledger: Table I nonDerivativeTransaction, code S, acquiredDisposedCode D; exclude M, F, A, G, C, X and Table II.
2. Footnotes are child elements footnoteId id="Fn" (e.g. under pricePerShare); entity from ownershipNature/natureOfOwnership/value, else direct.
3. Inventory before trading (filer, entity, day) = post-trade shares of the first sale + its shares sold; holding rows excluded.
4. WAP = sum(q p)/sum q; range "$lo to $hi": width_bps = (hi - lo)/ref x 1e4.
5. Ratios (decimal): sold/ADV20, /float, /outstanding, /day volume; seller HHI = sum (q_j/Q)^2; ranks descending.
## Variants to decide
- V1 band_ref: band width denominator. Options: (a) WAP; (b) midpoint (lo+hi)/2. Default when silent: (a) - SEC Form 4 instruction 4(a) weighted-average reporting.
- V2 plan_flag: 10b5-1 detection. Options: (a) aff10b5One element; (b) footnote text; (c) either. Default when silent: (c) - SEC Form 4 instructions (checkbox since 2023).
- V3 rank_ties: rank ties. Options: (a) method=min descending, then ticker; (b) dense; (c) first by ticker. Default when silent: (a) - pandas rank min.
- V4 ratio_units: ratio scale. Options: (a) decimal fraction; (b) percent. Default when silent: (a) - card unit rule.
## Invariants
- start - sold = end per entity; shares >= 0; ledger holds only S/D rows.
## Pitfalls
- Searching for non-existent priceFootnoteId tags; counting F withholding as a sale.
- Merging direct and indirect lots of one filer.
- Entity left blank when nature says 'See footnote': resolve the name from that footnote; static holdings are buckets too.
## Sources
- SEC Form 4 General Instructions; EDGAR ownership XML specification; Cohen-Malloy-Pomorski 2012.
