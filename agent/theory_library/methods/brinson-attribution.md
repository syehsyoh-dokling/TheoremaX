method: brinson-attribution
family: portfolio-construction
keywords: brinson, brinson-fachler, brinson-hood-beebower, bhb, performance attribution, allocation effect, selection effect, interaction effect, active return, sector attribution, carino linking, multi-period attribution
---
## Formulas
1. Sector i: wp_i, wb_i, Rp_i, Rb_i; Rp = sum wp Rp_i, Rb = sum wb Rb_i.
2. BHB: allocation (wp-wb) Rb_i; selection wb (Rp_i - Rb_i); interaction (wp-wb)(Rp_i - Rb_i).
3. BF: allocation (wp-wb)(Rb_i - Rb); selection, interaction as BHB.
4. Two-factor: selection wp (Rp_i - Rb_i), interaction folded in.
5. Sum of effects = Rp - Rb per period. Carino: k_t = (ln(1+R_t) - ln(1+B_t))/(R_t - B_t), k likewise for totals; effect x k_t/k.
## Variants to decide
- V1 brinson_scheme: allocation formula. Options: (a) BHB; (b) Brinson-Fachler. Default when silent: (a) - Brinson-Hood-Beebower 1986.
- V2 interaction: interaction handling. Options: (a) separate term; (b) merged into selection. Default when silent: (a) - Brinson-Hood-Beebower 1986.
- V3 linking: multi-period aggregation. Options: (a) arithmetic sum; (b) Carino log linking; (c) Menchero. Default when silent: (a) - Bacon 2008.
- V4 weight_timing: weights used. Options: (a) beginning of period; (b) average. Default when silent: (a) - Bacon 2008.
## Invariants
- Allocation totals equal across BHB and BF; sector allocations differ.
- alloc + sel + inter = Rp - Rb each period.
## Pitfalls
- End-of-period weights; a sector missing on one side (weight 0, return still defined).
- Daily prices spanning several years: clip to the evaluation months first, each month anchored on the prior month's last trading day; drift weights between resets (month_end_returns, brinson_fachler_drift).
## Sources
- Brinson-Fachler 1985; Brinson-Hood-Beebower 1986; Carino 1999; Bacon 2008.
