method: hull-white-trinomial-bermudan
family: short-rate-and-rate-options
keywords: trinomial tree, hull-white tree, arrow-debreu, state prices, alpha shift, jmax, branching probabilities, bermudan swaption, co-terminal bermudan, backward induction, optimal stopping, exercise boundary, tree european swaption, swaption dv01
---
## Formulas
1. dx = sigma sqrt(3 dt); jmax = max(1, ceil(0.184/(a dt))); M = -a dt.
2. Normal branch: pu = 1/6 + (j^2M^2 + jM)/2, pm = 2/3 - j^2M^2, pd = 1/6 + (j^2M^2 - jM)/2. Top (j = jmax, to j, j-1, j-2): pu = 7/6 + (j^2M^2 + 3jM)/2, pm = -1/3 - j^2M^2 - 2jM, pd = 1/6 + (j^2M^2 + jM)/2. Bottom mirrors.
3. Arrow-Debreu: Q_0(0) = 1; alpha_m = [ln sum_j Q_m(j) e^{-j dx dt} - ln P(0,t_{m+1})]/dt; Q_{m+1}(k) = sum_j Q_m(j) p_{j->k} e^{-(alpha_m + j dx) dt}.
4. Node rate r_{m,j} = alpha_m + j dx; discount e^{-r dt}.
5. European payer on the tree at step n0: payoff max(1 - sum_i c_i P_HW(T0,Ti,r_j), 0), rolled back.
6. Bermudan: V = exercise at last step; back, at exercise steps V = max(exercise, continuation); exercise uses payments with T_i > t_k only and the alpha of that step.
7. Boundary j* = smallest j where exercise > continuation; r* = alpha + j* dx. DV01 = (P(curve +1bp) - P) 1e4 per bp, tree rebuilt.
## Variants to decide
- V1 tree_bond: node bond. Options: (a) P_HW in dt-rate R: A^ e^{-B^ R}, B^=B dt/B(t,t+dt); (b) instant P_HW(t,T,r); (c) rollback. Default when silent: (a) - Hull OFOD ch. 32.
- V2 exercise_coupons: remaining bond at t_k. Options: (a) payments strictly after t_k; (b) include the coupon at t_k. Default when silent: (a) - Andersen-Piterbarg 2010.
- V3 dv01_bump: curve shock. Options: (a) parallel +1bp to continuous zeros; (b) +1bp to DFs-implied par rates. Default when silent: (a) - Hull OFOD ch. 32.
## Invariants
- sum_j Q_n(j) = P(0,t_n) within 1e-6; probabilities in [0,1].
- Bermudan >= European; tree European within a few bp of Jamshidian; j* weakly decreasing.
## Pitfalls
- One alpha for all exercise steps; step index round(t/dt), not int().
- Exercise-date mapping off by one when t/dt is not an integer.
## Sources
- Hull-White (1994) Numerical procedures; Hull OFOD ch. 32; Brigo-Mercurio ch. 3.
