method: crank-nicolson-psor
family: american-early-exercise
keywords: crank-nicolson, psor, projected successive over-relaxation, projected sor, american option finite difference, free boundary, early exercise boundary, cash dividend jump condition, richardson extrapolation, dirichlet boundary, tridiagonal, over-relaxation
---
## Formulas
1. Grid S_i = i dS, t_n = n dt, backward from t = T. Operator L: sub a_i = 0.5 (sigma^2 i^2 - (r-q) i), diag b_i = -(sigma^2 i^2 + r), super c_i = 0.5 (sigma^2 i^2 + (r-q) i).
2. CN: (I - 0.5 dt L) V^n = (I + 0.5 dt L) V^{n+1}: the HALF factor sits on both sides.
3. Dirichlet rows 0 and N_S: identity row, value = boundary at t_n (put V(0) = K e^{-r tau}, V(Smax) = 0; call V(0) = 0, V(Smax) = Smax e^{-q tau} - K e^{-r tau}); interior rhs gets the boundary times its coefficient.
4. PSOR: y_i = (rhs_i - l_i v_{i-1} - u_i v_{i+1})/d_i using already-updated v_{i-1}; v_i = max(payoff_i, v_i + omega (y_i - v_i)); stop when max|dv| < tol.
5. Cash dividend D at t_d: V(S, t_d-) = V(S - D, t_d+) by interpolation (S - D clipped at 0), then project max(V, payoff) for American.
6. Richardson: (4 P_fine - P_coarse)/3, both dS and dt halved.
## Variants to decide
- V1 cn_theta: time weighting. Options: (a) CN theta = 1/2 on both sides; (b) Rannacher start with implicit steps then CN. Default when silent: (a) - Crank-Nicolson 1947 / Wilmott.
- V2 boundary_scan: exercise boundary search. Options: (a) largest S with |V - payoff| < tol among nodes with payoff > 0; (b) all nodes. Default when silent: (a) - Wilmott-Howison-Dewynne 1995.
- V3 sor_omega: relaxation. Options: (a) fixed omega in [1.2, 1.5]; (b) Brennan-Schwartz direct solve. Default when silent: (a) - Wilmott.
## Invariants
- No-div American call = European; American >= European; dividends raise puts, lower calls.
- Put delta = -1 below S*(t); S*(T) at K side; ratio |coarse - rich|/|fine - rich| about 4.
## Pitfalls
- Full dt operator on both sides doubles diffusion: call prices far too high.
- Boundary row left as PDE stencil instead of identity.
## Sources
- Wilmott, Howison, Dewynne (1995); Brennan-Schwartz (1977); Hull ch. 21.
