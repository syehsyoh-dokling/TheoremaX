method: par-yield-bootstrap-spline
family: curve-bootstrapping
keywords: par yield bootstrap, treasury par yields, cmt par curve, coupon-equivalent par yield, semi-annual par, bootstrap par instruments, g(t) = z(t) t, interpolate exponent, not-a-knot cubic spline, cubic spline on g, par bond reprices to 100, bill point, zero curve diagnostics, re-bootstrap
---
## Formulas
1. Par bond (freq f, coupon = par yield y): 100 = 100 (y/f) sum_{k=1..n} D(t_k) + 100 D(T), t_k = T - (n-k)/f.
2. Sub-period point (T < 1/f, bill): D(T) = 1/(1 + y T) (simple, money-market); alternative bond-equivalent (1 + y/f)^(-f T).
3. Exponent spline: knots (0, T_1..T_m), values (0, g_1..g_m), g = z T = -ln D; CubicSpline(bc_type='not-a-knot'); D(t) = exp(-g(t)), z(t) = g(t)/t, instantaneous f(t) = g'(t).
4. Sequential solve: root-find g_m so the par bond at T_m prices to 100; gap coupon dates read from the spline incl. the trial g_m.
5. Pillar forward: f_m = (g_m - g_{m-1})/(T_m - T_{m-1}); first row f_1 = z_1.
6. Risk: bump the PAR yields (shape per task), re-bootstrap, rebuild the spline, reprice.
## Variants to decide
- V1 short_end: sub-coupon par points. Options: (a) simple D = 1/(1 + yT); (b) bond-equivalent compounding. Default when silent: (a) - US Treasury bill convention.
- V2 gap_dfs: coupon dates between pillars. Options: (a) final interpolation incl. the trial pillar, root-find; (b) linear in g; (c) interpolate par yields to every coupon date. Default when silent: (a) - Hagan-West 2006.
- V3 forward_column: reported forward. Options: (a) discrete between adjacent pillars; (b) instantaneous g'(T). Default when silent: (a) - Hull OFOD ch. 4.
- V4 spline_bc: spline end conditions. Options: (a) not-a-knot; (b) natural. Default when silent: (a) - de Boor / scipy default.
## Invariants
- Par instruments reprice to 100 (1e-8); z nonzero at every pillar; D decreasing.
## Pitfalls
- Coupon in percent or annual freq on semi-annual yields: par bonds far above 100.
- Bill pillar left unsolved (zero rate 0); spline on raw zeros when g is asked.
## Sources
- Hagan-West (2006); Tuckman-Serrat ch. 2.
