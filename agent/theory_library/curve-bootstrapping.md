family: curve-bootstrapping
category: fixed-income
keywords: bootstrap, zero curve, discount factor, par rate, par yield, zero rate, forward rate, deposit, fra, futures, swap curve, ois, sofr, libor, dual curve, projection curve, interpolation, log-linear, reprice, nelson-siegel, svensson, spot rate, pillar, instantaneous forward, dual-curve, 3m libor, irs par rate, repriced quotes, residual bps, par yields, not-a-knot, cubic spline, act/365.25, model par rate
---
## Worked frame
DIKETAHUI: instruments one row each [P] (type, maturity in years, quote in percent -> decimal, freq). Check: sorted by maturity, no duplicate tenors.
DITANYAKAN: one row per pillar [P]: discount factor, zero rate, forward (stated compounding); repricing errors [P].
RUMUS: card Core 1-4 (pillar solve) or 5 (dual curve) -> 6 -> 7; toolkit par_curve, bootstrap_par, zero_rate, forward_rate, discount_factor.
JAWAB: 1) quotes as decimals [P]. 2) solve DF pillar by pillar in maturity order [P]. 3) zero rates [P]. 4) forwards between adjacent pillars [P-1]; state how row one is filled. 5) convert to output compounding and unit. 6) reprice inputs -> errors [P]. 7) check: DF(0) = 1, DF decreasing, errors ~ 0; then write.

## Core formulas
DF(t) = discount factor, tau = accrual fraction in the instrument's day count, t = curve time.
1. Deposit / single-payment OIS: DF(T) = DF(t0) / (1 + r tau(t0,T)); DF(t0) = 1 when the curve is anchored at spot (else DF(spot) from ON/TN).
2. FRA t1 x t2: DF(t2) = DF(t1) / (1 + F tau12). Futures: F_fut = 100 - price (percent); convexity adj. F_fwd = F_fut - 0.5 sigma^2 t1 t2 (Ho-Lee, Hull).
3. Single-curve par swap: S = (DF(T0) - DF(Tn)) / A, A = sum_i tau_i DF(Ti) (fixed-leg accruals). Solve pillar: DF(Tn) = (DF(T0) - S sum_{i<n} tau_i DF(Ti)) / (1 + S tau_n).
4. Par bond (price 1, freq f): 1 = (c/f) sum_{k<n} DF(t_k) + (1 + c/f) DF(T_n); annual: d(n) = (1 - c sum_{i<n} d(i))/(1 + c).
5. Dual curve: float forward F_j = (P(t_{j-1})/P(t_j) - 1)/tau_j from the PROJECTION curve P; S = sum_j tau_j F_j DF_ois(t_j) / sum_i tau_i DF_ois(t_i). Build OIS first, then root-find each projection pillar with OIS discounting fixed.
6. Rates from DF: continuous z = -ln DF / t; annual (1+z)^-t; periodic (1+z/f)^(-f t); simple DF = 1/(1+z t).
7. Forwards: continuous f(t1,t2) = ln(DF1/DF2)/(t2-t1) = (z2 t2 - z1 t1)/(t2 - t1); simple (DF1/DF2 - 1)/tau; instantaneous f(t) = -d ln DF/dt.
8. Nelson-Siegel (x = tau/lambda): y = b0 + b1 (1-e^-x)/x + b2 [(1-e^-x)/x - e^-x]; y(0+) = b0+b1, y(inf) = b0; inst. forward f = b0 + b1 e^-x + b2 x e^-x. Svensson adds b3 [(1-e^-x2)/x2 - e^-x2], x2 = tau/lambda2. Diebold-Li write x = lambda*tau (lambda = 0.0609 per month): check which form the task uses.
## Conventions and defaults when the task is silent
- Interpolation default: log-linear DF (= linear in g = z t, piecewise-flat forwards; QuantLib PiecewiseLogLinearDiscount). Linear-in-zero gives saw-tooth forwards; cubic spline on z or g gives smooth forwards but can oscillate; monotone-convex (Hagan-West 2006) is the "smooth and positive" choice. The stated method always wins; if a task says "interpolate g(T) = z T with spline incl. g(0)=0", do exactly that.
- Extrapolation: flat zero (or flat forward) beyond the last pillar; before the first pillar flat zero from t=0 (DF(0)=1).
- Diagnose the generator's convention: when input prices are synthetic, the right interpolation reprices them to ~1e-10 / gives calibration RMSE ~0; a wrong one leaves bp-level residuals.
- Treasury CMT par yields = semi-annual bond-equivalent; sub-1Y points (bills) are single-payment: DF = 1/(1 + y T), NOT semi-annual compounding.
- Post-2008 standard: discount on OIS (collateral rate), project IBOR on its own curve; the projection curve is NOT a discount curve.
- Output zero rates continuous unless the task names compounding; percent inputs (4.33) -> /100.
- NS fitting: with lambda fixed the problem is linear OLS in betas; use the given start/optimiser; objective on yields (not prices) unless stated.
## Completing missing inputs
- Missing pillars (1,2,3,5,7,10 with annual coupons): coupon DFs at 4, 6, 8, 9 depend on the unknown pillar through the interpolation -> 1-D root-find (brentq) on the new pillar's z or ln DF so the instrument reprices exactly; the closed form only works when every earlier coupon date is already known.
- Coupon dates before the first pillar (freq > 1): flat-extrapolate the first zero (state it) or solve jointly.
- Model par rate at any tenor: S = (1 - DF(T))/A from the built curve (single curve) or float PV / annuity (dual curve).
- Swap PV01 sanity: PV01 ~ notional * A * 1e-4 (5Y annuity ~4.4-4.7 at 3-5% rates).
- Forward from pillar at a discontinuity (log-linear): use the right-hand forward -(ln DF(t+h) - ln DF(t))/h.
## Invariants to self-check
- Every input instrument reprices: |model - quote| < 1e-8 (<1e-6 bp); model par rate at a pillar = input.
- DF(0) = 1, 0 < DF <= 1, strictly decreasing for positive rates; z <-> DF round trip exact.
- Telescoping: sum_i f_i (t_i - t_{i-1}) = z(T) T.
- Upward-sloping par curve: forward > zero > par at the long end; inverted: reverse.
- Swap at the fair rate has NPV = 0; NS: b0 > 0, y(0+) = b0 + b1 close to the short rate.
- Projection curve above OIS by a small positive spread (IBOR-OIS) in normal markets.
## Variants to decide
- V1 interpolation: between pillars. Options: (a) log-linear DF (piecewise-flat forwards); (b) linear in zero rate; (c) cubic spline / monotone-convex. Default when silent: (a) - Hagan-West 2006 / QuantLib.
- V2 zero_compounding: reported zero rates. Options: (a) continuous; (b) annual; (c) semi-annual. Default when silent: (a) - Hull.
- V3 discounting: curve set-up. Options: (a) OIS discounting with separate IBOR projection curve; (b) single curve for both. Default when silent: (a) - post-2008 standard (Bianchetti 2010; Hull-White 2013).
- V4 futures_convexity: futures-to-forward adjustment. Options: (a) subtract 0.5 sigma^2 T1 T2 when sigma given, else none; (b) always none. Default when silent: (a) - Hull (Ho-Lee convexity adjustment).
- V5 short_end: sub-1Y par points. Options: (a) simple money-market DF=1/(1+yT); (b) same compounding as coupon bonds. Default when silent: (a) - US Treasury bill convention.
- V6 extrapolation: beyond last pillar. Options: (a) flat zero rate; (b) flat forward. Default when silent: (a) - Hagan-West 2006.
- V7 curve_time_basis: curve time axis for interpolation and zeros. Options: (a) ACT/365F; (b) ACT/365.25; (c) the instrument accrual basis. Default when silent: (a) - Henrard 2014 / OpenGamma; a stated basis always wins.
- V8 gap_coupon_dfs: coupon dates between pillars. Options: (a) root-find each pillar with the final interpolation (incl. the trial node) for gap dates; (b) interpolate par yields to every coupon date, then classical bootstrap. Default when silent: (a) - Hagan-West 2006.
- V9 ois_payment_structure: OIS pillar cash flows. Options: (a) single payment up to 1Y, annual coupons beyond; (b) single payment at every tenor. Default when silent: (a) - Ametrano-Bianchetti 2013; (b) when the convention file says zero-coupon.

## Pitfalls
- Counting coupons by the list index of a maturity instead of maturity_years * freq.
- Flat-extrapolating the last known DF inside a gap instead of interpolating toward the pillar being solved.
- Dual curve done single-curve inside the root-finder (trial projection curve also used to discount); building projection before OIS.
- Using spline during the bootstrap when the task says sequential solve then spline (or vice versa) - follow wording; small construction changes move long zeros by ~bp.
- Central-difference instantaneous forward at a pillar (averages two sides).
- Forward via T-1 when pillars are not consecutive: use the previous pillar or interpolate z(T-1) as instructed.
- Semi-annual compounding for a 3M bill point; percent/decimal mix; float keys "1.0" vs "1".
- scipy hybr may report "failure" at machine-precision residuals: judge convergence by residual.
- 'Interpolate the zero rate linearly for coupon dates between maturities' makes each segment implicit in the unknown zero: solve it (bootstrap_par_linear_zero); interpolating par rates breaks repricing at gap maturities.
- Debug-the-template tasks: patch the given script (curve order, OIS discounting inside the IBOR bootstrap, fixed-leg day count, NPV sign) instead of rewriting; dual_curve_bootstrap is the reference build.
- Mixed deposits / FRAs (+adjustment bp) / par OIS swaps: bootstrap_df_curve solves each node so the quote reprices; never retype curve labels.
- Clean quote files (no time / unit column) need no normalize_quote_book: pass them to the bootstrap bundle.
## Sources
Hull OFOD ch. 4, 7; Andersen-Piterbarg Interest Rate Modeling vol. 1 ch. 6; Hagan-West (2006) Interpolation Methods for Curve Construction; Ametrano-Bianchetti (2013) multiple-curve bootstrapping; Nelson-Siegel (1987); Svensson (1994); Diebold-Li (2006).
