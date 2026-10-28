method: kernel-density-var
family: var-es-methods
keywords: kernel density var, kde var, gaussian kernel, silverman bandwidth, scott bandwidth, gaussian_kde, smoothed historical simulation, kernel quantile, kernel expected shortfall, bandwidth selection
---
## Formulas
1. Gaussian KDE: f(x) = (1/(n h)) sum_i phi((x - r_i)/h); CDF F(x) = (1/n) sum_i Phi((x - r_i)/h).
2. Silverman: h = 0.9 min(s, IQR/1.34) n^(-1/5); Scott (scipy default): h = s n^(-1/5) (scipy bw factor n^(-1/(d+4)) times s).
3. VaR_c: solve F(q) = 1-c by brentq (or interpolate a fine grid); VaR = -q.
4. ES_c = -(1/(1-c)) int_{-inf}^{q} x f(x) dx = -(1/(1-c)) (1/n) sum_i [r_i Phi(a_i) - h phi(a_i)], a_i = (q - r_i)/h (closed form for the Gaussian kernel).
## Variants to decide
- V1 bandwidth: smoothing width. Options: (a) Scott via scipy gaussian_kde default; (b) Silverman rule-of-thumb; (c) cross-validated. Default when silent: (a) - scipy.stats.gaussian_kde; (b) when "Silverman" is named.
- V2 quantile_solver: inverting the CDF. Options: (a) root-finding on the exact mixture CDF; (b) grid plus np.interp. Default when silent: (a) - Sheather-Marron 1990.
- V3 es_integral: ES computation. Options: (a) closed-form Gaussian-kernel tail mean; (b) numerical quadrature on the grid. Default when silent: (a) - Scaillet 2004.
## Invariants
- KDE VaR close to HS VaR; ES >= VaR; F monotone with F(min - 5h) ~ 0.
## Pitfalls
- scipy bw_method scalar is a FACTOR multiplying s, not the bandwidth h itself.
- Grid too narrow: tail quantile clipped at the grid edge.
- Kernel on losses vs returns: keep the sign convention consistent.
## Sources
- Silverman 1986; Sheather-Marron 1990; Scaillet 2004 (Math Finance); Butler-Schachter 1998; scipy docs.
