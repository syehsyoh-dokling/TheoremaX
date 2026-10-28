method: compound-poisson-aggregate-loss
family: evt-and-tails
keywords: compound poisson, aggregate loss distribution, panjer recursion, fft aggregate loss, frequency severity, severity discretisation, loss distribution approach, operational risk var, collective risk model, negative binomial frequency
---
## Formulas
1. S = sum_{i<=N} X_i, N ~ Poisson(lambda): E S = lambda E X, Var S = lambda E X^2.
2. Rounding discretisation on step h: f_0 = F(h/2), f_k = F((k+1/2)h) - F((k-1/2)h).
3. Panjer (a,b,0): g_0 = PGF_N(f_0); g_s = [1/(1 - a f_0)] sum_{j=1..s} (a + b j/s) f_j g_{s-j}; Poisson a = 0, b = lambda.
4. FFT: g = real(ifft(exp(lambda(fft(f) - 1)))) on a zero-padded grid; CDF = cumsum(g).
5. VaR_a = h x first index with CDF >= a.
## Variants to decide
- V1 aggregation_algorithm: method. Options: (a) FFT; (b) Panjer recursion; (c) Monte Carlo. Default when silent: (a) - Embrechts-Frei 2009; (b) when Panjer is named.
- V2 discretisation: severity lattice. Options: (a) rounding midpoint; (b) lower/upper bound; (c) local moment matching. Default when silent: (a) - Klugman-Panjer-Willmot.
- V3 grid_span: grid size. Options: (a) span well beyond the 99.99% aggregate quantile, padded; (b) fixed stated n. Default when silent: (a) - Embrechts-Frei 2009 (aliasing).
- V4 es_def: lattice ES. Options: (a) Acerbi-Tasche with atom; (b) strict E[S|S>VaR]. Default when silent: (a) - Acerbi-Tasche 2002; (b) when ">" is written.
## Invariants
- sum g = 1; mean of g ~ lambda E X; g_0 = exp(-lambda(1 - f_0)); FFT and Panjer agree.
## Pitfalls
- No padding (tail wraps onto small losses); pdf x h instead of CDF differences.
- Reporting the index, not index x h.
- Task text 'scale beta': pass scale=, never beta= (refused as ambiguous); compound_poisson_risk_table runs FFT and MC together.
## Sources
- Panjer 1981; Klugman-Panjer-Willmot, Loss Models; Embrechts-Frei 2009; Shevchenko 2011.
