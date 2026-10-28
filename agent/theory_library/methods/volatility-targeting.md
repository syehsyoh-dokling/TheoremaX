method: volatility-targeting
family: portfolio-construction
keywords: volatility targeting, vol target, target volatility, vol scaling, leverage cap, ex-ante volatility, ewma volatility, cta, trend following, time-series momentum, ema crossover signal, garch vol regime, managed volatility
---
## Formulas
1. Scale_t = min(L_max, sigma_target/sigma_hat_t), sigma_hat from data through t, applied to t+1.
2. Rolling: std(r, ddof=1) sqrt(P). EWMA: s2_t = lambda s2_{t-1} + (1-lambda) r_t^2, lambda = 1 - alpha; seed with each asset's own first r^2.
3. TSMOM: w_i = signal_i x (sigma_target/N)/sigma_i, signal per asset from its own prices.
4. Portfolio ex-ante: sigma_p = sqrt(P w'Sigma w); scale the book.
5. GARCH fitted on 100 x r: omega/100^2 back to decimal; persistence alpha + beta; long-run var omega/(1 - alpha - beta).
## Variants to decide
- V1 vol_estimator: volatility estimate. Options: (a) rolling std ddof=1; (b) EWMA RiskMetrics; (c) GARCH(1,1). Default when silent: (a) - Moreira-Muir 2017.
- V2 vol_level: where scaling applies. Options: (a) per asset then combine; (b) whole portfolio with full covariance. Default when silent: (a) - Moskowitz-Ooi-Pedersen 2012.
- V3 vol_lag: estimate timing. Options: (a) sigma_hat through t applied to t+1; (b) through t-1. Default when silent: (a) - Moskowitz-Ooi-Pedersen 2012.
- V4 ewma_mean: EWMA centring. Options: (a) zero-mean squared returns; (b) demeaned. Default when silent: (a) - RiskMetrics 1996.
## Invariants
- 0 <= scale <= cap; warm-up has zero exposure; realised vol near target.
## Pitfalls
- Same-day vol; annual target vs daily sigma; alpha passed where decay is expected.
- One asset's signal or seed reused for all; GARCH output left in percent^2.
- Price-length EMA put into a returns-length array: compute on prices, align with [1:].
## Sources
- Moskowitz-Ooi-Pedersen 2012; Moreira-Muir 2017; Harvey et al. 2018; RiskMetrics 1996; Bollerslev 1986.
