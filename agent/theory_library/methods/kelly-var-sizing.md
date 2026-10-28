method: kelly-var-sizing
family: portfolio-construction
keywords: kelly criterion, kelly fraction, fractional kelly, half kelly, growth optimal, var constraint, var cap, position sizing, leverage, value at risk sizing, monte carlo wealth paths, terminal wealth
---
## Formulas
1. Binary f = p - (1-p)/b. Continuous f = (mu - r)/sigma^2; multi f = Sigma^-1(mu - r), per-period inputs, rf_p = rf/P.
2. Growth g = r + f'(mu - r) - f'Sigma f/2.
3. VaR over h (loss positive) = z_a sigma_p sqrt(h) - mu_p h, sigma_p = sqrt(f'Sigma f).
4. Cap s = min(1, VaR_max/VaR(f)); final = s x f (VaR is degree-1 homogeneous).
5. MC: one seeded draw set shared by all schemes; port r = rf(1 - sum f) + f'r; W = W0 prod(1+r); drawdown includes W0.
## Variants to decide
- V1 kelly_inputs: moment estimates. Options: (a) sample mean and ddof=1 covariance of simple returns; (b) log-return moments. Default when silent: (a) - Thorp 2006.
- V2 var_mean_term: VaR mean term. Options: (a) z sigma - mu; (b) z sigma only. Default when silent: (a) - Jorion 2007.
- V3 sim_cash_leg: simulated portfolio return. Options: (a) rf on the unlevered remainder plus f'r; (b) f'r only. Default when silent: (a) - Merton 1969 budget.
- V5 mc_draw: simulated returns. Options: (a) default_rng(seed).multivariate_normal(mean, cov, size=(paths, days)); (b) mean + Z L' Cholesky. Default when silent: (a) - the literal multivariate normal draw.
- V4 path_sharpe: Monte Carlo Sharpe summary. Options: (a) median of per-path Sharpe; (b) pooled Sharpe. Default when silent: (a) - when "per path" or medians are reported.
## Invariants
- s keeps direction; VaR(final) <= VaR_max; s = 1 when the cap does not bind.
## Pitfalls
- Annual mu with daily sigma; z at the wrong tail; fresh draws per scheme.
- Skipping feed cleaning (duplicates, inf, returns <= -1) before the estimation tail.
## Sources
- Kelly 1956; Thorp 2006; MacLean-Thorp-Ziemba 2011; Jorion 2007.
