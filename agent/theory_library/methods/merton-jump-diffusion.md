method: merton-jump-diffusion
family: bsm-greeks-pde
keywords: merton jump diffusion, jump-diffusion option price, merton 1976, poisson series, lognormal jumps, jump intensity, jump compensator, jump diffusion mle, mixture of normals likelihood, compound poisson
---
## Formulas
Model: dS/S = (r - q - lambda k) dt + sigma dW + (J - 1) dN, ln J ~ N(mu_J, delta^2), k = e^{mu_J + delta^2/2} - 1.
1. Series price: C = sum_{n=0}^{N} e^{-lambda' T} (lambda' T)^n / n! x BS(S, K, T, r_n, sigma_n, q), lambda' = lambda (1 + k), sigma_n^2 = sigma^2 + n delta^2/T, r_n = r - lambda k + n ln(1 + k)/T.
2. Log-return density over dt (MLE): f(x) = sum_n Pois(n; lambda dt) N(x; m dt + n mu_J, sigma^2 dt + n delta^2), m = mu - sigma^2/2 - lambda k (physical drift) or the stated drift.
3. Truncate when the Poisson weight tail < tiny tolerance (N about lambda' T + 10 sqrt(lambda' T) + 10).
## Variants to decide
- V1 jump_compensator: drift in the MLE. Options: (a) compensated drift mu - sigma^2/2 - lambda k; (b) free constant drift. Default when silent: (b) - Press 1967 / Ball-Torous 1983 estimation.
- V2 jump_units: lambda and moments. Options: (a) annual lambda with dt = 1/252; (b) per-period (daily) lambda. Default when silent: (a) - Merton 1976.
- V3 series_truncation: number of terms. Options: (a) adaptive until tail weight below tolerance; (b) fixed N. Default when silent: (a) - Haug 2007.
- V4 series_poisson_rate: series weights. Options: (a) lambda (1 + k); (b) lambda, as the task's formula writes. Default when silent: (a) - Merton 1976.
## Invariants
- lambda -> 0 gives BS; put-call parity holds with S e^{-qT} - K e^{-rT}; price increases in lambda for ATM.
- Log-likelihood finite; delta^2, sigma^2 > 0.
## Pitfalls
- MLE: logsumexp over the jump counts n per return, never over the returns (merton_jd_loglik).
- Forgetting lambda' (not lambda) in the Poisson weights of the series.
- Using sigma_n without the /T on the jump variance.
## Sources
- Merton (1976) J. Financial Economics; Haug (2007) ch. 6; Ball and Torous (1983).
