method: student-t-copula-simulation
family: copulas-and-dependence
keywords: student-t copula, t copula simulation, normal variance mixture, chi-squared mixing variable, tail dependence, gaussian copula simulation, cholesky factor, correlated defaults, t_copula_df, degrees of freedom, joint default clustering
---
## Formulas
1. Gaussian: X = L G, L = chol(R) lower, G iid N(0,1); U = Phi(X).
2. t (normal variance mixture): W ~ chi2(nu)/nu, ONE per scenario; T = X/sqrt(W); U = t_nu.cdf(T).
3. Latent-factor form: A_i = [sqrt(rho_i) F_s(i) + sqrt(1-rho_i) eps_i]/sqrt(W); event iff A_i < t_nu^-1(p_i) preserves marginal p_i.
4. Tail dependence lambda = 2 t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho))); Gaussian 0.
## Variants to decide
- V1 mixing_scope: what W scales. Options: (a) both factors and idiosyncratic shocks; (b) factors only. Default when silent: (a) - Demarta-McNeil 2005 (gives a true t copula).
- V2 threshold: event threshold. Options: (a) t_nu quantile; (b) normal quantile. Default when silent: (a) - marginals preserved.
- V3 mixing_draw: W generator. Options: (a) rng.chisquare(nu, n)/nu; (b) rng.gamma(nu/2, 2/nu, n). Default when silent: (a) - McNeil-Frey-Embrechts 2015 (same law, different stream).
- V4 seed_plan: t run seed. Options: (a) stated offset seed; (b) base seed. Default when silent: (a) - task text when an offset is given.
## Invariants
- Simulated marginal event rates ~ p_i for both copulas; t joint tail clustering > Gaussian.
## Pitfalls
- Independent W per coordinate; upper Cholesky (numpy returns lower).
## Sources
- Demarta-McNeil 2005; McNeil-Frey-Embrechts 2015 ch.7; Li 2000; Glasserman 2004 ch.9.
