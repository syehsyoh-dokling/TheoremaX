method: vasicek-one-factor-credit-var
family: portfolio-credit-risk
keywords: vasicek, one-factor model, asrf, conditional default probability, credit var, cvar, euler allocation, marginal cvar contribution, granularity adjustment, gordy, multi-factor merton, wrong-way risk, expected loss
---
## Formulas
1. p_q,i = Phi((Phi^-1(PD_i) + sqrt(rho_i) Phi^-1(q))/sqrt(1-rho_i)); ASRF VaR_q = sum_i EAD_i LGD_i p_q,i.
2. MC: A_i = sqrt(rho_i) F_s(i) + sqrt(1-rho_i) eps_i, F = chol(Sigma) G; default iff A_i < Phi^-1(PD_i). t copula: see student-t-copula-simulation.
3. VaR_q = empirical quantile of L; CVaR_q = mean(L | L >= VaR_q).
4. Euler: C_i = mean(L_i | L >= VaR_q) on the same rows; sum C_i = CVaR.
5. Gordy: VaR_Gordy = VaR_ASRF + GA (family card).
## Variants to decide
- V1 var_measure: credit VaR. Options: (a) loss quantile; (b) quantile minus EL. Default when silent: (a) - empirical-quantile wording; (b) for unexpected loss or economic capital (Basel IRB).
- V2 euler_tail: tail set. Options: (a) L >= VaR; (b) L > VaR. Default when silent: (a) - Tasche 2008.
- V3 memory_plan: large n_sim x N. Options: (a) row chunks from one generator, second pass with the same seed for contributions; (b) float32 loss matrix. Default when silent: (a) - same stream, same numbers.
- V4 stress_seed: wrong-way rerun. Options: (a) base seed; (b) new seed. Default when silent: (a) - Glasserman 2004 common random numbers.
## Invariants
- MC mean / analytic EL within ~2% per copula; contributions sum to CVaR; wrong-way CVaR above base.
## Pitfalls
- Interleaved chunk draws (M chunk, eps chunk) when the reference drew all M first.
- Writing nothing after a timeout: write cheap analytic outputs first.
## Sources
- Vasicek 2002; Gordy 2003; Gordy-Lutkebohmert 2013; Tasche 2008; Glasserman 2004.
