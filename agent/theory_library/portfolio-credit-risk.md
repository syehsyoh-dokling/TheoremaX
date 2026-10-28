family: portfolio-credit-risk
category: credit
keywords: creditmetrics, credit var, credit portfolio, gaussian copula, student-t copula, vasicek, asymptotic single risk factor, basel irb, asset correlation, expected loss, unexpected loss, lgd, ead, expected shortfall, cvar, euler allocation, marginal contribution, wrong-way risk, granularity adjustment, hhi, concentration, loss distribution, monte carlo, default threshold, rating migration, transition thresholds, upgrade rate, downgrade rate, gordy, risk-weight ratio, sector risk decomposition
---
## Worked frame
DIKETAHUI: obligors one row each [N] (PD, LGD as decimal, EAD in currency, rho, sector), correlation [S x S], n_sim, seed, levels. Check: PD percent -> decimal, sector labels match the matrix.
DITANYAKAN: EL, VaR, ES per level (positive loss, currency); contributions [N]; capital, RWA [N].
RUMUS: card Core 1 -> 2 or 5 -> 7 -> 8; analytic 3-4; toolkit one_factor_mc_losses, vasicek_loss_quantile, basel_irb_capital.
JAWAB: 1) thresholds [N]. 2) factors [n_sim] or [n_sim x S], shocks [n_sim x N]. 3) default flags [n_sim x N]. 4) losses [n_sim x N] -> total [n_sim]. 5) EL, VaR, ES; contributions [N] on the same tail rows. 6) loss rate x EAD -> currency. 7) check: MC mean ~ EL, VaR <= ES, contributions sum to ES; then write.

## Core formulas
1. EL = PD * LGD * EAD, LGD = 1 - recovery. Portfolio EL = sum_i PD_i LGD_i EAD_i (copula-invariant).
2. One-factor threshold: A_i = sqrt(rho_i) M + sqrt(1-rho_i) eps_i; default iff A_i < Phi^-1(PD_i).
   Multi-factor: sector factors F = L Z, L = cholesky(Sigma) (lower), A_i uses F_{s(i)}; corr(A_i,A_j) =
   sqrt(rho_i rho_j) Sigma_{s(i)s(j)}.
3. Vasicek conditional PD: p(M) = Phi((Phi^-1(PD) - sqrt(rho) M)/sqrt(1-rho)). At confidence q the
   bad factor is M = -Phi^-1(q): p_q = Phi((Phi^-1(PD) + sqrt(rho) Phi^-1(q))/sqrt(1-rho)).
   ASRF VaR_q = sum_i EAD_i LGD_i p_q,i. LHP loss-rate CDF: F(x) = Phi((sqrt(1-rho) Phi^-1(x) - Phi^-1(PD))/sqrt(rho)).
4. Basel IRB corporate (BCBS 128 para 272): R = 0.12 w + 0.24 (1-w), w = (1-e^(-50 PD))/(1-e^(-50));
   b = (0.11852 - 0.05478 ln PD)^2; K = LGD [Phi((Phi^-1(PD) + sqrt(R) Phi^-1(0.999))/sqrt(1-R)) - PD]
   * (1 + (M-2.5) b)/(1 - 1.5 b); RWA = 12.5 K EAD; capital = 8% RWA = K EAD. SME: R -= 0.04(1-(S-5)/45),
   S in [5,50] EUR mn. Retail: mortgage R=0.15, QRRE 0.04, other retail 0.03..0.16 with k=35, no
   maturity adjustment. Large regulated financials: R x 1.25 (Basel III).
5. Student-t copula (normal variance mixture): W ~ chi2(nu)/nu ONE per scenario, shared by all factors
   and shocks; A_i = [sqrt(rho) M + sqrt(1-rho) eps_i]/sqrt(W); threshold t_nu^-1(PD_i).
6. CreditMetrics migration thresholds (Gupton et al 1997): for origin row p (best..worst, D last),
   default iff Z < Phi^-1(p_D); worst-first cumulative: z_k = Phi^-1(sum_{j>=k} p_j). Right-tail variant:
   cumulate best->worst, state = first k with Z <= Phi^-1(C_k) (default = largest Z). Same distribution,
   different sample path per seed. Mark-to-model: value each bond at horizon with the forward curve of
   its new rating; defaulted bond = recovery * face.
7. Loss VaR_q = empirical q-quantile of L (loss > 0); ES_q = mean(L | L >= VaR_q).
8. Euler: ES contribution C_i = E[L_i | L >= VaR_q]; sum_i C_i = ES exactly on the same tail set.
   VaR contribution = E[L_i | L = VaR_q] (kernel/window estimate). Normal loss: C_i = w_i (Sigma w)_i z/sigma_p.
9. Gordy-Lutkebohmert GA(q) = M_tot sigma^2(q)/(2 C'(q)), sigma^2 = sum w_i^2 LGD_i^2 p_i(1-p_i),
   C' = sum w_i LGD_i phi(h_i) sqrt(rho_i)/(sqrt(1-rho_i) phi(Phi^-1(q))), sums over PD_i > 0.
10. HHI = sum s_i^2 (shares sum to 1), in [1/N, 1]; effective number = 1/HHI.
11. Single-name UL = EAD sqrt(PD sigma_LGD^2 + LGD^2 PD(1-PD)).
## Conventions and defaults when the task is silent
- Loss reported positive; "credit VaR" / "economic capital" / "unexpected loss" = VaR - EL (Basel K
  subtracts PD*LGD). Plain "VaR of the loss distribution" = quantile itself. CreditMetrics value-based
  VaR is measured from the MEAN horizon value (mean - quantile), not from today's value.
- Quantile: np.quantile linear (type 7) unless "order statistic" / "k-th worst" is stated.
- ES uses >= VaR (includes the VaR atom). Loss distributions are discrete: ties at VaR matter.
- Maturity M = 2.5 y if not given (foundation IRB). PD floor 0.03% (Basel II; 0.05% Basel III 2017).
  Confidence 99.9% for IRB, 1-year horizon.
- Default-only loss model unless "revalue", "mark-to-market", "forward curve" or spreads per rating are
  given; then r and horizon are used for discounting.
- Seeds: np.random.default_rng(seed); draw order = the order the task lists (factor first, then
  idiosyncratic) and array shape (n_sim, n_obligors) in C order. Wrong-way / stress reruns reuse the
  base seed (common random numbers); the t run uses the stated offset seed.
- Loss std ddof=1 for sample statistics; skew/kurt from scipy (excess kurtosis, Fisher).
## Completing missing inputs
- rho from IRB formula when only PD is given; LGD 45% senior unsecured, 75% subordinated (F-IRB).
- PD from rating via the transition matrix default column; PD from spread: s/(1-R).
- Sector map by LABEL into the correlation matrix (it may contain unused sectors); keep full Cholesky.
- PD = 0: never defaults, excluded from Phi^-1 sums. Boosted rho capped (e.g. 0.999) before sqrt(1-rho).
## Invariants to self-check
- MC mean loss / analytic EL within ~2% for EVERY copula (fails -> wrong threshold quantile).
- VaR_q <= ES_q; both non-decreasing in q; VaR <= sum EAD LGD.
- sum Euler contributions / ES = 1; sector shares sum to 1; HHI in [1/N,1].
- t-copula VaR at 99.9% > Gaussian (tail dependence); wrong-way ES > base ES.
- Higher rho: same EL, fatter tail. rho = 0 -> loss is sum of independent Bernoullis.
- IRB: R in [0.12, 0.24] decreasing in PD; 0 <= K <= LGD; K increasing in PD for typical PD.
- Migration partition: upgrade + stay + downgrade(excl. default) + default = 1; simulated bucket PD
  ~ matrix PD (copula preserves marginals).
## Variants to decide
- V1 credit_var_def: what "credit VaR" measures. Options: (a) quantile - EL (unexpected loss); (b) quantile itself. Default when silent: (a) - Basel IRB / CreditMetrics (Gupton et al. 1997).
- V2 quantile_method: loss quantile. Options: (a) np.quantile linear; (b) order statistic / k-th worst. Default when silent: (a) - Hyndman-Fan type 7.
- V3 es_tail: ES tail inclusion. Options: (a) mean of losses >= VaR; (b) > VaR strict. Default when silent: (a) - McNeil-Frey-Embrechts 2005.
- V4 pd_floor: PD floor. Options: (a) 0.03%; (b) 0.05%. Default when silent: (a) - Basel II 2006; (b) Basel III 2017.
- V5 default_threshold: default trigger. Options: (a) Gaussian Phi^-1(PD); (b) t-copula t_nu^-1(PD). Default when silent: (a) - Vasicek 2002 / CreditMetrics.
- V6 loss_mode: loss model. Options: (a) default-only; (b) mark-to-market migration. Default when silent: (a) - Vasicek 2002.
- V7 migration_thresholds: Z-to-rating map. Options: (a) default in the left tail with worst-first cumulative thresholds; (b) best-first cumulation from the right tail. Default when silent: (a) - Gupton-Finger-Bhatia 1997.
- V8 bucket_loss: loss by initial rating. Options: (a) mean over simulations of the bucket total loss (buckets sum to portfolio mean); (b) mean per bond in the bucket. Default when silent: (a) - CreditMetrics TD contribution reporting.
- V9 euler_tail_set: scenarios for contributions. Options: (a) L >= VaR, same rows as CVaR; (b) L > VaR. Default when silent: (a) - Tasche 2008.
- V10 mc_memory: n_sim x N too big. Options: (a) row chunks from one generator in the stated draw order, second pass with the same seed for contributions; (b) full float32 matrix. Default when silent: (a) - Glasserman 2004 (common random numbers).

## Pitfalls
- Normal quantile threshold under the t copula (breaks marginals); W per obligor instead of per
  scenario; scaling only M or only eps.
- Euler contributions from a different scenario set than ES; chunk draws without changing RNG order.
- Using Phi^-1(1-q) vs Phi^-1(q) sign confusion in p_q: p_q must exceed PD for q > 0.5.
- Mixing $ and millions ("face 1 = 1 million"); PD column max > 1 -> percent, divide by 100.
- Counting default as downgrade; forgetting clip of cumulative prob 1 -> Phi^-1 = inf.
- Broadcasting a [ratings x states] threshold matrix against (n_sims, n_bonds) crashes: index thresholds by each bond's initial rating (simulate_migrations). Self-checks like upgrades + downgrades + defaults = all pairs are false (most bonds stay).
- Multi-factor default MC: one chunked copula_default_mc call per scenario; never a Python loop over paths or a full float latent matrix.
- euler_es_contributions returns a 3-tuple (contributions, VaR, ES): unpack it; np.array of the tuple raises 'inhomogeneous shape'.
## Sources
Gupton-Finger-Bhatia 1997 CreditMetrics Technical Document; Vasicek 2002 (Risk); Gordy 2003 (JFI);
BCBS 128 (2006) Basel II paras 272-328; BCBS 2017 Basel III finalisation; Gordy-Lutkebohmert 2013;
McNeil-Frey-Embrechts 2015 ch.11; Tasche 2008 (Euler allocation); Li 2000 (Gaussian copula).
