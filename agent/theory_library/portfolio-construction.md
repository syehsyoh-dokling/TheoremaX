family: portfolio-construction
category: portfolio
keywords: mean-variance, minimum variance, tangency, risk parity, equal risk contribution, inverse volatility, black-litterman, views, idzorek, tau, kelly, half kelly, var constraint, volatility targeting, regime, hmm, baum-welch, viterbi, cvar, expected shortfall, absorption ratio, brinson, attribution, allocation effect, selection effect, brinson-fachler, volatility targeting, vol target, cta, risk budget, kelly var cap, monte carlo wealth
---
## Worked frame
DIKETAHUI: returns [T x N] (decimal per period), or mu [N] and Sigma [N x N]; views P [V x N], Q [V]; sectors [S]. Check: sorted dates; mu, Sigma, weights in one asset order.
DITANYAKAN: weights [N] (fractions), risk contributions [N], return and vol (scalars), states [T], attribution per sector [S].
RUMUS: card Core 1-2 (MV, ERC), 3-5 (BL), 6-8 (sizing), 9 (HMM), 11 (Brinson); toolkit component_var, kelly_fraction, brinson_attribution.
JAWAB: 1) returns [T x N]. 2) mu [N], Sigma [N x N] in one frequency. 3) solve weights [N]. 4) Sigma w [N] -> contributions [N], vol scalar. 5) lagged weights x returns -> [T]. 6) annualise mean and vol. 7) check: sum w = 1, contributions sum to vol, effects sum to Rp - Rb; then write.

## Core formulas
1. MV utility w'mu-(delta/2)w'Sw -> w*=(1/delta)S^-1 mu. GMV w=S^-1 1/(1'S^-1 1). Tangency w=S^-1(mu-rf)/1'S^-1(mu-rf).
2. Risk contribution RC_i=w_i(Sw)_i/sqrt(w'Sw), sum RC=sigma_p. ERC: all RC equal; solve (Spinu) min 0.5y'Sy-sum b_i ln y_i, w=y/sum y, or Newton/SLSQP. Inverse-vol w_i=(1/sigma_i)/sum(1/sigma_j) equals ERC only under equal correlations.
3. Black-Litterman: Pi=delta S w_mkt (no tau). mu_BL=[(tS)^-1+P'O^-1P]^-1[(tS)^-1 Pi+P'O^-1 Q]. M=[(tS)^-1+P'O^-1P]^-1. He-Litterman weights use S_post=S+M: w=(delta S_post)^-1 mu_BL; simple tasks use (delta S)^-1 mu_BL - follow text.
4. Omega default (He-Litterman): diag(P tS P'). Idzorek confidence c in (0,1]: omega_k=((1-c)/c) p_k tS p_k' (c=1 -> 0, singular; use the limit mu=view). Stated view std s_k -> omega_k=s_k^2.
5. delta=(E[r_m]-rf)/sigma_m^2 (~2.5-3). tau: 0.025-0.05 (He-Litterman 0.025) or 1/T; use the given value.
6. Kelly: binary f=p-q/b (b net odds). Continuous single f=(mu-r)/sigma^2; multi f=S^-1(mu-r); growth g=r+f'(mu-r)-f'Sf/2. Half Kelly=0.5f. Leverage=sum|f| may exceed 1.
7. Parametric VaR (positive loss) of w: z_c sqrt(w'Sw)-w'mu; degree-1 homogeneous, so cap by s=min(1,VaR_max/VaR(f)).
8. Vol target: scale=min(cap, sigma_tgt/sigma_hat); ex-ante sigma_hat=sqrt(P w'S_trail w) with full covariance (not diagonal, not realised strategy std) unless text says realised/EWMA (lambda 0.94).
9. Gaussian HMM (log-space): log a_t(k)=log b_k(x_t)+lse_j(log a_{t-1}(j)+log A_jk); log beta_t(k)=lse_j(log A_kj+log b_j(x_{t+1})+log beta_{t+1}(j)); gamma=exp(a+beta-logL). M-step: mu_k=sum g x/sum g; sigma_k^2=sum g(x-mu_k)^2/sum g (MLE, no ddof); A_ij=sum_{t<T-1} xi_t(i,j)/sum_{t<T-1} g_t(i); pi=g_0. Viterbi: delta_t(j)=max_i[delta_{t-1}(i)+log A_ij]+log b_j(x_t), backtrack.
10. CVaR (Rockafellar-Uryasev): min z+1/((1-a)S) sum_s u_s, u_s>=L_s-z, u_s>=0 (LP). Historical: VaR=-q_{1-a}(r); ES=-mean(r | r<=q) (report positive).
11. Brinson (weights begin-of-period): BHB alloc (wp-wb)Rb_i; BF alloc (wp-wb)(Rb_i-Rb); selection wb(Rp_i-Rb_i); interaction (wp-wb)(Rp_i-Rb_i). BF and BHB totals agree (sum(wp-wb)=0), sectors differ. Cash: wb=0 -> alloc only.
12. Absorption ratio (Kritzman-Page) = sum top-k eigenvalues/trace (=N for a correlation matrix); Marchenko-Pastur edge (1+sqrt(N/T))^2.

## Conventions and defaults when the task is silent
- Budget sum w=1 (fully invested); gross sum|w|=1 only when long-short sizing is stated.
- Sample covariance ddof=1; annualise mu x P, S x P; keep BL in ONE frequency: daily views Q/P, omega/P^2 when confidence is an annual std.
- rf per day = rf_annual/252 simple unless stated.
- HMM: label bull = higher mean (not state index); hard assignment gamma>0.5 strict; after convergence run a final E-step with final params; iteration count = E/M passes performed; accept the new params on convergence.
- Weights drift between rebalances: w_i(1+r_i)/sum_j w_j(1+r_j); rebalance weights set at close apply from the next period.
- Historical quantile: np.quantile linear by default; Basel-style ES = mean of worst ceil((1-a)N) losses - read text.
- Monte Carlo: one seeded generator (np.random.default_rng(seed)), one draw set shared by all schemes; wealth = W0*cumprod(1+r); drawdown path includes W0.
- Multi-period attribution: plain sum of monthly effects unless linking (Carino, Menchero, GRAP) is asked.

## Completing missing inputs
- No market weights for BL: use equal weights or cap weights from data; no delta: (mu_m-rf)/var_m.
- Missing views -> mu_BL=Pi, weights = w_mkt (sanity test).
- Regime covariance with too few days: fall back to the full-sample covariance (typical threshold stated).
- Cash return annual_rate/12 when monthly and text says so.

## Invariants to self-check
- sum w=1 (or sum|w|=1); sum RC = sigma_p; ERC RC_i equal to 1e-8.
- BL posterior lies between prior and views; no views -> Pi.
- A rows and gamma rows sum to 1; 0<=gamma<=1; n_bull+n_bear=T; bull mu>bear mu.
- ES>=VaR at the same level; VaR non-decreasing in confidence; both positive magnitudes.
- Each period: sum(alloc+sel+inter)=Rp-Rb exactly; quarterly and sector sums reconcile to totals.
- Kelly scaled by s keeps direction; VaR(s f)=VaR_max when s<1.

## Variants to decide
- V1 cov_ddof: covariance estimator. Options: (a) sample ddof=1; (b) MLE ddof=0. Default when silent: (a) - Markowitz practice (np.cov).
- V2 annualisation: mean/cov scaling. Options: (a) arithmetic mu x P, Sigma x P; (b) geometric (1+mu)^P-1. Default when silent: (a) - Meucci 2005.
- V3 bl_tau: Black-Litterman tau/Omega. Options: (a) tau=0.05, Omega=diag(tau P Sigma P'); (b) tau=1/T; (c) tau=1 (Meucci). Default when silent: (a) - He-Litterman 1999.
- V4 risk_parity: risk parity definition. Options: (a) equal risk contributions (ERC); (b) inverse volatility. Default when silent: (a) - Maillard-Roncalli-Teiletche 2010.
- V5 kelly_fraction: Kelly sizing. Options: (a) full Kelly Sigma^-1(mu-rf); (b) half Kelly. Default when silent: (a) - Kelly 1956 / Thorp.
- V6 hist_quantile: historical quantile. Options: (a) np.quantile linear; (b) order statistic. Default when silent: (a) - Hyndman-Fan type 7.
- V7 brinson: attribution scheme. Options: (a) BHB allocation (w_p-w_b)R_b,i; (b) Brinson-Fachler (w_p-w_b)(R_b,i-R_b). Default when silent: (a) - Brinson-Hood-Beebower 1986.
- V8 hmm_mstep: Baum-Welch re-estimation scope. Options: (a) update pi, A (from xi), mu, then sigma with the new mu; (b) keep A and pi fixed, update emissions only. Default when silent: (a) - Rabiner 1989.
- V9 rebalance_apply: when new weights earn returns. Options: (a) weights set at close t earn from t+1; (b) same-day return t. Default when silent: (a) - no look-ahead practice (Bailey et al. 2014).
- V10 bl_view_frequency: annual views with daily covariance. Options: (a) Q/P and omega = s^2/P; (b) omega = (s/P)^2; (c) annualise Sigma instead. Default when silent: (a) - He-Litterman 1999 (one frequency, variances scale linearly).
- V11 ewma_param: EWMA vol parameter meaning. Options: (a) decay lambda, weight (1-lambda) on the newest r^2; (b) smoothing alpha = 2/(span+1) on the newest. Default when silent: (a) - RiskMetrics 1996; convert alpha to lambda = 1 - alpha.

## Pitfalls
- Tau multiplied into Pi; omega as std instead of variance; mixing annual views with daily S.
- Probability-space HMM underflow over long series; indexing states instead of means.
- Soft (probability-weighted) covariance when hard assignment is specified.
- Using excess mean inside VaR when text uses raw mean (or reverse).
- Diagonal covariance in ex-ante vol for a hedged long-short book (overstates vol).
- End-of-period weights in attribution; drifting benchmark weights that are fixed within a quarter.
- Normalising after a final turnover shrink that is meant to leave gross<1.
- Rolling-window outputs (regimes, AR) start at row W-1: index them by data row through ONE map, never mix window positions with row numbers (absorption_riskparity_report).
- Report bundles take the RAW input table: cleaning and the 'most recent N valid observations' cut happen inside, so a count 'after cleaning' is taken BEFORE the cut - slicing first turns it into N. Holdings attribution with drift, rebalance and cash: brinson_fachler_report.

## Sources
Markowitz 1952; Maillard-Roncalli-Teiletche 2010; Spinu 2013; Black-Litterman 1992; He-Litterman 1999; Idzorek 2005; Walters 2014; Kelly 1956; Thorp 2006; Rabiner 1989; Hamilton 1989; Rockafellar-Uryasev 2000; Brinson-Hood-Beebower 1986; Brinson-Fachler 1985; Kritzman-Li-Page-Rigobon 2011.
