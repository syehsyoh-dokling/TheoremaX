family: factor-models-and-sorts
category: factor-research
keywords: fama-french, three-factor, five-factor, smb, hml, alpha, jensen, newey-west, hac, fama-macbeth, grs, vif, double sort, quintile, breakpoints, residual momentum, winsorize, z-score, cross-sectional, ipca, pca, beta-neutral, long-short, information coefficient, rolling beta, lead-lag, cross-correlation, fisher z, sentiment factor, signal fusion, alpha hedge, stable residual, sign stability, alternating least squares
---
## Worked frame
DIKETAHUI: panel (date, id, return, signals) -> [T x N]; factors [T x K] (percent -> decimal); risk-free [T]. Check: sorted dates, unique (date, id), integer ids.
DITANYAKAN: one row per asset (alpha, betas [K], t-stats, R2); bucket returns [T x Q], long-short [T]; weights per (date, id).
RUMUS: card Core 1-4 (regression, HAC), 5 (Fama-MacBeth), 8-10 (sorts, neutral); toolkit ols, newey_west_tstat, spearman_ic.
JAWAB: 1) align y [T] with X [T x K]. 2) excess returns [T]. 3) per date: winsorise, z-score, rank -> bucket [N_t]. 4) join signal at t with return t+1. 5) coefficients [K+1], or slopes [T] -> mean. 6) annualise alpha; decimal -> percent if asked. 7) check: residual mean ~ 0, sum w = 0, sum|w| = 1; then write.

## Core formulas
1. Time-series: R_i-RF = a + b'F + e, X=[1,F] (prepend ones yourself). k=#factors: s2=SSR/(n-k-1), Var=s2(X'X)^-1, t-dist df n-k-1; adjR2=1-(1-R2)(n-1)/(n-k-1). alpha_ann = a*252 (daily) or a*12 (monthly), linear.
2. NW HAC for coefficients: S = sum e_t^2 x_t x_t' + sum_{l=1..L} w_l sum_{t>l} e_t e_{t-l}(x_t x_{t-l}' + x_{t-l} x_t'), w_l = 1-l/(L+1). V=(X'X)^-1 S (X'X)^-1. Raw sums: no 1/n, no n/(n-k).
3. NW for a mean (FM slope series, T periods): u=g-mean(g); gamma_l=(1/T) sum_{t>l} u_t u_{t-l}; S=gamma_0+2 sum w_l gamma_l; t=mean(g)/sqrt(S/T). 1/T is NW-1987 textbook; statsmodels HAC (OLS on constant) gives other small-sample numbers.
4. Lag rule of thumb L=floor(4(T/100)^(2/9)) with FLOAT 2/9. Alternatives: floor(T^(1/4)); lag stated in text wins.
5. Fama-MacBeth: per date CS OLS of r(t->t+1) on [1,z_t]; lambda=mean_t lambda_t; plain se=std(lambda_t,ddof=1)/sqrt(T).
6. GRS=((T-N-K)/N) a'Sig^-1 a/(1+mu'Om^-1 mu), Sig=E'E/T, Om=(F-mu)'(F-mu)/T (MLE 1/T); ~F(N,T-N-K), p=f.sf. Same as unbiased Sig times (T/N)(T-N-K)/(T-K-1).
7. VIF_j=1/(1-R2_j) (aux regression with intercept); DW=sum(diff e)^2/sum e^2 in [0,4]; IR=alpha_ann/(std(e,ddof=1)sqrt(ann)); Treynor=ann excess ret/beta_mkt.
8. Rank quantile: sort (signal asc, id asc), 0-based r, q=min(Q, floor(Q r/n)+1). Dependent double sort: bucket on var1, re-sort each bucket on var2 with its own n. Independent: breakpoints of each var on the full cross-section, intersect.
9. Beta/dollar-neutral projection: w - X(X'X)^+ X'w, X=[1,beta] on selected names; then scale so sum|w|=1.
10. Min-norm hedge: min||h||^2 s.t. V(w+h)=0, 1'h=0 -> lstsq on stacked [V;1'] (h=A'(AA')^-1 b).
11. Residual momentum (Blitz-Huij-Martens 2011): FF3 regression over 36m, residuals summed t-12..t-2, divided by residual std of the window.
12. IPCA (Kelly-Pruitt-Su 2019): r_{i,t+1}=z_{i,t}'Gamma f_{t+1}. ALS: f_t=(G'Z'ZG)^-1 G'Z'r_t; vec(G) solves sum_t(Z'Z kron ff')vec(G)=sum_t vec(Z'r f'). Identify each sweep: G'G=I (Cholesky), F'F/T diagonal descending, flip so mean(f_k)>=0.

## Conventions and defaults when the task is silent
- Signal at t, return t->t+1 (ret_lead1m). Current-month return in a signal is look-ahead.
- CS standardisation: z=(x-mean)/std, ddof=0 (population); std 0 -> z=0. Winsorise per date at 1/99 pct (np.percentile linear) on the eligible universe BEFORE residualising or z-scoring.
- 12-1 momentum = t-12..t-2, skip t-1; compound unless text says sum.
- Fama-French breakpoints: NYSE-only (size median; B/M 30/70), value weights with lagged (t-1) market cap. No exchange field -> all-stock breakpoints, equal weights.
- Ties: integer id ascending (cast float ids to int; never string order). np.lexsort((ids, sig)) or kind=stable.
- Factor file units: Ken French CSV is percent (0.5 means 0.5%) -> /100. Values ~0.005 are already decimal. Statement wins.
- Annualise factor means x12/x252, Sharpe sqrt(12)/sqrt(252).
- Eigen/PCA signs are arbitrary: use the stated rule (sklearn svd_flip, mean>=0, corr with anchor>=0). Predictions, R2 and neutral weights are sign-invariant; loadings, scores and Sharpe are not.
- Turnover one-way=0.5 sum|dw| over union of names; cost usually bps x sum|dw| (two-way) - read which. First book vs zero book. Empty/skipped month: keep last EMITTED book.
- Rolling window label = window END date; n-W+1 windows.
- Expanding OOS: coefficients from valid months strictly before t; append t after predicting.

## Completing missing inputs
- No RF column: regress raw returns and say so; RF given -> always excess returns.
- Neutralise with the raw beta, and report exposure with the SAME beta vector (exposure then ~1e-16).
- Even-count median = mean of two middle values (np.median).
- Missing lead return inside a held book: 0 if stated, never drop the weight.
- Month enters a regression only if n >= minimum after dropping NaN in target and all regressors of THAT model.

## Invariants to self-check
- Residuals orthogonal to X, mean ~0; 0<=R2<=1; adjR2<=R2; VIF>=1; DW in [0,4].
- Long-short: sum w=0, sum|w|=1, beta'w~0 after projection.
- Quantile bucket sizes differ by <=1; Q1 = lowest signal; some small buckets have no Q5 (skip, do not force).
- IC in [-1,1]; Spearman = Pearson on average ranks; undefined (empty cell) if a vector is constant.
- PCA loadings unit norm and orthogonal; explained ratios descending, sum<=1.
- IPCA: total R2 and GRS invariant to rotation; Sharpe/variance shares are not.

## Variants to decide
- V1 nw_lag: Newey-West lag. Options: (a) task-given lag; (b) floor(4 (T/100)^{2/9}); (c) floor(T^{1/4}). Default when silent: (b) - Newey-West 1994.
- V2 cs_zscore: cross-sectional std. Options: (a) ddof=0 (population); (b) ddof=1. Default when silent: (a) - Grinold-Kahn practice.
- V3 winsorise: per-date clipping. Options: (a) 1/99 pct (linear percentile) before residualising; (b) 0.5/99.5; (c) none. Default when silent: (a) - Bali-Engle-Murray 2016.
- V4 breakpoints: sort breakpoints and weights. Options: (a) NYSE breakpoints, value weights with lagged ME; (b) all-stock breakpoints, equal weights. Default when silent: (a) - Fama-French 1993.
- V5 momentum_window: momentum signal. Options: (a) compounded t-12..t-2 (skip t-1); (b) summed returns; (c) t-12..t-1 no skip. Default when silent: (a) - Jegadeesh-Titman 1993 / Fama-French.
- V6 fm_se: Fama-MacBeth standard error. Options: (a) std(ddof=1) of period coefficients / sqrt(T); (b) Newey-West adjusted. Default when silent: (a) - Fama-MacBeth 1973.
- V7 regression_sample: rows per cross-section. Options: (a) drop NaN in target and every regressor of that model, then apply the minimum-n gate; (b) drop NaN in target and signal only. Default when silent: (a) - Fama-MacBeth 1973 (complete cases).
- V8 lead_lag_orientation: lag sign. Options: (a) corr(A_t, B_{t+k}), k > 0 means A leads; (b) corr(A_{t+k}, B_t). Default when silent: (a) - Box-Jenkins ccf convention.
- V9 ipca_identification: IPCA normalisation. Options: (a) Gamma'Gamma = I, F'F/T diagonal descending, mean(f) >= 0; (b) rotation by eigenvectors of uncentred F'F/T keeping that scale. Default when silent: (a) - Kelly-Pruitt-Su 2019.
- V10 signal_lag_once: timing of weights vs returns. Options: (a) weights at t earn r_{t+1}, one shift only; (b) shift both weights and returns (double lag). Default when silent: (a) - Grinold-Kahn 2000.

## Pitfalls
- 2/9 integer division; Bartlett 1-l/L; dividing HAC sums by n; normal instead of t p-values.
- GRS with T-1 covariances, or factor covariance not demeaned (mu itself uses raw means).
- String ids ("10"<"9"); pandas rank(method=average) breaks deterministic buckets.
- Pooled panel OLS instead of per-date CS regressions.
- Winsorising after z-scoring; ddof=1 z-scores when population is stated.
- Beta source must be decided per step: a task that says "post-winsorization beta" only for the exposure and plain "beta" for the beta-neutral projection projects on the RAW beta (know77 replay); keep both switches explicit (stable_residual_backtest projection_beta / exposure_beta).
- A backtest helper that already returns the factor regression must not be followed by a second regression on the full-sample factor table (rows differ from the trading-period returns); helper tables are DataFrames - pass out['signal_return'], never out.values.
- IPCA without per-sweep normalisation: scale drifts, diagnostics depend on seed.
- Hyphenated tickers sort by ASCII ("BRK-B" < "GOOGL").
- A multi-column pivot of characteristics flattens to [T x N*L]; build [T x N x L] with panel_to_array before IPCA. Raw return NaNs: count first, then nan_to_num before signals and backtests (one NaN makes every metric NaN).
- Skip-month momentum "t-12 through t-2" is rolling(11).sum().shift(2), not shift(1); rank across the stocks of one date (axis=1), never one stock through time. One intercept only: a ones column plus add_const=True is singular and shifts every slope.

## Sources
Fama-French 1993/2015; Fama-MacBeth 1973; Newey-West 1987/1994; Gibbons-Ross-Shanken 1989; Jegadeesh-Titman 1993; Blitz-Huij-Martens 2011; Frazzini-Pedersen 2014; Kelly-Pruitt-Su 2019; Ken French data library.
