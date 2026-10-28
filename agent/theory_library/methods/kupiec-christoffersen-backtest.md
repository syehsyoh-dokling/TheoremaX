method: kupiec-christoffersen-backtest
family: var-backtesting-and-data-prep
keywords: kupiec, unconditional coverage, proportion of failures, christoffersen, independence test, conditional coverage, lr_uc, lr_ind, lr_cc, violation clustering, rolling one day ahead backtest, var violations, 0 ln 0
---
## Formulas
1. I_t = 1{L_t > VaR_t}; x = sum I; T forecast days; p = 1 - c.
2. LR_uc = -2[(T-x) ln(1-p) + x ln p - (T-x) ln(1-x/T) - x ln(x/T)], chi2(1).
3. Over T-1 pairs: n00, n01, n10, n11; pi01 = n01/(n00+n01), pi11 = n11/(n10+n11), pi = (n01+n11)/(T-1).
4. LR_ind = -2[(n00+n10) ln(1-pi) + (n01+n11) ln pi] + 2[n00 ln(1-pi01) + n01 ln pi01 + n10 ln(1-pi11) + n11 ln pi11], chi2(1).
5. LR_cc = LR_uc + LR_ind, chi2(2); p = chi2.sf; 0 ln 0 = 0 (scipy.special.xlogy).
6. Rolling: VaR_t from observations t-W..t-1; T = n - W.
## Variants to decide
- V1 pi_ind: pooled rate. Options: (a) from T-1 transitions; (b) x/T. Default when silent: (a) - Christoffersen 1998.
- V2 hit_rule: inequality. Options: (a) strict L > VaR; (b) L >= VaR. Default when silent: (a) - Kupiec 1995.
- V3 window_end: data for VaR_t. Options: (a) t-W..t-1; (b) t-W+1..t. Default when silent: (a) - Christoffersen 2012.
- V4 refit: rolling model. Options: (a) refit each window; (b) fit once. Default when silent: (a) - Christoffersen 2012.
## Invariants
- LR >= 0; p in [0,1]; n00+n01+n10+n11 = T-1; x = 0 gives finite LR_uc.
## Pitfalls
- ln 0 = NaN; chi2.cdf for sf; n - W + 1 tests; window containing day t.
- Rolling GARCH refits are slow: warm-start from the previous fit.
## Sources
- Kupiec 1995; Christoffersen 1998; Christoffersen 2012 ch.13; Jorion ch.6.
