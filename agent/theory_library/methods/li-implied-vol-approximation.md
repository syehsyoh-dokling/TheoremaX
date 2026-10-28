method: li-implied-vol-approximation
family: implied-vol-and-parity
keywords: li approximation, li atm, li non-atm, li 2005, trigonometric implied vol, cubic implied vol, rational approximation, closed-form implied volatility, implied vol approximation, arccos domain, rho threshold, sigma_li_atm, sigma_li_nonatm
---
## Formulas
Notation: S~ = S e^{-qT}, K~ = K e^{-rT}, C call price, x = sigma sqrt T, sigma = x / sqrt T.
1. Li ATM (K~ = S~): alpha = sqrt(2 pi) C / S~; z = cos( arccos(3 alpha / sqrt 32) / 3 ); x = 2 sqrt2 z - sqrt(8 z^2 - 6 alpha / (sqrt2 z)). Not applicable (NaN) when |3 alpha / sqrt32| > 1.
2. Li non-ATM: eta = K~/S~, c = C/S~, alpha = sqrt(2 pi) (2c + eta - 1)/(1 + eta), rho = |eta - 1| / c^2.
   rho <= 1.4: the ATM cubic of item 1 with this alpha.
   rho > 1.4: x = [alpha + sqrt(alpha^2 - 4 (eta - 1)^2 / (1 + eta))] / 2.
3. A task that spells its own regime rule or coefficients wins; Li 2008 rational-function inversion is a different method (many fitted coefficients).
## Variants to decide
- V1 li_discriminant: negative term under a square root. Options: (a) set to 0 only when within a tiny tolerance of 0, otherwise NaN and exclude from stats; (b) floor every negative value at 0. Default when silent: (a) - Li 2005 (formula valid only on its domain).
- V2 li_inputs: price and level inputs. Options: (a) discounted S~ and K~; (b) raw S and K. Default when silent: (a) - Li 2005 / Chin-Olafsson-Nel.
- V3 li_atm_rows: which rows count as ATM. Options: (a) K = S0 with np.isclose; (b) K~ = S~ (ATM forward). Default when silent: (a) - task grid moneyness K/S0.
## Invariants
- Error small near ATM and grows with |ln(K~/S~)| and with x; NaN only where a domain test failed.
- RMSE and max-abs statistics over finite values only; the count of NaN rows is reportable.
## Pitfalls
- Flooring a clearly negative discriminant fabricates a value and changes RMSE.
- Using the quadratic branch for all rows, or the cubic for all rows.
## Sources
- Li (2005) Applied Mathematics and Computation; Chin, Olafsson, Nel, Handbook of Computational Finance Vol II, 7.2.1.
