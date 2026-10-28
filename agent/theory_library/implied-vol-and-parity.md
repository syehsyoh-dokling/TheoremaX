family: implied-vol-and-parity
category: derivatives-pricing
keywords: implied volatility, newton-raphson, brent, bisection, brenner-subrahmanyam, corrado-miller, hallerbach, li approximation, put-call parity, synthetic forward, discount factor, implied dividend, implied borrow, bid ask, stale quote, crossed quote, one-sided, option chain, inverse option, coin-margined, forward audit, violation, li non-atm, corrado-miller-hallerbach, newton-raphson implied vol, synthetic forward bounds, implied borrow, parity audit
---
## Worked frame
DIKETAHUI: chain one row per quote [M] (expiry, strike, type, bid, ask, timestamp); S, r, q scalars. Check: flag one-sided, crossed, stale in that order; unique (expiry, strike, type).
DITANYAKAN: IV per row [M] (decimal); per expiry [E]: forward, DF, implied yield; audit one row per (expiry, strike) [P]; counts.
RUMUS: card Core 5-8 (parity, forward), 1 (solver), 2-4 (approximations); toolkit implied_vol, bs_price.
JAWAB: 1) T in years [M]. 2) clean mask [M], counts. 3) pair call and put per (expiry, strike) [P]. 4) forward, DF [E]. 5) IV per clean row [M], NaN outside bounds. 6) coin premium -> currency; vol as decimal. 7) check: BS(IV) reprices, F_bid <= F_ask, exclusions + clean = total; then write.

## Core formulas
Notation: S~ = S e^{-qT}, K~ = K e^{-rT}, C market call, x = sigma sqrt T.
1. Newton: sigma_{n+1} = sigma_n - (BS(sigma_n) - V)/vega(sigma_n), vega = S~ n(d1) sqrt T. Safeguard: if vega < 1e-8 or step leaves (1e-6, 5), fall back to bisection/brentq on [1e-6, 5]. Start sigma0 = 0.2..0.3 or Manaster-Koehler sigma0 = sqrt(2|ln(F/K)|/T).
2. Brenner-Subrahmanyam (ATM-forward): sigma = sqrt(2 pi / T) C / S~.
3. Corrado-Miller (1996): x = sqrt(2pi)/(S~ + K~) [C - (S~-K~)/2 + sqrt((C - (S~-K~)/2)^2 - (S~-K~)^2/pi)].
   Hallerbach (2004) variant: x = sqrt(2pi)/(2(S~+K~)) [2C + K~ - S~ + sqrt((2C + K~ - S~)^2 - 1.85 (S~+K~)(K~-S~)^2/(pi sqrt(K~ S~)))]. Floor the discriminant at 0.
4. Li (2005) ATM: alpha = sqrt(2pi) C/S~; z = cos(arccos(3 alpha/sqrt 32)/3); x = 2 sqrt2 z - sqrt(8 z^2 - 6 alpha/(sqrt2 z)). NaN if |3 alpha/sqrt32| > 1.
   Li non-ATM: eta = K~/S~, c = C/S~, alpha = sqrt(2pi)(2c + eta - 1)/(1 + eta), rho = |eta - 1|/c^2. rho <= 1.4: use the ATM cubic with this alpha; rho > 1.4: x = [alpha + sqrt(alpha^2 - 4(eta-1)^2/(1+eta))]/2. Other textbook variants exist - if the task spells a formula, follow it verbatim.
5. Put-call parity (European): C - P = S e^{-qT} - K e^{-rT} = DF (F - K). Synthetic forward F = K + (C - P)/DF.
6. Executable synthetic forward: F_bid = K + (C_bid - P_ask)/DF, F_ask = K + (C_ask - P_bid)/DF. Violation of a theory forward F_th = max(F_bid - F_th, F_th - F_ask, 0).
7. Carry: F = S exp((r - q - b_borrow) T); implied q (or borrow) = r - ln(F/S)/T; implied rate r = -ln(DF)/T.
8. Multi-strike extraction: regress (C_mid - P_mid) on K per expiry: slope = -DF, intercept = DF F.
9. Inverse / coin-margined (Deribit-style) quotes: premium in coin; USD premium = premium_coin x reference price (the quote's underlying/futures or index price as the task defines). Parity in USD: C_usd - P_usd = DF (F - K); in coin units divide by the reference: C_c - P_c = DF (F - K)/F_ref. Do not apply USD parity to raw coin premiums.
## Conventions and defaults when the task is silent
- Time: ACT/365 on exact timestamps (seconds / (365 x 86400)); expiry timestamps are UTC; keep the original ISO string with trailing Z for output.
- Quote cleaning precedence when rules overlap and the task says each row is counted once: one_sided (bid <= 0 or ask <= 0 or NaN) -> crossed (bid >= ask; locked counts as crossed only if the rule says >=) -> stale (age > max age, strict). Clean = no flag. Apply masks with & ~earlier.
- Pairing only among CLEAN quotes on (expiry, strike); missing-pair count = keys on exactly one clean side (symmetric difference).
- Thresholds strict ">" unless ">=" is written. Sort by expiry then numeric strike.
- IV input: invert on mid unless told bid/ask; use BS with spot, r, q (Black-76 on F only if the task gives F and says so). Target tolerance 1e-8 on price.
- Approximations take S~ and K~ (discounted), not raw S and K, unless the task writes raw.
- Prices below intrinsic (C < max(0, S~ - K~)) or above S~ have no IV: return NaN, count them, do not clip silently.
## Completing missing inputs
- Missing r: from parity pairs (item 8) or a given forward: DF = (C - P)/(F - K).
- Missing q: from F and S (item 7). Missing F: ATM pair with smallest |C - P|, F = K + (C - P)/DF.
- ATM IV: interpolate IV linearly in strike between the two strikes straddling F (K_lo < F <= K_hi) unless told total-variance or log-moneyness.
## Invariants to self-check
- BS(sigma_implied) reprices input within tol; IV of a call and put at the same K equal (if parity holds).
- Newton converged on all points; errors of approximations largest far from ATM and at high x.
- F_bid <= F_mid <= F_ask for clean quotes; violation = 0 iff F_th in [F_bid, F_ask]; violations table = subset of audit table with amount > threshold.
- Exclusion counts + clean count = total rows.
- Put-call parity on mids reproduces F_mid to machine precision.
## Variants to decide
- V1 iv_input: price inverted. Options: (a) mid; (b) bid and ask separately; (c) last trade. Default when silent: (a) - Hull.
- V2 time_basis: time to expiry. Options: (a) ACT/365 on exact timestamps; (b) trading days/252. Default when silent: (a) - Hull / Cboe VIX methodology (minutes/525600).
- V3 pricing_model: model inverted. Options: (a) BS with spot, r, q; (b) Black-76 on an implied forward. Default when silent: (a) - Black-Scholes 1973 / Merton 1973.
- V4 approx_inputs: closed-form IV approximations. Options: (a) discounted S~, K~; (b) raw S, K. Default when silent: (a) - Corrado-Miller 1996.
- V5 solver: IV root finder. Options: (a) Newton-vega with Brent fallback, tol 1e-8 on price; (b) bisection. Default when silent: (a) - Brent 1973 / Hull.
- V6 approx_discriminant: negative root term in a closed-form IV approximation. Options: (a) floor at 0; (b) NaN unless within a tiny tolerance of 0, then exclude from RMSE. Default when silent: (a) - Corrado-Miller 1996 (Li formulas follow (b): Li 2005 domain).
- V7 newton_converged_flag: convergence flag. Options: (a) from the iteration itself: all |sigma_{n+1} - sigma_n| < tol before max_iter; (b) a reprice/error check afterwards. Default when silent: (a) - the stated stopping rule (Hull).
- V8 missing_pair_unit: unmatched count. Options: (a) clean quotes lacking a clean opposite type at the same (expiry, strike); (b) distinct (expiry, strike) keys. Default when silent: (a) - quote-level audit wording.

## Pitfalls
- Swapping legs: bid bound uses call BID and put ASK.
- Multiplying by DF instead of dividing ((C - P) e^{+rT}).
- Borrow cost sign: it lowers the forward like a dividend.
- Using 365.25, trading days, or date-only differences for intraday expiries.
- Output: CSV numerics as fixed-decimal strings when required (f"{x:.6f}"); JSON floats rounded as numbers; numpy ints cast to int.
- Dict key names in intermediate JSON may differ from summary key names (e.g. reason labels vs *_count) - read each schema separately.
- Newton with all-points convergence criterion: iterate vectorised until max|step| < tol, not per-point early exit, when the task defines "all()".
- Deep OTM short-dated: vega ~ 0, Newton diverges - bracket method.
- BS/Li ATM formulas are exact-ish only at K~ = S~ (ATM-forward). Applied at K = S0 with r != q they carry a real bias (several vol points); that is the expected error, not a bug to "fix".
- np.maximum(a, b, 0.0) passes 0.0 as out= (TypeError): nest np.maximum(np.maximum(a, b), 0.0). newton_iv_vec returns (sigma, converged, iterations) - unpack it.
## Sources
Hull (ch. 11, 20); Brenner and Subrahmanyam (1988); Corrado and Miller (1996); Hallerbach (2004); Li (2005, Applied Math and Computation); Manaster and Koehler (1982); Jaeckel "Let's be rational" (2015); Deribit contract specifications (inverse options).
