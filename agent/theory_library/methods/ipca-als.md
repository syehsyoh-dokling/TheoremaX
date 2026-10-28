method: ipca-als
family: factor-models-and-sorts
keywords: ipca, instrumented pca, instrumented principal components, kelly pruitt su, latent factors, gamma matrix, characteristic-based loadings, alternating least squares, managed portfolios, total r2, predictive r2, grs test
---
## Formulas
1. r_{i,t+1} = z_{i,t}' Gamma f_{t+1} + e; Z_t [N_t x L], Gamma [L x K]; first month lost to the lag.
2. Init: Gamma_0 = top-K left singular vectors of managed portfolios Z_t' r_{t+1}/N_t (or seeded random if stated).
3. f-step: f = (Gamma' Z'Z Gamma)^-1 Gamma' Z' r.
4. Gamma-step: vec(Gamma) = [sum Z'Z kron f f']^-1 sum vec(Z' r f') (kron order matches the vec reshape).
5. Identify each sweep; stop when max|dGamma| (and/or dF) < tol.
6. Total R2 = 1 - sum(r - Z Gamma f)^2/sum r^2 (uncentred); predictive uses mean(f).
## Variants to decide
- V1 char_transform: characteristic scaling. Options: (a) rank mapped to [-0.5, 0.5]; (b) z-score; (c) raw. Default when silent: (a) - Kelly-Pruitt-Su 2019.
- V2 r2_centering: R2 denominator. Options: (a) uncentred sum r^2; (b) centred. Default when silent: (a) - Kelly-Pruitt-Su 2019.
- V3 ipca_init: starting Gamma. Options: (a) SVD of managed portfolios; (b) seeded random. Default when silent: (a) - Kelly-Pruitt-Su 2019.
- V4 ipca_rotation: identification. Options: (a) Gamma'Gamma = I, F'F/T diagonal descending; (b) eigenvectors of uncentred F'F/T keeping that scale; signs mean(f) >= 0 in both. Default when silent: (a) - Kelly-Pruitt-Su 2019.
## Invariants
- Fitted values and R2 invariant to rotation; objective non-increasing.
## Pitfalls
- Pairing z_t with r_t; multi-column pivots give (char, id) columns, build [T x N x L] explicitly.
- Using a generic seed key when an init-specific seed is given.
## Sources
- Kelly, Pruitt, Su 2019 (JFE); ipca package (bkelly-lab); Gibbons-Ross-Shanken 1989.
