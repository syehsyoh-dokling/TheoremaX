"""Copulas: parameter <-> Kendall tau / Spearman rho, tail dependence, sampling, densities, fitting.

Families and parameter `theta` (bivariate unless stated):
    gaussian  theta = rho in (-1,1)   (d-dim sampling: scalar -> equicorrelation, or a (d,d) matrix R)
    t         theta = rho, plus nu > 0 degrees of freedom (d-dim like gaussian)
    clayton   theta > 0 (frailty, any d); -1 < theta < 0 allowed for d=2 (conditional inversion)
    gumbel    theta >= 1 (theta = 1 is independence), any d
    frank     theta != 0 (theta > 0 frailty, any d; theta < 0 for d=2 by conditional inversion)

Conventions (theory card copulas-and-dependence):
  * pseudo-observations u = rank/(n+1), average ranks for ties - never rank/n.
  * Kendall tau: gaussian/t tau = (2/pi) arcsin(rho); clayton theta/(theta+2); gumbel 1 - 1/theta;
    frank 1 - (4/theta)(1 - D1(theta)), D1 the first Debye function.
  * Tail dependence (lower, upper): gumbel (0, 2 - 2^(1/theta)); clayton (2^(-1/theta), 0);
    t both 2 t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho))); gaussian, frank (0, 0).
  * log-likelihood of the copula only (pseudo-obs in, no margins); AIC = -2LL + 2k, BIC = -2LL + k ln n,
    k = 1 (gaussian, clayton, gumbel, frank), 2 (t: rho, nu).
  * u is clipped to [eps, 1-eps] (eps = 1e-12) inside densities so no ppf/log returns inf.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate, optimize
from scipy.special import gammaln
from scipy.stats import kendalltau, norm, rankdata
from scipy.stats import t as student_t

__all__ = [
    "copula_param_from_tau", "copula_tau", "spearman_rho", "copula_param_from_spearman",
    "tail_dependence", "copula_sample", "copula_logpdf", "copula_pdf", "copula_cdf",
    "copula_loglik", "pseudo_observations", "fit_copula", "select_copula",
    "empirical_kendall_tau", "empirical_spearman_rho", "empirical_tail_dependence",
]

FAMILIES = ("gaussian", "t", "clayton", "gumbel", "frank")
_EPS = 1e-12
_ALIASES = {"normal": "gaussian", "gauss": "gaussian", "student": "t", "student_t": "t",
            "studentt": "t", "t-copula": "t", "gumbel_hougaard": "gumbel", "gumbel-hougaard": "gumbel"}


def _fam(family) -> str:
    f = str(family).strip().lower().replace(" copula", "").replace("_copula", "")
    f = _ALIASES.get(f, f)
    if f not in FAMILIES:
        raise ValueError(f"unknown copula family {family!r}; use one of {FAMILIES}")
    return f


def _need_nu(nu):
    if nu is None:
        raise ValueError("t copula needs nu (degrees of freedom)")
    nu = float(nu)
    if not nu > 0:
        raise ValueError(f"nu must be > 0, got {nu}")
    return nu


# ---------------------------------------------------------------------------------------------
# Debye functions (Frank)
# ---------------------------------------------------------------------------------------------
def _debye(k: int, x: float) -> float:
    """D_k(x) = (k/x^k) int_0^x t^k/(e^t - 1) dt; D_k(0) = 1. Valid for negative x too."""
    x = float(x)
    if abs(x) < 1e-10:
        return 1.0

    def f(t):
        return 1.0 if t == 0.0 else t ** k / math.expm1(t)
    val, _ = integrate.quad(f, 0.0, x, limit=200, epsabs=1e-14, epsrel=1e-12)
    return k * val / x ** k


def _frank_tau(theta: float) -> float:
    if abs(theta) < 1e-8:
        return theta / 9.0          # series: tau ~ theta/9
    return 1.0 - 4.0 / theta * (1.0 - _debye(1, theta))


def _frank_rho_s(theta: float) -> float:
    if abs(theta) < 1e-8:
        return theta / 6.0          # series: rho_S ~ theta/6
    return 1.0 - 12.0 / theta * (_debye(1, theta) - _debye(2, theta))


# ---------------------------------------------------------------------------------------------
# Parameter <-> rank correlation
# ---------------------------------------------------------------------------------------------
def copula_tau(family, theta, nu=None) -> float:
    """Kendall tau of the copula with parameter theta (nu is ignored: tau of t = tau of gaussian)."""
    f = _fam(family)
    if f in ("gaussian", "t"):
        return _out(2.0 / np.pi * np.arcsin(np.asarray(theta, float)))
    theta = float(theta)
    if f == "clayton":
        return theta / (theta + 2.0)
    if f == "gumbel":
        if theta < 1:
            raise ValueError("gumbel theta must be >= 1")
        return 1.0 - 1.0 / theta
    return _frank_tau(theta)


def copula_param_from_tau(family, tau):
    """Parameter theta from Kendall tau (moment / 'tau inversion').

    gaussian/t: rho = sin(pi tau/2) (elementwise for a tau matrix); clayton 2tau/(1-tau);
    gumbel 1/(1-tau) (tau < 0 has no gumbel -> ValueError; clamp to 0 yourself for independence);
    frank: numeric inversion of 1 - (4/theta)(1 - D1(theta)) by brentq (theta = 0 at tau = 0).
    """
    f = _fam(family)
    if f in ("gaussian", "t"):
        return _out(np.sin(np.pi * np.asarray(tau, float) / 2.0))
    tau = float(tau)
    if not -1.0 < tau < 1.0:
        raise ValueError(f"tau must be in (-1,1), got {tau}")
    if f == "clayton":
        return 2.0 * tau / (1.0 - tau)
    if f == "gumbel":
        if tau < 0:
            raise ValueError("gumbel needs tau >= 0 (theta >= 1); use theta=1 for independence")
        return 1.0 / (1.0 - tau)
    if abs(tau) < 1e-12:
        return 0.0
    lo, hi = (1e-10, 1.0) if tau > 0 else (-1.0, -1e-10)
    while (_frank_tau(hi) - tau) * (_frank_tau(lo) - tau) > 0:
        if tau > 0:
            hi *= 2.0
        else:
            lo *= 2.0
        if abs(hi) > 1e4 or abs(lo) > 1e4:
            raise ValueError(f"frank tau {tau} out of numerically reachable range")
    return optimize.brentq(lambda th: _frank_tau(th) - tau, lo, hi, xtol=1e-13, rtol=1e-13)


def spearman_rho(family, theta, nu=None) -> float:
    """Spearman rho_S of the copula.

    gaussian: (6/pi) arcsin(rho/2) (closed); frank: 1 - (12/theta)(D1 - D2) (closed, Debye);
    clayton, gumbel: numeric 12 * int int C(u,v) du dv - 3 (Gauss-Legendre 200x200, ~1e-7);
    t: numeric 12 * E[F(X)F(Y)] - 3 over the bivariate t density (needs nu; ~1e-6).
    """
    f = _fam(family)
    if f == "gaussian":
        return _out(6.0 / np.pi * np.arcsin(np.asarray(theta, float) / 2.0))
    theta = float(theta)
    if f == "frank":
        return _frank_rho_s(theta)
    if f == "t":
        nu = _need_nu(nu)
        rho = theta
        if abs(rho) >= 1:
            return float(np.sign(rho))

        def integrand(y, x):
            q = (x * x - 2 * rho * x * y + y * y) / (nu * (1 - rho * rho))
            logf = (gammaln((nu + 2) / 2) - gammaln(nu / 2) - math.log(nu * math.pi)
                    - 0.5 * math.log(1 - rho * rho) - (nu + 2) / 2 * math.log1p(q))
            return student_t.cdf(x, nu) * student_t.cdf(y, nu) * math.exp(logf)
        val, _ = integrate.dblquad(integrand, -np.inf, np.inf, -np.inf, np.inf,
                                   epsabs=1e-10, epsrel=1e-9)
        return 12.0 * val - 3.0
    # archimedean: 12 * int int C - 3 via Gauss-Legendre on [0,1]^2
    x, w = np.polynomial.legendre.leggauss(200)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    U, V = np.meshgrid(x, x, indexing="ij")
    C = copula_cdf(f, np.column_stack([U.ravel(), V.ravel()]), theta).reshape(U.shape)
    return float(12.0 * (w @ C @ w) - 3.0)


def copula_param_from_spearman(family, rho_s, nu=None) -> float:
    """Inverse of spearman_rho. gaussian closed form rho = 2 sin(pi rho_S/6); others by brentq."""
    f = _fam(family)
    if f == "gaussian":
        return _out(2.0 * np.sin(np.pi * np.asarray(rho_s, float) / 6.0))
    rho_s = float(rho_s)
    brackets = {"t": (-0.9999, 0.9999), "clayton": (-0.999, 200.0), "gumbel": (1.0, 200.0),
                "frank": (-200.0, 200.0)}
    lo, hi = brackets[f]
    return optimize.brentq(lambda th: spearman_rho(f, th, nu) - rho_s, lo, hi, xtol=1e-10)


# ---------------------------------------------------------------------------------------------
# Tail dependence
# ---------------------------------------------------------------------------------------------
def tail_dependence(family, theta, nu=None):
    """Coefficients of tail dependence -> tuple (lambda_lower, lambda_upper).

    gumbel (0, 2 - 2^(1/theta)); clayton (2^(-1/theta), 0) for theta > 0 else (0, 0);
    t: lambda_L = lambda_U = 2 * t_{nu+1}.cdf(-sqrt((nu+1)(1-rho)/(1+rho)));
    gaussian and frank: (0, 0) (gaussian (1,1) only at rho = 1).
    """
    f = _fam(family)
    theta = float(theta)
    if f == "gaussian":
        return (1.0, 1.0) if theta >= 1 else (0.0, 0.0)
    if f == "t":
        nu = _need_nu(nu)
        if theta >= 1:
            return 1.0, 1.0
        if theta <= -1:
            return 0.0, 0.0
        lam = 2.0 * student_t.cdf(-math.sqrt((nu + 1) * (1 - theta) / (1 + theta)), nu + 1)
        return float(lam), float(lam)
    if f == "clayton":
        return (2.0 ** (-1.0 / theta), 0.0) if theta > 0 else (0.0, 0.0)
    if f == "gumbel":
        return 0.0, 2.0 - 2.0 ** (1.0 / theta)
    return 0.0, 0.0


# ---------------------------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------------------------
def _corr_matrix(theta, d):
    R = np.asarray(theta, float)
    if R.ndim == 0:
        r = float(R)
        if d > 1 and not (-1.0 / (d - 1) < r < 1.0):
            raise ValueError(f"equicorrelation {r} not positive definite for d={d}")
        R = np.full((d, d), r)
        np.fill_diagonal(R, 1.0)
    if R.shape != (d, d):
        raise ValueError(f"correlation matrix shape {R.shape} != ({d},{d})")
    return R


def _positive_stable(alpha, size, rng):
    """Positive alpha-stable S (0 < alpha <= 1) with Laplace transform E[exp(-s S)] = exp(-s^alpha).

    Chambers-Mallows-Stuck / Kanter representation: Theta ~ U(0, pi), W ~ Exp(1),
    S = sin(alpha Theta) / sin(Theta)^(1/alpha) * (sin((1-alpha) Theta) / W)^((1-alpha)/alpha).
    """
    if alpha >= 1.0:
        return np.ones(size)
    th = rng.uniform(0.0, np.pi, size)
    w = rng.exponential(1.0, size)
    logS = (np.log(np.sin(alpha * th)) - np.log(np.sin(th)) / alpha
            + (1.0 - alpha) / alpha * (np.log(np.sin((1.0 - alpha) * th)) - np.log(w)))
    return np.exp(logS)


def copula_sample(family, theta, n, d=2, nu=None, seed=None, rng=None):
    """Draw n samples of a d-dim copula -> ndarray (n, d) of uniforms in (0,1).

    Algorithms (one generator; pass rng= to continue a shared stream, else seed= builds one):
      gaussian: Z ~ N(0,I), X = Z L^T with L = chol(R), U = Phi(X). theta scalar -> equicorrelation.
      t:        X as gaussian, ONE W ~ chi2(nu) per row shared by all coordinates,
                T = X / sqrt(W/nu), U = t_nu.cdf(T).
      clayton (theta > 0), Marshall-Olkin frailty: V ~ Gamma(1/theta, 1), E_i ~ Exp(1),
                U_i = psi(E_i/V) = (1 + E_i/V)^(-1/theta).
                theta in (-1,0), d=2: conditional inversion U2 = (U1^-theta (W^(-theta/(1+theta)) - 1) + 1)^(-1/theta).
      gumbel (theta >= 1), Marshall-Olkin: V positive stable with index 1/theta (Chambers-Mallows-Stuck,
                Laplace transform exp(-s^(1/theta))), U_i = exp(-(E_i/V)^(1/theta)).
      frank (theta > 0), Marshall-Olkin: V ~ Logarithmic series with p = 1 - e^-theta,
                U_i = -(1/theta) ln(1 - (1 - e^-theta) e^(-E_i/V)).
                theta < 0, d=2: conditional U2 = -(1/theta) ln(1 + W(e^-theta - 1)/(W + (1-W) e^(-theta U1))).
    Order of draws is fixed per family so results are reproducible for a given seed.
    """
    f = _fam(family)
    if rng is None:
        rng = np.random.default_rng(seed)
    n, d = int(n), int(d)
    if f in ("gaussian", "t"):
        R = _corr_matrix(theta, d)
        L = np.linalg.cholesky(R)
        X = rng.standard_normal((n, d)) @ L.T
        if f == "gaussian":
            return norm.cdf(X)
        nu = _need_nu(nu)
        W = rng.chisquare(nu, size=(n, 1))
        return student_t.cdf(X / np.sqrt(W / nu), nu)
    theta = float(theta)
    if f == "clayton":
        if theta > 0:
            V = rng.gamma(1.0 / theta, 1.0, size=(n, 1))
            E = rng.exponential(1.0, size=(n, d))
            return np.exp(-np.log1p(E / V) / theta)
        if theta == 0:
            return rng.uniform(size=(n, d))
        if d != 2 or theta <= -1:
            raise ValueError("clayton theta < 0 needs d=2 and theta > -1")
        u1, w = rng.uniform(size=n), rng.uniform(size=n)
        u2 = (u1 ** (-theta) * (w ** (-theta / (1 + theta)) - 1) + 1) ** (-1.0 / theta)
        return np.column_stack([u1, u2])
    if f == "gumbel":
        if theta < 1:
            raise ValueError("gumbel theta must be >= 1")
        V = _positive_stable(1.0 / theta, (n, 1), rng)
        E = rng.exponential(1.0, size=(n, d))
        return np.exp(-(E / V) ** (1.0 / theta))
    # frank
    if theta > 0:
        p = -math.expm1(-theta)
        V = rng.logseries(p, size=(n, 1)).astype(float)
        E = rng.exponential(1.0, size=(n, d))
        return -np.log1p(-p * np.exp(-E / V)) / theta
    if theta == 0:
        return rng.uniform(size=(n, d))
    if d != 2:
        raise ValueError("frank theta < 0 is only a valid copula for d=2")
    u1, w = rng.uniform(size=n), rng.uniform(size=n)
    u2 = -np.log1p(w * math.expm1(-theta) / (w + (1 - w) * np.exp(-theta * u1))) / theta
    return np.column_stack([u1, u2])


# ---------------------------------------------------------------------------------------------
# CDF, density, log-likelihood (bivariate)
# ---------------------------------------------------------------------------------------------
def _uv(u):
    u = np.asarray(u, float)
    if u.ndim == 1 and u.shape[0] == 2:
        u = u[None, :]
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError(f"u must be (n,2) pseudo-observations, got shape {u.shape}")
    u = np.clip(u, _EPS, 1.0 - _EPS)
    return u[:, 0], u[:, 1]


def copula_cdf(family, u, theta, nu=None):
    """Bivariate copula CDF C(u,v) for u of shape (n,2) -> ndarray (n,).

    clayton, gumbel, frank closed form; gaussian via scipy multivariate_normal.cdf (~1e-8);
    t via scipy multivariate_t.cdf, which is Monte-Carlo based (slow, ~1e-4 noise: do not
    finite-difference it).
    """
    f = _fam(family)
    a, b = _uv(u)
    if f == "gaussian" or f == "t":
        from scipy.stats import multivariate_normal, multivariate_t
        rho = float(theta)
        cov = np.array([[1.0, rho], [rho, 1.0]])
        if f == "gaussian":
            x = np.column_stack([norm.ppf(a), norm.ppf(b)])
            return np.atleast_1d(multivariate_normal(mean=[0, 0], cov=cov).cdf(x))
        nu = _need_nu(nu)
        x = np.column_stack([student_t.ppf(a, nu), student_t.ppf(b, nu)])
        mt = multivariate_t(loc=[0, 0], shape=cov, df=nu)
        return np.array([mt.cdf(xi) for xi in x])
    theta = float(theta)
    if f == "clayton":
        if theta == 0:
            return a * b
        s = a ** (-theta) + b ** (-theta) - 1.0
        return np.where(s > 0, np.maximum(s, 0.0) ** (-1.0 / theta), 0.0) if theta < 0 else s ** (-1.0 / theta)
    if f == "gumbel":
        S = (-np.log(a)) ** theta + (-np.log(b)) ** theta
        return np.exp(-S ** (1.0 / theta))
    if abs(theta) < 1e-12:
        return a * b
    return -np.log1p(np.expm1(-theta * a) * np.expm1(-theta * b) / math.expm1(-theta)) / theta


def copula_logpdf(family, u, theta, nu=None):
    """Bivariate copula log-density ln c(u,v) for u of shape (n,2) -> ndarray (n,).

    u is clipped to [1e-12, 1-1e-12]. Stable forms (x = Phi^-1(u), a = -ln u, ...):
      gaussian  -0.5 ln(1-rho^2) - (rho^2(x^2+y^2) - 2 rho x y)/(2(1-rho^2))
      t         ln f2(x,y) - ln f1(x) - ln f1(y) with x = t_nu^-1(u)
      clayton   ln(1+theta) - (1+theta)(ln u + ln v) - (2+1/theta) ln(u^-theta + v^-theta - 1) (log-sum-exp)
      gumbel    -S^(1/theta) + a + b + (theta-1) ln(ab) + (1/theta-2) ln S + ln(S^(1/theta)+theta-1)
      frank     ln(theta(1-e^-theta)) - theta(u+v) - 2 ln|(1-e^-theta) - (1-e^-theta u)(1-e^-theta v)|
    Outside the support (clayton theta < 0) the value is -inf.
    """
    f = _fam(family)
    a, b = _uv(u)
    if f == "gaussian":
        rho = float(theta)
        if not -1 < rho < 1:
            raise ValueError("gaussian rho must be in (-1,1)")
        x, y = norm.ppf(a), norm.ppf(b)
        r2 = 1.0 - rho * rho
        return -0.5 * math.log(r2) - (rho * rho * (x * x + y * y) - 2 * rho * x * y) / (2 * r2)
    if f == "t":
        rho, nu = float(theta), _need_nu(nu)
        if not -1 < rho < 1:
            raise ValueError("t rho must be in (-1,1)")
        x, y = student_t.ppf(a, nu), student_t.ppf(b, nu)
        r2 = 1.0 - rho * rho
        lf2 = (gammaln((nu + 2) / 2) - gammaln(nu / 2) - math.log(nu * math.pi) - 0.5 * math.log(r2)
               - (nu + 2) / 2 * np.log1p((x * x - 2 * rho * x * y + y * y) / (nu * r2)))
        c1 = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * math.log(nu * math.pi)
        lf1x = c1 - (nu + 1) / 2 * np.log1p(x * x / nu)
        lf1y = c1 - (nu + 1) / 2 * np.log1p(y * y / nu)
        return lf2 - lf1x - lf1y
    theta = float(theta)
    la, lb = np.log(a), np.log(b)
    if f == "clayton":
        if theta == 0:
            return np.zeros_like(a)
        if theta <= -1:
            raise ValueError("clayton theta must be > -1")
        if theta > 0:
            p, q = -theta * la, -theta * lb            # ln u^-theta, >= 0
            m = np.maximum(p, q)
            lns = m + np.log(np.exp(p - m) + np.exp(q - m) - np.exp(-m))
        else:
            s = a ** (-theta) + b ** (-theta) - 1.0
            with np.errstate(divide="ignore", invalid="ignore"):
                lns = np.where(s > 0, np.log(np.where(s > 0, s, 1.0)), np.nan)
        out = math.log1p(theta) - (1 + theta) * (la + lb) - (2 + 1 / theta) * lns
        return np.where(np.isnan(out), -np.inf, out)
    if f == "gumbel":
        if theta < 1:
            raise ValueError("gumbel theta must be >= 1")
        A, B = -la, -lb
        lA, lB = np.log(A), np.log(B)
        lnS = np.logaddexp(theta * lA, theta * lB)
        S1 = np.exp(lnS / theta)
        return -S1 + A + B + (theta - 1) * (lA + lB) + (1 / theta - 2) * lnS + np.log(S1 + theta - 1)
    # frank
    if abs(theta) < 1e-12:
        return np.zeros_like(a)
    if theta < 0:                                       # c_{-t}(u,v) = c_t(u, 1-v)
        theta, b = -theta, 1.0 - b
    # denominator (1-e^-t) - (1-e^-tu)(1-e^-tv) = e^-t [e^{t(1-u)}(1-e^-tv) + (e^{t(1-v)} - 1)]:
    # both bracket terms are >= 0, so no cancellation near the corners
    ln_den = -theta + np.logaddexp(theta * (1.0 - a) + np.log(-np.expm1(-theta * b)),
                                   _log_expm1(theta * (1.0 - b)))
    return (math.log(theta) + math.log(-math.expm1(-theta)) - theta * (a + b) - 2.0 * ln_den)


def _log_expm1(x):
    x = np.asarray(x, float)
    big = x > 30.0
    return np.where(big, x + np.log1p(-np.exp(-np.where(big, x, 30.0))),
                    np.log(np.expm1(np.where(big, 1.0, x))))


def copula_pdf(family, u, theta, nu=None):
    """exp(copula_logpdf) -> ndarray (n,)."""
    return np.exp(copula_logpdf(family, u, theta, nu))


def copula_loglik(family, u, theta, nu=None) -> float:
    """Sum of copula_logpdf over rows (pseudo-log-likelihood)."""
    return float(np.sum(copula_logpdf(family, u, theta, nu)))


# ---------------------------------------------------------------------------------------------
# Pseudo-observations and fitting
# ---------------------------------------------------------------------------------------------
def pseudo_observations(x):
    """Column-wise ranks / (n+1), average ranks for ties -> ndarray (n,d) strictly in (0,1).

    Accepts ndarray or DataFrame (NaN rows must be dropped jointly beforehand)."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    if np.isnan(x).any():
        raise ValueError("NaN in input: drop rows with any NaN jointly before ranking")
    return rankdata(x, axis=0, method="average") / (x.shape[0] + 1.0)


_BOUNDS = {"gaussian": (-0.9999, 0.9999), "t": (-0.9999, 0.9999), "clayton": (1e-6, 100.0),
           "gumbel": (1.0, 100.0), "frank": (-100.0, 100.0)}


def _result(f, theta, nu, ll, n, k, method, extra=None):
    if not np.isfinite(ll):
        raise ValueError(f"{f} copula log-likelihood is not finite (theta={theta}, nu={nu}); "
                         "check u is strictly in (0,1) (use pseudo_observations) and the family's "
                         "parameter range")
    out = {"family": f, "theta": float(theta), "nu": None if nu is None else float(nu),
           "loglik": float(ll), "aic": float(-2 * ll + 2 * k), "bic": float(-2 * ll + k * math.log(n)),
           "n_params": k, "n": int(n), "method": method,
           "tau": float(copula_tau(f, theta)),
           "tail_dependence": tail_dependence(f, theta, nu)}
    if extra:
        out.update(extra)
    return out


def _fit_nu_given_rho(u, rho, nu_bounds):
    lo, hi = float(nu_bounds[0]), float(nu_bounds[1])
    grid = np.exp(np.linspace(math.log(lo), math.log(hi), 25))
    lls = np.array([copula_loglik("t", u, rho, g) for g in grid])
    i = int(np.nanargmax(lls))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    if a == b:
        return float(grid[i])
    res = optimize.minimize_scalar(lambda ln_nu: -copula_loglik("t", u, rho, math.exp(ln_nu)),
                                   bounds=(math.log(a), math.log(b)), method="bounded",
                                   options={"xatol": 1e-6})
    return float(math.exp(res.x)) if -res.fun >= lls[i] else float(grid[i])


def fit_copula(family, u, method="mle", nu_bounds=(2.01, 100.0), nu=None, theta_bounds=None):
    """Fit a bivariate copula to pseudo-observations u (n,2) -> dict.

    Returns dict{family, theta, nu, loglik, aic, bic, n_params, n, method, tau, tail_dependence,
    at_bound}. AIC = -2LL + 2k, BIC = -2LL + k ln n; k = 2 for t (rho, nu) unless nu is fixed (k=1).
    method="mle": canonical maximum likelihood (bounded; theta bounds default gaussian/t |rho|<=0.9999,
        clayton [1e-6,100], gumbel [1,100], frank [-100,100]); t: rho and nu jointly (start: tau
        inversion + nu profile grid over nu_bounds, then L-BFGS-B in (atanh rho, ln nu)).
    method="itau": theta by Kendall-tau inversion; t: rho by tau, nu by profile likelihood over
        nu_bounds. clayton/gumbel with empirical tau <= 0 -> independence (theta 0 / 1, LL = 0).
    Pass nu= to fix the t degrees of freedom. `at_bound` is True when a parameter sits on a bound
    (nu at the upper bound means the data look Gaussian - not a fitting failure).
    Raises ValueError if the log-likelihood is not finite, so AIC/BIC are always finite.
    """
    f = _fam(family)
    u = np.asarray(u, float)
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError("fit_copula is bivariate: u must be (n,2)")
    if (u <= 0).any() or (u >= 1).any():
        raise ValueError("u must be strictly inside (0,1): use pseudo_observations(x) (rank/(n+1))")
    n = u.shape[0]
    lo, hi = theta_bounds if theta_bounds is not None else _BOUNDS[f]
    tau_hat = empirical_kendall_tau(u)
    method = str(method).lower()

    if method in ("itau", "tau", "kendall", "moment"):
        if f in ("clayton", "gumbel") and tau_hat <= 0:
            th = 0.0 if f == "clayton" else 1.0
            return _result(f, th, None, 0.0, n, 1, "itau", {"at_bound": True})
        th = copula_param_from_tau(f, tau_hat)
        th = float(np.clip(th, lo, hi))
        if f == "t":
            nu_hat = float(nu) if nu is not None else _fit_nu_given_rho(u, th, nu_bounds)
            k = 1 if nu is not None else 2
            at_b = nu is None and (abs(nu_hat - nu_bounds[0]) < 1e-3 or abs(nu_hat - nu_bounds[1]) < 1e-2)
            return _result(f, th, nu_hat, copula_loglik(f, u, th, nu_hat), n, k, "itau",
                           {"at_bound": bool(at_b)})
        return _result(f, th, None, copula_loglik(f, u, th), n, 1, "itau",
                       {"at_bound": bool(th <= lo or th >= hi)})

    if method not in ("mle", "ml", "cml"):
        raise ValueError("method must be 'mle' or 'itau'")

    if f == "t":
        rho0 = float(np.clip(copula_param_from_tau("t", tau_hat), lo * 0.99, hi * 0.99))
        if nu is not None:
            nu = float(nu)
            res = optimize.minimize_scalar(lambda r: -copula_loglik("t", u, r, nu), bounds=(lo, hi),
                                           method="bounded", options={"xatol": 1e-8})
            return _result(f, res.x, nu, -res.fun, n, 1, "mle",
                           {"at_bound": bool(abs(res.x) > 0.9998)})
        nu0 = _fit_nu_given_rho(u, rho0, nu_bounds)
        lnlo, lnhi = math.log(nu_bounds[0]), math.log(nu_bounds[1])

        def negll(p):
            ll = copula_loglik("t", u, math.tanh(p[0]), math.exp(p[1]))
            return -ll if np.isfinite(ll) else 1e300
        zlo, zhi = math.atanh(lo), math.atanh(hi)
        x0 = [math.atanh(rho0), float(np.clip(math.log(nu0), lnlo, lnhi))]
        res = optimize.minimize(negll, x0, method="L-BFGS-B", bounds=[(zlo, zhi), (lnlo, lnhi)])
        best = res.x if res.fun <= negll(x0) else np.array(x0)
        rho_hat, nu_hat = math.tanh(best[0]), math.exp(best[1])
        at_b = (best[1] <= lnlo + 1e-6) or (best[1] >= lnhi - 1e-6) or abs(rho_hat) > 0.9998
        return _result(f, rho_hat, nu_hat, -negll(best), n, 2, "mle", {"at_bound": bool(at_b)})

    def nll(th):
        ll = copula_loglik(f, u, th)
        return -ll if np.isfinite(ll) else 1e300
    # bracket the optimum around the tau-inversion start on a coarse grid, then bounded Brent
    try:
        th0 = float(np.clip(copula_param_from_tau(f, tau_hat), lo, hi))
    except ValueError:
        th0 = lo
    if f == "frank" and abs(th0) < 1e-6:
        th0 = 1e-3
    grid = np.unique(np.clip(np.concatenate([np.linspace(lo, hi, 41), [th0]]), lo, hi))
    vals = np.array([nll(g) for g in grid])
    i = int(np.argmin(vals))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = optimize.minimize_scalar(nll, bounds=(a, b), method="bounded", options={"xatol": 1e-10})
    th_hat, fbest = (res.x, res.fun) if res.fun <= vals[i] else (grid[i], vals[i])
    return _result(f, th_hat, None, -fbest, n, 1, "mle",
                   {"at_bound": bool(th_hat <= lo + 1e-8 or th_hat >= hi - 1e-8)})


def select_copula(u, families=FAMILIES, criterion="aic", method="mle", nu_bounds=(2.01, 100.0)):
    """Fit every family and rank -> list of dicts (fit_copula output) sorted by criterion ascending
    ("aic"|"bic", lower is better; "loglik" sorts descending). A family whose fit fails is appended
    at the end with keys family, error and criterion value inf."""
    crit = str(criterion).lower()
    ok, bad = [], []
    for fam in families:
        try:
            ok.append(fit_copula(fam, u, method=method, nu_bounds=nu_bounds))
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            bad.append({"family": _fam(fam), "error": f"{type(exc).__name__}: {exc}", crit: math.inf})
    if crit == "loglik":
        ok.sort(key=lambda r: -r["loglik"])
    elif crit in ("aic", "bic"):
        ok.sort(key=lambda r: r[crit])
    else:
        raise ValueError("criterion must be 'aic', 'bic' or 'loglik'")
    for rank, r in enumerate(ok, 1):
        r["rank"] = rank
    return ok + bad


# ---------------------------------------------------------------------------------------------
# Empirical dependence measures
# ---------------------------------------------------------------------------------------------
def empirical_kendall_tau(u):
    """Kendall tau-b (scipy; equals tau-a without ties). (n,2) -> float; (n,d>2) -> (d,d) matrix.
    Rank statistic: identical on raw data and on pseudo-observations."""
    u = np.asarray(u, float)
    if u.ndim != 2 or u.shape[1] < 2:
        raise ValueError("u must be (n,d) with d >= 2")
    d = u.shape[1]
    if d == 2:
        return float(kendalltau(u[:, 0], u[:, 1])[0])
    M = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            M[i, j] = M[j, i] = kendalltau(u[:, i], u[:, j])[0]
    return M


def empirical_spearman_rho(u) -> float:
    """Spearman rho = Pearson correlation of ranks, (n,2) -> float."""
    r = rankdata(np.asarray(u, float), axis=0)
    return float(np.corrcoef(r[:, 0], r[:, 1])[0, 1])


def empirical_tail_dependence(u, q=0.95, tail="upper", denominator="first"):
    """Empirical tail-dependence coefficient at threshold q -> float in [0,1]. Strict inequalities.

    tail="upper": joint = #(u > q & v > q);   tail="lower": joint = #(u < 1-q & v < 1-q).
    denominator="first"   (default, theory-card convention: condition on the FIRST variable)
                          lambda = joint / #(u > q)   [lower: #(u < 1-q)]
    denominator="nominal" lambda = (joint/n) / (1-q) = P_n(U>q, V>q)/(1-q)
    On pseudo-observations the two agree up to rounding (#(u>q) ~ n(1-q)).
    Finite q biases gaussian/frank upward (true limit 0); use q -> 1 with care (few points).
    Pass raw data through pseudo_observations first.
    """
    u = np.asarray(u, float)
    a, b = u[:, 0], u[:, 1]
    q = float(q)
    if q > 1:
        q /= 100.0
    if q < 0.5:
        q = 1.0 - q
    if str(tail).lower().startswith("u"):
        m1, joint = a > q, (a > q) & (b > q)
    else:
        m1, joint = a < 1 - q, (a < 1 - q) & (b < 1 - q)
    if denominator == "first":
        den = m1.sum()
        return float(joint.sum() / den) if den > 0 else float("nan")
    if denominator == "nominal":
        return float(joint.mean() / (1.0 - q))
    raise ValueError("denominator must be 'first' or 'nominal'")


def _out(x):
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x
