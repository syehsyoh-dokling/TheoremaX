"""Aggregate (compound) loss S = X_1 + ... + X_N on a lattice {0, h, 2h, ...}.

Workflow: severity pmf f = discretize_severity(dist, h, n) -> aggregate pmf g by
compound_poisson_fft(lam, f) or panjer_poisson(lam, f, n) (identical up to ~1e-12 when the FFT is
padded) -> aggregate_var_es(g, h, level). Index k of every pmf is the loss amount k*h.

Conventions:
  * Severity discretisation (method="rounding", default): mass of [(k-.5)h, (k+.5)h) at kh,
    f_0 = F(h/2), f_k = F((k+.5)h) - F((k-.5)h). "lower" puts [kh,(k+1)h) at kh (discrete <= X),
    "upper" puts ((k-1)h, kh] at kh (discrete >= X). Mass beyond the last cell is DROPPED by default
    (tail="drop", pmf sums to F((n-.5)h) < 1) - choose n*h far into the severity tail, or tail="last".
  * FFT: g = Re ifft(PGF_N(fft(f))), Poisson PGF exp(lam (z - 1)). The DFT is circular: aggregate mass
    beyond the FFT length wraps around onto small losses (aliasing). pad=True zero-pads f to a
    power-of-two length that covers the aggregate tail, so the result equals Panjer to round-off.
  * VaR = smallest lattice point x with P(S <= x) >= level. ES default ("geq") = E[S | S >= VaR];
    "gt" = E[S | S > VaR]; "tail_expectation" = Acerbi-Tasche
    ES = (E[S 1{S > VaR}] + VaR (P(S <= VaR) - level)) / (1 - level) (mean of the upper quantiles;
    coherent; lies between "gt" and "geq" weighting). All >= VaR.
  * Moments: E[S] = lam E[X], Var[S] = lam E[X^2], skew = lam E[X^3] / (lam E[X^2])^1.5.
  * `level` accepts 0.99, 99 or 0.01 (all mean 99%).
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.stats import poisson

__all__ = [
    "discretize_severity", "compound_poisson_fft", "compound_fft", "panjer_poisson", "panjer_ab",
    "panjer_negbin", "panjer_binomial", "aggregate_var_es", "aggregate_moments",
    "compound_poisson_moments", "compound_poisson_mc",
]


def _level(level) -> float:
    c = float(level)
    if c > 1.0:
        c /= 100.0
    if c < 0.5:
        c = 1.0 - c
    if not 0.0 < c < 1.0:
        raise ValueError(f"level must be in (0,1), got {level}")
    return c



def severity_dist(name, **params):
    """A scipy frozen distribution from the usual actuarial parametrisations:
      gamma(shape|alpha|k, scale|theta)  or  gamma(shape, rate)  -> scale = 1/rate  ("beta" is refused: texts use it for BOTH)
      lognormal|lognorm(mu|meanlog, sigma|sdlog)                       -> scipy lognorm(s=sigma, scale=e^mu)
      pareto(alpha|shape, xm|scale|x_m)   classic Pareto on x >= xm     -> scipy pareto(b=alpha, scale=xm)
      lomax|pareto2(alpha|shape, scale|theta)  Pareto II on x >= 0      -> scipy lomax(c=alpha, scale=theta)
      exponential|exp(mean|scale  or rate|lambda)
      weibull(shape|k|c, scale|lambda)
    Unknown names or missing parameters raise with the accepted spellings."""
    from scipy import stats
    key = str(name).strip().lower().replace("-", "").replace("_", "")
    p = {k.lower(): v for k, v in params.items()}

    def pick(*names, required=True):
        for nme in names:
            if nme in p and p[nme] is not None:
                return float(p[nme])
        if required:
            raise ValueError(f"severity_dist({name!r}) needs one of {names}; got {sorted(p)}")
        return None

    if key == "gamma":
        shape = pick("shape", "alpha", "k", "a")
        scale = pick("scale", "theta", required=False)
        if scale is None:
            if "beta" in p:
                raise ValueError("severity_dist('gamma'): 'beta' is ambiguous (rate in some texts, scale in "
                                 "others) - pass scale= (mean = shape*scale) or rate= (mean = shape/rate) as "
                                 "the task defines it")
            scale = 1.0 / pick("rate")
        return stats.gamma(shape, scale=scale)
    if key in ("lognormal", "lognorm", "ln"):
        return stats.lognorm(s=pick("sigma", "sdlog", "s"), scale=float(np.exp(pick("mu", "meanlog", "m"))))
    if key in ("pareto", "pareto1", "paretoi"):
        return stats.pareto(pick("alpha", "shape", "b"), scale=pick("xm", "x_m", "scale", "xmin", "theta"))
    if key in ("lomax", "pareto2", "paretoii"):
        return stats.lomax(pick("alpha", "shape", "c"), scale=pick("scale", "theta", "lambda"))
    if key in ("exponential", "exp", "expon"):
        mean = pick("mean", "scale", required=False)
        return stats.expon(scale=mean if mean is not None else 1.0 / pick("rate", "lambda", "lam"))
    if key == "weibull":
        return stats.weibull_min(pick("shape", "k", "c"), scale=pick("scale", "lambda"))
    raise ValueError(f"severity_dist: unknown family {name!r}; use gamma, lognormal, pareto, lomax, "
                     "exponential, weibull, or pass a scipy frozen distribution")


def _as_dist(dist):
    """Accept a scipy frozen distribution, or a dict {"dist"|"name"|"family": ..., params...}."""
    if hasattr(dist, "cdf") and hasattr(dist, "ppf"):
        return dist
    if isinstance(dist, dict):
        d = dict(dist)
        name = d.pop("dist", None) or d.pop("name", None) or d.pop("family", None) or d.pop("type", None)
        if name:
            return severity_dist(name, **d)
    raise TypeError(f"expected a scipy frozen distribution (e.g. severity_dist('gamma', shape=2, scale=500)); "
                    f"got {type(dist).__name__} {str(dist)[:60]!r}")

def _cdf_fn(dist):
    if hasattr(dist, "cdf"):
        return dist.cdf
    if callable(dist):
        return dist
    raise TypeError("dist must be a scipy frozen distribution (has .cdf) or a CDF callable")


def discretize_severity(dist, h, n, method="rounding", tail="drop"):
    """Severity pmf on the lattice 0, h, ..., (n-1)h -> ndarray (n,).

    dist: scipy frozen distribution (e.g. scipy.stats.lognorm(s, scale=e^mu)) or a CDF callable.
    method="rounding" (mass-rounding / mid-point): f_0 = F(h/2), f_k = F((k+.5)h) - F((k-.5)h)
           "lower": f_k = F((k+1)h) - F(kh), f_0 = F(h)  (floor; discrete variable <= X)
           "upper": f_0 = F(0), f_k = F(kh) - F((k-1)h)  (ceiling; discrete variable >= X)
    tail="drop" (default, textbook) leaves the mass beyond the last cell out (sum < 1);
    tail="last" adds it to cell n-1 so the pmf sums to 1.
    """
    dist = _as_dist(dist)
    F = _cdf_fn(dist)
    h, n = float(h), int(n)
    if h <= 0 or n < 1:
        raise ValueError("need h > 0 and n >= 1")
    k = np.arange(n, dtype=float)
    m = str(method).lower()
    if m in ("rounding", "round", "midpoint", "mid"):
        Fe = np.asarray(F((k + 0.5) * h), float)
        f = np.diff(np.concatenate([[0.0], Fe]))
    elif m in ("lower", "floor", "forward"):
        Fe = np.asarray(F((k + 1.0) * h), float)
        f = np.diff(np.concatenate([[0.0], Fe]))
    elif m in ("upper", "ceil", "ceiling", "backward"):
        Fe = np.asarray(F(k * h), float)
        f = np.diff(np.concatenate([[0.0], Fe]))
    else:
        raise ValueError("method must be 'rounding', 'lower' or 'upper'")
    f = np.clip(f, 0.0, None)
    if tail == "last":
        f[-1] += max(0.0, 1.0 - f.sum())
    elif tail != "drop":
        raise ValueError("tail must be 'drop' or 'last'")
    return f


def _next_pow2(x: int) -> int:
    return 1 << max(0, int(math.ceil(math.log2(max(int(x), 1)))))


def compound_fft(severity_pmf, pgf, n=None, pad=True, fft_len=None, mean_count=None, var_count=None):
    """Generic compound pmf by FFT: g = Re ifft(pgf(fft(f))) -> ndarray.

    pgf: callable on a complex array, the probability generating function of N.
    Length: fft_len if given; else pad=True picks a power of two >= max(n, len(f)) covering
    mean + 40 sd of the aggregate (index units) when mean_count/var_count are given, else 8*max(n,len f).
    Returns the first n entries (n=None -> the whole FFT length, sums to ~1).
    """
    f = np.asarray(severity_pmf, float)
    m = len(f)
    out_n = n
    if fft_len is not None:
        L = int(fft_len)
    elif not pad:
        L = int(n) if n is not None else m
    else:
        base = max(int(n) if n is not None else 0, m)
        if mean_count is not None:
            k = np.arange(m)
            ex, ex2 = float(k @ f), float((k * k) @ f)
            vc = var_count if var_count is not None else mean_count
            mu = mean_count * ex
            var = mean_count * (ex2 - ex * ex) + vc * ex * ex
            need = mu + 40.0 * math.sqrt(max(var, 0.0)) + m + 1
            L = _next_pow2(max(base, need) * 2)
        else:
            L = _next_pow2(8 * base)
    if L < m:
        f = f[:L].copy()   # severity longer than the FFT: truncated (documented aliasing risk)
        warnings.warn("fft length shorter than the severity pmf; severity truncated", stacklevel=2)
    fz = np.zeros(L)
    fz[:len(f)] = f
    phi = np.fft.fft(fz)
    g = np.real(np.fft.ifft(pgf(phi)))
    g = np.where(np.abs(g) < 1e-300, 0.0, g)
    if out_n is None:
        return g
    out_n = int(out_n)
    return g[:out_n] if out_n <= L else np.concatenate([g, np.zeros(out_n - L)])


def compound_poisson_fft(lam, severity_pmf, n=None, pad=True):
    """Compound Poisson aggregate pmf by FFT -> ndarray (index k = loss k*h).

    g = Re ifft(exp(lam (fft(f) - 1))). The DFT is circular, so aggregate mass above the FFT length
    wraps around onto small losses. pad=True (default) zero-pads f to a power of two >= 2*(mean +
    40 sd + len(f)) in index units, making aliasing negligible (FFT == Panjer to ~1e-12).
    pad=False uses length n (or len(f)) exactly - the raw textbook version, aliased if the span is short.
    n: number of returned cells (None -> whole padded length; then sum(g) ~ sum-of-f-adjusted 1).
    Tiny negative round-off values (~1e-17) are left as is; clip if you need a strict pmf.
    """
    lam = float(lam)
    return compound_fft(severity_pmf, lambda z: np.exp(lam * (z - 1.0)), n=n, pad=pad,
                        mean_count=lam, var_count=lam)


def _panjer(a, b, g0, f, n):
    f = np.asarray(f, float)
    n = int(n)
    m = len(f)
    g = np.zeros(n)
    g[0] = g0
    denom = 1.0 - a * f[0]
    j_all = np.arange(1, m, dtype=float)
    for k in range(1, n):
        jmax = min(k, m - 1)
        if jmax < 1:
            continue
        j = j_all[:jmax]
        g[k] = np.dot((a + b * j / k) * f[1:jmax + 1], g[k - j.astype(int)]) / denom
    return g


def panjer_poisson(lam, severity_pmf, n):
    """Panjer recursion for compound Poisson -> ndarray (n,) exact on the lattice.

    g_0 = exp(lam (f_0 - 1)); g_k = (lam/k) sum_{j=1..k} j f_j g_{k-j}. O(n * len(f)).
    g_0 underflows for lam (1 - f_0) > ~745: use the FFT (or scale lam down and convolve) then.
    """
    lam = float(lam)
    f = np.asarray(severity_pmf, float)
    g0 = math.exp(lam * (f[0] - 1.0))
    if g0 == 0.0:
        raise ValueError("Panjer g_0 = exp(lam(f0-1)) underflows; use compound_poisson_fft")
    return _panjer(0.0, lam, g0, f, n)


def panjer_ab(a, b, p0, severity_pmf, n):
    """Panjer recursion for any (a,b,0) claim-count law -> ndarray (n,).

    p_k = (a + b/k) p_{k-1}:  Poisson a=0, b=lam; negative binomial (r, beta) a=beta/(1+beta),
    b=(r-1)a; binomial (m, q) a=-q/(1-q), b=(m+1)q/(1-q).
    g_0 = PGF_N(f_0) is derived from (a, b); p0 = P(N=0) is optional (None) and, if given, is
    checked against the value implied by (a, b) (ValueError on mismatch > 1e-8).
    g_k = sum_{j=1..k} (a + b j/k) f_j g_{k-j} / (1 - a f_0).
    """
    a, b = float(a), float(b)
    f = np.asarray(severity_pmf, float)
    f0 = float(f[0])
    if abs(a) < 1e-15:
        lam = b
        imp_p0, g0 = math.exp(-lam), math.exp(lam * (f0 - 1.0))
    elif 0 < a < 1:
        r = 1.0 + b / a
        imp_p0 = (1.0 - a) ** r
        g0 = ((1.0 - a) / (1.0 - a * f0)) ** r
    elif a < 0:
        mm = -1.0 - b / a        # b/a = -(m+1)
        q = -a / (1.0 - a)
        imp_p0 = (1.0 - q) ** mm
        g0 = (1.0 + q * (f0 - 1.0)) ** mm
    else:
        raise ValueError("a must be < 1 for an (a,b,0) distribution")
    if p0 is not None and abs(float(p0) - imp_p0) > 1e-8 * max(1.0, imp_p0):
        raise ValueError(f"p0={p0} inconsistent with (a,b) -> P(N=0)={imp_p0}")
    if g0 == 0.0:
        raise ValueError("g_0 underflows; use compound_fft with the count PGF")
    return _panjer(a, b, g0, f, n)


def panjer_negbin(r, beta, severity_pmf, n):
    """Compound negative binomial (P(N=k) = C(k+r-1,k) (1/(1+beta))^r (beta/(1+beta))^k, mean r*beta)."""
    a = beta / (1.0 + beta)
    return panjer_ab(a, (r - 1.0) * a, None, severity_pmf, n)


def panjer_binomial(m, q, severity_pmf, n):
    """Compound binomial(m, q)."""
    a = -q / (1.0 - q)
    return panjer_ab(a, (m + 1.0) * q / (1.0 - q), None, severity_pmf, n)


def aggregate_moments(pmf, h=1.0):
    """Moments of a lattice pmf -> dict{mean, var, sd, total_mass} (loss = k*h; uses the pmf as given)."""
    p = np.asarray(pmf, float)
    x = np.arange(len(p)) * float(h)
    tot = float(p.sum())
    mean = float(x @ p)
    var = float((x * x) @ p) - mean * mean
    return {"mean": mean, "var": var, "sd": math.sqrt(max(var, 0.0)), "total_mass": tot}


def aggregate_var_es(pmf, h, level, es_mode="geq", normalize=False):
    """VaR and ES of a lattice aggregate-loss pmf -> tuple (VaR, ES) in loss units (k*h).

    VaR = smallest x = k*h with P(S <= x) >= level (1e-12 tolerance on the cumulative sum).
    es_mode="geq" (default): ES = E[S | S >= VaR]
            "gt":            ES = E[S | S > VaR]  (VaR itself if no mass above)
            "tail_expectation" / "acerbi": ES = (E[S 1{S>VaR}] + VaR (F(VaR) - level)) / (1 - level)
                              = (1/(1-level)) int_level^1 VaR_u du  (Acerbi-Tasche, continuous-corrected).
    The pmf is used as given; truncated mass (sum < 1) biases ES down - normalize=True rescales to 1.
    Raises if the total mass is below `level`.
    """
    c = _level(level)
    p = np.clip(np.asarray(pmf, float), 0.0, None)
    if normalize:
        p = p / p.sum()
    x = np.arange(len(p)) * float(h)
    F = np.cumsum(p)
    if F[-1] < c - 1e-12:
        # 30 Sep fft-compound-poisson: "extend the lattice" was repeated 5 times with no idea by how
        # much. Say how long the lattice must be and which call sizes it.
        raise ValueError(f"pmf total mass {F[-1]:.6g} < level {c}: the lattice (n={len(p)}, h={h}) is too "
                         "short for this tail. Size it with (h, n) = suggest_lattice(lam, severity_dist, "
                         "level) and discretize the severity with tail='last' (keeps total mass 1), then "
                         "rerun compound_poisson_fft on that n")
    k = int(np.searchsorted(F, c - 1e-12, side="left"))
    var = float(x[k])
    mode = str(es_mode).lower()
    if mode in ("geq", "ge", ">=", "conditional"):
        tail = p[k:]
        es = float(x[k:] @ tail / tail.sum())
    elif mode in ("gt", ">", "strict"):
        tail = p[k + 1:]
        es = float(x[k + 1:] @ tail / tail.sum()) if tail.sum() > 0 else var
    elif mode in ("tail_expectation", "acerbi", "acerbi_tasche", "te", "cvar"):
        es = float((x[k + 1:] @ p[k + 1:] + var * (F[k] - c)) / (1.0 - c))
    else:
        raise ValueError("es_mode must be 'geq', 'gt' or 'tail_expectation'")
    return var, es


def compound_poisson_moments(lam, sev_mean, sev_second_moment, sev_third_moment=None):
    """Compound Poisson moments -> dict{mean, var, sd, skew (if 3rd moment given, else None)}.

    E[S] = lam E[X], Var[S] = lam E[X^2] (raw second moment, NOT the variance of X),
    skew = lam E[X^3] / (lam E[X^2])^1.5.
    """
    lam = float(lam)
    mean = lam * float(sev_mean)
    var = lam * float(sev_second_moment)
    skew = None
    if sev_third_moment is not None and var > 0:
        skew = lam * float(sev_third_moment) / var ** 1.5
    return {"mean": mean, "var": var, "sd": math.sqrt(var), "skew": skew}


def compound_poisson_mc(lam, sampler, n_sims, seed=None, rng=None):
    """Monte Carlo compound Poisson totals -> ndarray (n_sims,).

    N_i ~ Poisson(lam), then all sum(N) severities drawn in ONE call and summed per scenario.
    sampler: scipy frozen distribution (uses .rvs(size=, random_state=rng)) or a callable
    sampler(rng, size) -> array. One generator: rng= or seed= (np.random.default_rng).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    n_sims = int(n_sims)
    N = rng.poisson(float(lam), size=n_sims)
    tot = int(N.sum())
    if hasattr(sampler, "rvs"):
        X = np.asarray(sampler.rvs(size=tot, random_state=rng), float)
    else:
        if isinstance(sampler, dict):
            X = np.asarray(_as_dist(sampler).rvs(size=tot, random_state=rng), float)
        else:
            try:
                X = np.asarray(sampler(rng, tot), float)
            except TypeError:
                X = np.asarray(sampler(tot), float)       # a sampler written as f(size)
    idx = np.repeat(np.arange(n_sims), N)
    return np.bincount(idx, weights=X, minlength=n_sims)


def suggest_lattice(lam, severity_dist, level=0.999, *, n_points=None, tail_prob=1e-7, max_n=2 ** 22):
    """(h, n) for an FFT/Panjer lattice that holds the aggregate tail: the span n*h must exceed both the
    severity's (1 - tail_prob) quantile and a generous aggregate bound mean + 12 sd (Chebyshev-safe for
    the levels used in practice), and n is a power of two. `severity_dist` is a scipy frozen distribution.
    h defaults to span / n_points (n_points=2**16 unless given)."""
    lam = float(lam)
    severity_dist = _as_dist(severity_dist)
    m1 = float(severity_dist.mean())
    m2 = float(severity_dist.var()) + m1 ** 2
    mean, sd = lam * m1, (lam * m2) ** 0.5
    sev_tail = float(severity_dist.ppf(1.0 - tail_prob))
    span = max(mean + 12.0 * sd, sev_tail * max(1.0, lam))
    n = int(n_points) if n_points else 2 ** 16
    n = 1 << int(np.ceil(np.log2(max(n, 16))))
    h = span / n
    return float(h), int(min(n, max_n))
