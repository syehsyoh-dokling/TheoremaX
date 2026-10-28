"""know79 group 1: whole-task bundles that stay the single source of every deliverable, and GARCH fits that
are maximum likelihood in native units.

Task patterns covered (each helper is written for the PATTERN; conventions are keywords with textbook defaults;
no task names, no data values):
- "whole-task bundle -> {file name: object}" (dealer FX carry desks, MTM xccy swaps, event backtests ...):
  the per-file cells re-called a 6-17 s bundle once per file and ran out of model requests, or rebuilt one file
  by hand and overwrote the bundle's correct object. Every `*_bundle` loaded before this module is memoised
  (same inputs -> the same result, instantly) and returns a BundleDict (a plain dict that knows it came from a
  bundle), which the kernel's pre-run rewrite uses to keep the bundle's object for a file a later cell rebuilds;
- GARCH(1,1) fitted "in native units": the likelihood is scale-equivariant (fit c r, then omega/c^2, the
  variance forecast/c^2 and alpha, beta unchanged ARE the native-unit MLE), but arch's optimiser stalls at its
  starting values on raw decimal returns (alpha 0.1, beta 0.8, lower log-likelihood). garch11_mle fits on the
  candidate scales and keeps the highest native-unit log-likelihood.
Self-tests: proven_work/fail72/selftest79_1.py.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import math
import pathlib
import sys

import numpy as np
import pandas as pd


# ============================================================================= bundle results that know it
from ._rdict import ResultDict as _ResultDict, deep_rdict as _deep_rdict  # noqa: E402


class BundleDict(_ResultDict):
    """dict {file name: object} returned by a whole-task bundle (json.dump / ** / isinstance(dict) and .get
    unchanged). `_t1_bundle` marks it so a later hand-rebuilt emit of one of its files can be routed back to
    it. A bundle that already returns a ResultDict keeps its class and gets the same marker attribute.
    know89 (flip census, guard A): a ResultDict - a guessed key raises a KeyError that LISTS the real keys
    (and resolves one unambiguous near-miss), a positional read / tuple unpacking reads by the receiving
    names; its nested plain dicts are ResultDicts too. Exact keys behave exactly as before."""
    _t1_bundle = True


class _NoKey(Exception):
    pass


def _fp(x, depth=0):
    """Content fingerprint of a call argument (raises _NoKey when the object cannot be fingerprinted)."""
    if depth > 6:
        raise _NoKey
    if x is None or isinstance(x, (bool, int, float, str, bytes, np.integer, np.floating)):
        return ("v", type(x).__name__, repr(x))
    if isinstance(x, pathlib.PurePath):
        return ("p", str(x))
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            return ("nd", x.shape, tuple(_fp(v, depth + 1) for v in x.ravel().tolist()))
        return ("nd", x.shape, str(x.dtype), hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest())
    if isinstance(x, (pd.DataFrame, pd.Series)):
        try:
            h = pd.util.hash_pandas_object(x, index=True).to_numpy()
        except (TypeError, ValueError):
            raise _NoKey
        cols = tuple(map(str, x.columns)) if isinstance(x, pd.DataFrame) else (str(x.name),)
        dts = tuple(map(str, x.dtypes)) if isinstance(x, pd.DataFrame) else (str(x.dtype),)
        return ("pd", x.shape, cols, dts, hashlib.sha1(h.tobytes()).hexdigest())
    if isinstance(x, dict):
        return ("d", tuple(sorted(((str(k), _fp(v, depth + 1)) for k, v in x.items()), key=lambda kv: kv[0])))
    if isinstance(x, (list, tuple)):
        return ("l", tuple(_fp(v, depth + 1) for v in x))
    raise _NoKey


_MEMO: dict = {}
_MEMO_MAX = 8


def _text_key():
    """know86: the memo key includes the task text - *_report / *_table helpers (memoised since know85) read
    their conventions from instruction_text(), so the same arguments under another text are another call."""
    try:
        from .helpers78_4 import instruction_text
        return hashlib.sha1(str(instruction_text() or "").encode("utf-8", "replace")).hexdigest()
    except Exception:  # noqa: BLE001
        return ""
_CALLS = 0          # bundle calls in this process (the kernel's pre-run rewrite (b) is active only after one)


def memoize_bundle(fn):
    """Wrap a whole-task bundle: identical inputs (by content) return a deep copy of the first result, as a
    BundleDict. Inputs that cannot be fingerprinted are simply not cached. Idempotent."""
    if getattr(fn, "_t1_memo", False):
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _CALLS
        _CALLS += 1
        try:
            key = (fn.__module__, fn.__name__, _fp(args), _fp(kwargs), _text_key())
        except _NoKey:
            key = None
        if key is not None and key in _MEMO:
            return _as_bundle(copy.deepcopy(_MEMO[key]))
        res = fn(*args, **kwargs)
        if isinstance(res, dict):
            res = _as_bundle(res)
            if key is not None:
                if len(_MEMO) >= _MEMO_MAX:
                    _MEMO.pop(next(iter(_MEMO)))
                try:
                    _MEMO[key] = copy.deepcopy(res)
                except Exception:  # noqa: BLE001 - an uncopyable result is returned uncached
                    pass
        return res

    wrapper._t1_memo = True
    return wrapper


def _as_bundle(res):
    if isinstance(res, dict) and getattr(res, "_t1_bundle", False):
        try:                                     # know89: a memo copy (deepcopy skips __init__) is a result too
            from ._rdict import register
            register(res)
        except Exception:  # noqa: BLE001
            pass
        return res
    if not isinstance(res, dict):
        return res
    if type(res) is dict:
        seen = {}
        out = BundleDict()
        seen[id(res)] = out
        for k, v in res.items():
            dict.__setitem__(out, k, _deep_rdict(v, seen, 1))
        return out
    try:
        seen = {}
        for k, v in list(dict.items(res)):        # know89: nested plain dicts of a ResultDict result as well
            if type(v) is dict:
                dict.__setitem__(res, k, _deep_rdict(v, seen, 1))
    except Exception:  # noqa: BLE001
        pass
    try:
        object.__setattr__(res, "_t1_bundle", True)
    except (AttributeError, TypeError):
        pass
    return res


def _install_bundle_memo():
    """Memoise every function named *_bundle defined in the helper modules already imported."""
    pkg = __name__.rsplit(".", 1)[0]
    for name, mod in list(sys.modules.items()):
        if not name.startswith(pkg + ".helpers") or mod is None or name == __name__:
            continue
        for attr in dir(mod):
            obj = getattr(mod, attr, None)
            if (attr.endswith(("_bundle", "_suite", "_deliverables", "_schedule", "_report", "_table"))
                    and callable(obj) and getattr(obj, "__module__", "") == mod.__name__
                    and not getattr(obj, "_t1_memo", False)):
                setattr(mod, attr, memoize_bundle(obj))


_install_bundle_memo()


def bundle_lookup(name, default=None, namespace=None):
    """The object a whole-task bundle already produced for deliverable `name` (searching the BundleDicts in
    `namespace`, default the caller's globals), else `default`. Used by the pre-run rewrite
    emit('x.csv', hand_built) -> emit('x.csv', bundle_lookup('x.csv', hand_built))."""
    ns = namespace
    if ns is None:
        ns = sys._getframe(1).f_globals
    stem = name.rsplit(".", 1)[0] if isinstance(name, str) else ""
    for v in reversed(list(ns.values())):
        if getattr(v, "_t1_bundle", False) and isinstance(v, dict):
            if name in v:
                return v[name]
            if stem and stem in v and isinstance(v[stem], dict) and isinstance(default, dict):
                return v[stem]
    if hasattr(default, "columns") and hasattr(default, "shape"):
        def_cols = set(default.columns)
        for v in reversed(list(ns.values())):
            if hasattr(v, "columns") and hasattr(v, "shape") and v is not default:
                v_cols = set(v.columns)
                if len(v) == len(default) and def_cols < v_cols:
                    return v
    return default


# ============================================================================= GARCH(1,1) MLE in native units
def garch11_mle(returns, *, mean="zero", scales=(1.0, 100.0), dist="normal", periods_per_year=252):
    """GARCH(1,1) maximum likelihood reported in NATIVE (decimal) units, whatever scale the optimiser needs.

    For each c in `scales` it runs garch11_forecast(r, scale=c) (arch fit of c r, rescale=False, then
    omega = omega_c / c^2, sigma2_next = forecast_c / c^2, alpha and beta unchanged) and computes the
    native-unit log-likelihood LL_1 = LL_c + n ln c (the Jacobian of r -> c r). The fit with the highest LL_1
    is the MLE; ties keep the earlier scale. Returns the garch11_forecast dict of that fit plus
    fit_scale, loglik_native and loglik_by_scale."""
    from .helpers75_1 import garch11_forecast
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    best, lls = None, {}
    for c in scales:
        c = float(c)
        try:
            g = garch11_forecast(r, mean=mean, scale=c, dist=dist, periods_per_year=periods_per_year)
        except Exception:  # noqa: BLE001 - a failed scale is skipped
            continue
        ll = float(g["loglik"]) + len(r) * math.log(c)
        lls[c] = ll
        if best is None or ll > best[0] + 1e-9:
            best = (ll, c, g)
    if best is None:
        raise ValueError("GARCH(1,1) fit failed on every scale")
    out = dict(best[2])
    out.update({"fit_scale": best[1], "loglik_native": best[0], "loglik_by_scale": lls})
    return out
