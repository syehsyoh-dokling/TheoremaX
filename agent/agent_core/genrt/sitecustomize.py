"""Compatibility layer for GENERATED programs (loaded as sitecustomize in their subprocess only).

The sandbox runs numpy 2.x / pandas 3.x / scipy 1.18, and the model keeps writing the APIs those
releases removed. Measured over six local sweeps (25 Sep autopsy): ~30 crashes from removed APIs
(np.trapz, fillna(method=), freq "M", minimize(ftol=)), 51 NameErrors, most of them a missing
`import json` / `csv` / `sys`, and JSON dumps of numpy scalars. Each crash cost a repair request
out of 25, or ended the unit.

Rules this module keeps:
  * it changes nothing a correct program relies on - every shim is either a name that no longer
    exists, or a retry that only runs after the current library has already refused the call;
  * it never raises: any shim that cannot be installed is skipped.
"""
import builtins
import math as _math

# --- names the model forgets to import -------------------------------------------------------
for _name in ("json", "csv", "os", "sys", "math", "re", "time", "warnings", "collections",
              "itertools", "functools", "pathlib"):
    try:
        if not hasattr(builtins, _name):
            setattr(builtins, _name, __import__(_name))
    except Exception:
        pass
try:
    from pathlib import Path as _Path
    if not hasattr(builtins, "Path"):
        builtins.Path = _Path
except Exception:
    pass

# --- numpy 2.x removals ----------------------------------------------------------------------
try:
    import numpy as _np
    for _old, _new in (("trapz", "trapezoid"), ("product", "prod"), ("cumproduct", "cumprod"),
                       ("alltrue", "all"), ("sometrue", "any"), ("float_", "float64"),
                       ("complex_", "complex128"), ("NaN", "nan"), ("Inf", "inf"),
                       ("NAN", "nan"), ("PINF", "inf"), ("NINF", None), ("infty", "inf"),
                       ("row_stack", "vstack"), ("in1d", "isin")):
        if not hasattr(_np, _old):
            try:
                setattr(_np, _old, -_np.inf if _new is None else getattr(_np, _new))
            except Exception:
                pass
    if not hasattr(_np, "math"):
        _np.math = _math
    if not hasattr(builtins, "np"):
        builtins.np = _np
except Exception:
    _np = None

# --- pandas 3.x removals ---------------------------------------------------------------------
_FREQ = {"M": "ME", "BM": "BME", "SM": "SME", "CBM": "CBME", "Q": "QE", "BQ": "BQE", "A": "YE",
         "Y": "YE", "BA": "BYE", "BY": "BYE", "H": "h", "BH": "bh", "T": "min", "S": "s",
         "L": "ms", "U": "us", "N": "ns"}


def _new_freq(freq):
    """'M' -> 'ME', '3M' -> '3ME', 'Q-DEC' -> 'QE-DEC', 'A-JUN' -> 'YE-JUN'; else unchanged."""
    if not isinstance(freq, str):
        return freq
    import re as _re
    m = _re.fullmatch(r"(\d*)([A-Za-z]+)(-[A-Za-z]+)?", freq.strip())
    if not m or m.group(2) not in _FREQ:
        return freq
    return f"{m.group(1)}{_FREQ[m.group(2)]}{m.group(3) or ''}"


def _retry_freq(fn, arg_index):
    """Call fn; if pandas refuses an old frequency alias, retry once with the new one."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, TypeError) as err:
            msg = str(err)
            if "no longer supported" not in msg and "Invalid frequency" not in msg:
                raise
            if "freq" in kwargs:
                kwargs = {**kwargs, "freq": _new_freq(kwargs["freq"])}
            elif "rule" in kwargs:
                kwargs = {**kwargs, "rule": _new_freq(kwargs["rule"])}
            elif len(args) > arg_index:
                args = list(args)
                args[arg_index] = _new_freq(args[arg_index])
            else:
                raise
            return fn(*args, **kwargs)
    wrapper.__wrapped__ = fn
    return wrapper


try:
    import pandas as _pd
    for _cls in (_pd.DataFrame, _pd.Series):
        _orig_fillna = _cls.fillna

        def _fillna(self, value=None, *args, _orig=_orig_fillna, **kwargs):
            method = kwargs.pop("method", None)
            if method is None:
                return _orig(self, value, *args, **kwargs)
            keep = {k: v for k, v in kwargs.items() if k in ("axis", "limit", "inplace")}
            if str(method).lower() in ("ffill", "pad"):
                return self.ffill(**keep)
            return self.bfill(**keep)
        _cls.fillna = _fillna
        _cls.resample = _retry_freq(_cls.resample, 1)
        _cls.asfreq = _retry_freq(_cls.asfreq, 1)
        if not hasattr(_cls, "iteritems"):
            _cls.iteritems = _cls.items
    if not hasattr(_pd.DataFrame, "append"):
        def _append(self, other, ignore_index=False, **_kw):
            other = other if isinstance(other, (_pd.DataFrame, _pd.Series, list)) else [other]
            if isinstance(other, _pd.Series):
                other = other.to_frame().T
            elif isinstance(other, list) and other and isinstance(other[0], dict):
                other = _pd.DataFrame(other)
            elif isinstance(other, dict):
                other = _pd.DataFrame([other])
            return _pd.concat([self, *(other if isinstance(other, list) else [other])],
                              ignore_index=ignore_index)
        _pd.DataFrame.append = _append
    _pd.date_range = _retry_freq(_pd.date_range, 3)
    _pd.bdate_range = _retry_freq(_pd.bdate_range, 3)
    if not hasattr(builtins, "pd"):
        builtins.pd = _pd
except Exception:
    pass

# --- scipy renames and minimize() options ----------------------------------------------------
try:
    import scipy.integrate as _si
    for _old, _new in (("trapz", "trapezoid"), ("simps", "simpson"),
                       ("cumtrapz", "cumulative_trapezoid")):
        if not hasattr(_si, _old) and hasattr(_si, _new):
            setattr(_si, _old, getattr(_si, _new))
except Exception:
    pass
try:
    import scipy.optimize as _so
    _orig_minimize = _so.minimize
    _OPTION_KEYS = ("ftol", "gtol", "xtol", "maxiter", "maxfun", "maxfev", "eps", "disp",
                    "fatol", "xatol", "maxls")

    def _minimize(fun, x0, *args, **kwargs):
        moved = {k: kwargs.pop(k) for k in list(kwargs) if k in _OPTION_KEYS}
        if moved:
            kwargs["options"] = {**moved, **(kwargs.get("options") or {})}
        return _orig_minimize(fun, x0, *args, **kwargs)
    _minimize.__wrapped__ = _orig_minimize
    _so.minimize = _minimize
except Exception:
    pass

# --- json.dump(s) of numpy / pandas values ---------------------------------------------------
try:
    import json as _json
    import datetime as _dt
    _orig_default = _json.JSONEncoder.default

    def _default(self, o):
        if _np is not None:
            if isinstance(o, _np.generic):
                return o.item()
            if isinstance(o, _np.ndarray):
                return o.tolist()
        if isinstance(o, (_dt.datetime, _dt.date)):
            return o.isoformat()
        if hasattr(o, "isoformat"):
            return o.isoformat()
        if hasattr(o, "tolist"):
            return o.tolist()
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        return _orig_default(self, o)
    _json.JSONEncoder.default = _default
except Exception:
    pass

# --- runtime state at the failing line (LDB-style debugging feedback) ---------------------------
# 27 Sep full run: barrier-garch-var crashed 8 times on the same `float(portfolio_var)` - the repair
# prompt showed only "only 0-dimensional arrays can be converted", never WHAT portfolio_var was.
# Debugging LLMs improve when they see runtime values at the failing block (Zhong et al. 2024, LDB;
# Chen et al. 2023, Self-Debugging). On an uncaught exception, after the normal traceback, print the
# type/shape/short value of the variables in the innermost frame of the generated program, the ones
# named on the failing line first. Never raises; the default traceback is printed unchanged.
try:
    import sys as _sys
    import re as _re
    import linecache as _lc

    def _describe(value):
        try:
            name = type(value).__name__
            shape = getattr(value, "shape", None)
            if shape is not None and not callable(shape) and not hasattr(value, "columns"):
                dtype = getattr(value, "dtype", None)
                head = repr(value)[:90].replace("\n", " ")
                return f"{name} shape={tuple(shape)} dtype={dtype} {head}"
            if isinstance(value, dict):
                # the KEYS are what a KeyError needs (29 Sep yield-curve-pca: 6 crashes on '3 Yr')
                return f"dict len={len(value)} keys={list(value)[:40]!r}"[:400]
            if hasattr(value, "columns") and hasattr(value, "index"):
                return (f"{name} shape={tuple(getattr(value, 'shape', ()))} "
                        f"columns={list(value.columns)[:40]!r} index[:3]={list(value.index[:3])!r}")[:400]
            if isinstance(value, (list, tuple, dict, set, str)):
                return f"{name} len={len(value)} {repr(value)[:90]}"
            return f"{name} {repr(value)[:90]}"
        except Exception:
            return "<unprintable>"

    def _state_hook(etype, evalue, tb, _orig=_sys.excepthook):
        _orig(etype, evalue, tb)
        try:
            frame, lineno = None, None
            while tb is not None:
                fn = tb.tb_frame.f_code.co_filename
                if "site-packages" not in fn and "/lib/python" not in fn and "qf_toolkit" not in fn:
                    frame, lineno = tb.tb_frame, tb.tb_lineno
                tb = tb.tb_next
            if frame is None:
                return
            line = _lc.getline(frame.f_code.co_filename, lineno).strip()
            names = [n for n in dict.fromkeys(_re.findall(r"[A-Za-z_][A-Za-z0-9_]*", line))
                     if n in frame.f_locals]
            others = [n for n in frame.f_locals if n not in names and not n.startswith("__")
                      and not callable(frame.f_locals[n]) and type(frame.f_locals[n]).__name__ != "module"]
            out = [f"=== RUNTIME STATE AT FAILURE (line {lineno}: {line[:120]}) ==="]
            for n in names[:12]:
                out.append(f"  {n} = {_describe(frame.f_locals[n])}")
            if frame.f_code.co_name != "<module>":
                for n in others[:8]:
                    out.append(f"  ({n} = {_describe(frame.f_locals[n])})")
            if issubclass(etype, KeyError):
                shown = 0
                for n, v in list(frame.f_locals.items()):
                    if n in names or n.startswith("__"):
                        continue
                    if isinstance(v, dict) or (hasattr(v, "columns") and hasattr(v, "index")):
                        out.append(f"  [{n}] = {_describe(v)}")
                        shown += 1
                        if shown >= 5:
                            break
            out.append("=== END STATE ===")
            _sys.stderr.write("\n".join(out)[:1400] + "\n")
        except Exception:
            pass

    _sys.excepthook = _state_hook
except Exception:
    pass
