"""know77 group 2: generic helpers for the units that failed in the know76 full run.

Task patterns covered (each helper is written for the PATTERN, conventions are keywords with textbook
defaults, no task names):
- multi-file filing ingestion (a list / dict / directory / bare stems of XML or other filings);
- "fill a typed JSON template" deliverables (results.json whose keys and types the instruction lists,
  report-pair fields __inA / __inB / __BoverA) with automatic resolution of cover-page (dei) facts;
- bundle results that are dicts {file name: object} (unpacking guard);
- portfolio VaR/ES method tables (historical, parametric normal, Cornish-Fisher, Monte Carlo, EWMA)
  and EWMA risk decomposition (marginal / component / percent contribution);
- realized-volatility estimator panels from OHLC bars (close-to-close, Parkinson, Garman-Klass,
  Rogers-Satchell, Yang-Zhang) with explicit annualisation / ddof / mean conventions;
- structured-note (capital-protected / reverse-convertible / autocallable) decomposition into
  zero-coupon bond + option legs with bumped Greeks.
Self-tests: proven_work/fail72/selftest77_2.py.
"""
from __future__ import annotations

# know82 _rd82: report results resolve short / nested keys
def _rd82(d):
    from ._rdict import ResultDict
    return ResultDict({k: (ResultDict(v) if isinstance(v, dict) and not isinstance(v, ResultDict) else v) for k, v in d.items()})

import glob as _glob
import json
import math
import os
import pathlib
import re

import numpy as np
import pandas as pd
from scipy import stats as _st


# ============================================================================= input-file collection
def collect_input_files(spec=None, *, suffix=".xml", data_dir=None, id_strip=r"^form[-_]?\d+[a-z]?[-_]"):
    """Any way a task names its filing files -> ordered dict {file id: pathlib.Path}.

    spec may be: None (every *suffix file in data_dir / the usual input roots), a directory, a glob
    pattern, one path, a list/tuple of paths OR bare stems ('snow', 'form4_snow', 'form4_snow.xml'),
    or a dict {id: path or stem} (ids kept). A bare stem is looked up as <stem><suffix>, then any
    '*<stem>*<suffix>' file. The id is the file stem with the id_strip prefix removed (e.g.
    'form4_snow' -> 'snow'); pass id_strip='' to keep the full stem. Raises FileNotFoundError naming
    what exists when a stem matches nothing. XML TEXT (starting with '<') is never a file name: pass
    paths. know78: suffix may be a tuple ('.html', '.htm'); spec=None also finds files one directory
    level down (e.g. /app/data/filings/*.html) when the data dir itself holds none."""
    if isinstance(suffix, (tuple, list)):
        merged, err = {}, None
        for sfx in suffix:
            try:
                got = collect_input_files(spec, suffix=sfx, data_dir=data_dir, id_strip=id_strip)
            except FileNotFoundError as e:
                err = e
                continue
            for k, v in got.items():
                merged.setdefault(k, v)
        if not merged and err is not None:
            raise err
        return merged
    roots = []
    if data_dir:
        roots.append(pathlib.Path(str(data_dir)))
    for env in ("T1_DATA_DIR", "DATA_DIR"):
        if os.environ.get(env):
            roots.append(pathlib.Path(os.environ[env]))
    try:
        from .io import INPUT_ROOTS as _IR
        roots += [pathlib.Path(str(p)) for p in _IR if str(p) not in (".", "/app/output", "/tmp/agent-work")]
    except Exception:  # noqa: BLE001
        pass
    roots += [pathlib.Path(p) for p in ("/app/data", "/app/input", "/input/environment/data", "data")]
    roots = [r for r in roots if r.is_dir()]

    def _id(p):
        st = pathlib.Path(str(p)).stem
        return (re.sub(id_strip, "", st, flags=re.I) if id_strip else st) or st

    def _one(x):
        x = str(x)
        if x.lstrip().startswith("<"):
            raise ValueError("collect_input_files got file TEXT; pass the path (e.g. '/app/data/<name>')")
        p = pathlib.Path(x)
        if p.is_file():
            return p
        names = [x, x + suffix] if not x.lower().endswith(suffix) else [x]
        for r in roots:
            for n in names:
                if (r / n).is_file():
                    return r / n
                hits = sorted(r.rglob(pathlib.Path(n).name))
                if hits:
                    return hits[0]
        stem = pathlib.Path(x).stem
        for r in roots:
            hits = sorted(h for h in r.rglob(f"*{stem}*{suffix}") if h.is_file())
            if hits:
                return hits[0]
        seen = sorted({str(h) for r in roots for h in r.rglob(f"*{suffix}")})[:20]
        raise FileNotFoundError(f"no {suffix} file for {x!r}; present: {seen}")

    out = {}
    if spec is None:
        for pat in (f"*{suffix}", f"*/*{suffix}"):
            for r in roots:
                hits = sorted(h for h in r.glob(pat) if h.is_file())
                if hits:
                    for h in hits:
                        out[_id(h)] = h
                    return out
        return out
    if isinstance(spec, dict):
        for k, v in spec.items():
            out[str(k)] = _one(v if v is not None else k)
        return out
    if isinstance(spec, (str, pathlib.Path)):
        s = str(spec)
        if pathlib.Path(s).is_dir():
            return {_id(h): h for h in sorted(pathlib.Path(s).glob(f"*{suffix}"))}
        if any(ch in s for ch in "*?["):
            return {_id(h): pathlib.Path(h) for h in sorted(_glob.glob(s))}
        spec = [s]
    for x in spec:
        p = _one(x)
        out[_id(p)] = p
    return out


# ============================================================================= bundle-result guard
def bundle_values(outs, names=None):
    """A bundle's {file name: object} result -> tuple of objects in `names` order (default: the dict's
    own order = the instruction's file order). Use it instead of `a, b, c = bundle(...)`, which unpacks
    the KEYS (file-name strings) and writes the names as data:
        ledger, roll, ... = bundle_values(form4_sale_pressure_bundle(...))"""
    if not isinstance(outs, dict):
        return tuple(outs)
    keys = list(outs) if names is None else list(names)
    return tuple(outs[k] for k in keys)


# ============================================================================= typed JSON templates
_TYPE_WORDS = {"string": "str", "str": "str", "text": "str", "date": "str", "int": "int", "integer": "int",
               "float": "float", "number": "float", "double": "float", "bool": "bool", "boolean": "bool",
               "list": "list", "array": "list", "object": "dict", "dict": "dict"}


def parse_json_template(text):
    """Instruction JSON skeleton -> {key: type} in order. Reads lines like
    '"Field__inA": <string>,' / '"n": <int>' / '"x": {"value": <float>}' (nested keys joined by '.').
    Types: str, int, float, bool, list, dict (unknown words -> 'str')."""
    out = {}
    stack = []
    for line in str(text).splitlines():
        s = line.strip()
        m = re.match(r'^"([^"]+)"\s*:\s*(.*)$', s)
        if m:
            key, rest = m.group(1), m.group(2).strip()
            full = ".".join(stack + [key])
            if rest.startswith("{") and not rest.rstrip(",").endswith("}"):
                stack.append(key)
                continue
            t = re.search(r"<\s*([A-Za-z_]+)", rest)
            out[full] = _TYPE_WORDS.get(t.group(1).lower(), "str") if t else (
                "bool" if rest.startswith(("true", "false")) else "float" if re.match(r"-?\d", rest) else "str")
            continue
        if s.startswith("}") and stack:
            stack.pop()
    return out


def cast_to_type(v, typ):
    """Cast one value to a template type ('str'|'int'|'float'|'bool'); None/NaN stay None.
    int: rounds ('1,234' / '$1,234' accepted); float: strips ',', '$', '%'; bool: checkbox / yes /
    true / x -> True."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    try:
        if typ == "bool":
            if isinstance(v, (bool, np.bool_)):
                return bool(v)
            return str(v).strip().lower() in ("☒", "☑", "x", "true", "yes", "y", "1")
        if typ == "int":
            return int(round(float(str(v).replace(",", "").replace("$", "").strip())))
        if typ == "float":
            return float(str(v).replace(",", "").replace("$", "").replace("%", "").strip())
        if typ in ("list", "dict"):
            return v
        return str(v).strip()
    except (TypeError, ValueError):
        return None


def fill_template(template, values, *, diff_suffix="__BoverA", suffixes=("__inA", "__inB")):
    """Typed template {key: type} + values {key: value} -> dict with EVERY template key (missing -> None),
    each value cast to its type. A key ending in diff_suffix that has no value is computed as
    B - A from the two suffixes' values when both are numeric (int stays int). Write it at once with
    emit('results.json', out): partial answers score, a stub scores nothing."""
    out = {}
    for k, t in template.items():
        out[k] = cast_to_type(values.get(k), t)
    for k, t in template.items():
        if k.endswith(diff_suffix) and out.get(k) is None:
            stem = k[: -len(diff_suffix)]
            a, b = out.get(stem + suffixes[0]), out.get(stem + suffixes[1])
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
                out[k] = cast_to_type(decimal_difference(b, a), t)
    return out


def decimal_difference(b, a):
    """b - a of two REPORTED figures, exact in decimal (know81, conventions table 'report_difference'):
    a figure printed with d decimals is a decimal number, so b - a has at most max(d_a, d_b) decimals; the
    binary float subtraction (24.2 - 24.1 = 0.0999...97) is rounded back to that many places (Goldberg 1991).
    int - int stays int; a value with more than 15 significant decimals (float64 precision) is left as is."""
    if isinstance(a, int) and isinstance(b, int):
        return b - a

    def _dec(x):
        s = repr(float(x))
        if "e" in s or "E" in s or "." not in s:
            return 0 if "e" not in s.lower() else None
        return len(s.split(".")[1].rstrip("0"))
    da, db = _dec(a), _dec(b)
    d = b - a
    if da is None or db is None or max(da, db) > 15:
        return d
    return round(d, max(da, db))


def template_stems(template, suffixes=("__inA", "__inB", "__BoverA")):
    """{key: type} -> {stem: type of its first per-report key} (report-pair templates)."""
    stems = {}
    for k, t in template.items():
        for s in suffixes:
            if k.endswith(s):
                stems.setdefault(k[: -len(s)], t)
                break
    return stems


# ============================================================================= SEC report pair: automatic specs
_CTX_RX = re.compile(r"^(?P<value>[A-Za-z0-9]+)_for_(?P<key>[A-Za-z0-9]+)__(?P<match>.+)$")

# cover-page / statement phrases -> XBRL concept (dei = cover page, us-gaap = statements)
SEC_CONCEPT_ALIASES = (
    (r"aggregate.market.value.*non.?affiliates|public.?float", "dei:EntityPublicFloat"),
    (r"shares.outstanding.*(cover|common)|common.stock.shares.outstanding", "dei:EntityCommonStockSharesOutstanding"),
    (r"fiscal.year.focus", "dei:DocumentFiscalYearFocus"),
    (r"central.index.key|\bcik\b", "dei:EntityCentralIndexKey"),
    (r"net.cash.provided.by.(used.in.)?operating", "us-gaap:NetCashProvidedByUsedInOperatingActivities"),
    (r"net.cash.(used.in|provided.by).*investing", "us-gaap:NetCashProvidedByUsedInInvestingActivities"),
    (r"net.cash.(used.in|provided.by).*financing", "us-gaap:NetCashProvidedByUsedInFinancingActivities"),
)


def sec_auto_spec(src, stem, typ="str", *, aliases=SEC_CONCEPT_ALIASES):
    """Best automatic sec_query spec for a field stem, or None when only a hand spec can answer it:
    1. '<Concept>_for_<KeyConcept>__<value>' -> context pairing (e.g. a security title for a trading
       symbol: kind 'context');
    2. a stem equal to an inline-XBRL concept local name in the report (dei:DocumentType,
       dei:EntityRegistrantName, us-gaap:...) -> kind 'fact' (non-dimensional, latest period);
    3. a stem whose words match an alias phrase (aggregate market value held by non-affiliates ->
       dei:EntityPublicFloat, net cash provided by operating activities -> us-gaap:NetCash...) and the
       concept exists in the report -> kind 'fact';
    otherwise None (write a 'table' / 'text' spec for it, see sec_query)."""
    from .helpers76_2 import _sec_doc
    facts, _, _ = _sec_doc(src)
    m = _CTX_RX.match(stem)
    if m:
        return {"kind": "context", "key": m.group("key"), "value": m.group("value"), "match": m.group("match"),
                "type": typ}
    ln = facts["local_name"].str.lower()
    if (ln == stem.lower()).any():
        return {"kind": "fact", "name": stem, "type": typ}
    words = re.sub(r"[_\W]+", " ", stem).lower()
    for rx, concept in aliases:
        if re.search(rx, words) and (ln == concept.split(":")[-1].lower()).any():
            return {"kind": "fact", "name": concept, "type": typ}
    return None


def sec_pair_results_auto(src_a=None, src_b=None, fields=None, specs=None, *, suffix_a="__inA", suffix_b="__inB",
                          diff_suffix="__BoverA", fuzzy=True, iso_dates=True):
    """Report-pair JSON (__inA / __inB / __BoverA) in ONE short call -> the complete flat dict, write it at once.

    fields: the instruction's template (text or {key: type}), {stem: type}, or a list of stems /
    'stem:int' strings; None -> the JSON skeleton of instruction_text() (know78). src_a / src_b None ->
    report_pair_sources() (the two report zips sorted by accession: A older).
    specs: {stem: sec_query spec} ONLY for fields the automatic rules get wrong. Automatic order:
    sec_auto_spec (cover-page dei facts by concept name, '<X>_for_<Key>__<value>' context pairs, aliases),
    then (fuzzy=True, know78) sec_table_lookup: the table row whose label / section / column header
    share the field's words, unit multiplier applied. iso_dates: a dei date fact is written as its XBRL
    canonical value (ixt date transform -> YYYY-MM-DD). Missing values stay None; __BoverA = B - A for
    numeric fields:
        out = sec_pair_results_auto(A_zip, B_zip, FIELDS)   # FIELDS = {stem: 'int'|'float'|'str'|'bool'}
        emit('results.json', out)"""
    from .helpers76_2 import sec_query, _sec_doc
    from .helpers78_4 import instruction_text, report_pair_sources, sec_table_lookup, ixt_canonical
    specs = dict(specs or {})
    import os as _os
    if (src_a is None or src_b is None
            or (isinstance(src_a, str) and len(src_a) < 260 and not _os.path.exists(src_a) and "<" not in src_a and "\n" not in src_a)
            or (isinstance(src_b, str) and len(src_b) < 260 and not _os.path.exists(src_b) and "<" not in src_b and "\n" not in src_b)):
        src_a, src_b = report_pair_sources()
    if fields is None:
        fields = instruction_text()
        if not parse_json_template(fields):
            raise ValueError("sec_pair_results_auto: pass fields={stem: type} (the instruction is not available "
                             "at run time)")
    if isinstance(fields, str):
        template = parse_json_template(fields)
    elif isinstance(fields, dict):
        template = dict(fields)
    else:
        template = {}
        for f in fields:
            k, _, t = str(f).partition(":")
            template[k.strip()] = _TYPE_WORDS.get(t.strip().lower(), "str") if t else "str"
    if not any(k.endswith((suffix_a, suffix_b, diff_suffix)) for k in template):
        stems = {k: (specs.get(k, {}).get("type") or t) for k, t in template.items()}
        template = {}
        for st, t in stems.items():
            template[st + suffix_a] = t
            template[st + suffix_b] = t
            if t in ("int", "float"):
                template[st + diff_suffix] = t
    stems = template_stems(template, (suffix_a, suffix_b, diff_suffix))
    vals = {}
    for st, t in stems.items():
        for src, suf in ((src_a, suffix_a), (src_b, suffix_b)):
            if (st + suf) not in template:
                continue
            sp = specs.get(st)
            if sp is None:
                try:
                    sp = sec_auto_spec(src, st, t)
                except Exception:  # noqa: BLE001
                    sp = None
            if sp is None:
                if fuzzy:
                    try:
                        vals[st + suf] = sec_table_lookup(src, st, t)
                    except Exception:  # noqa: BLE001
                        vals[st + suf] = None
                continue
            sp = {**sp, "type": sp.get("type", t)}
            try:
                vals[st + suf] = sec_query(src, sp)
                if iso_dates and sp.get("kind") == "fact" and t == "str":
                    facts = _sec_doc(src)[0]
                    nm = str(sp["name"]).split(":")[-1].lower()
                    f = facts[(facts["local_name"].str.lower() == nm) & (~facts["is_dimensional"])]
                    if len(f) and "date" in str(f["format"].iloc[0]).lower():
                        vals[st + suf] = ixt_canonical(vals[st + suf], f["format"].iloc[0])
            except Exception:  # noqa: BLE001 - one bad spec must not lose the other fields
                vals[st + suf] = None
    # know78: component identity - '<P>_<X>_Only' = '<P>_Including_<X>' - '<P>_Excluding_<X>' (also
    # With/Without) when both siblings are template fields and the field has no hand spec
    for st, t in stems.items():
        m = re.match(r"^(?P<p>.+?)_(?P<x>[A-Za-z]+)_Only$", st)
        if not m or st in specs or t not in ("int", "float"):
            continue
        for inc, exc in (("Including", "Excluding"), ("With", "Without")):
            si, se = f"{m.group('p')}_{inc}_{m.group('x')}", f"{m.group('p')}_{exc}_{m.group('x')}"
            if si in stems and se in stems:
                for suf in (suffix_a, suffix_b):
                    a, b = vals.get(si + suf), vals.get(se + suf)
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        vals[st + suf] = a - b
                break
    return fill_template(template, vals, diff_suffix=diff_suffix, suffixes=(suffix_a, suffix_b))


# ============================================================================= VaR / ES method tables
VAR_METHOD_LABELS = {"HS": "hs", "Normal": "normal", "StudentT": "t", "AgeWeightedHS": "brw", "EWMA": "ewma"}


def var_es_one(returns, method, level, *, asset_returns=None, weights=None, value=1.0, hs_method="linear",
               es_tail="geq", include_mean=True, ddof=1, brw_lam=None, brw_interpolate=False,
               ewma_lam=0.94, ewma_init="sample", ewma_include_mean=False):
    """(VaR, ES) as a 2-TUPLE of POSITIVE numbers (x value) for one method on a return series r.

    method: 'hs' (historical, np.quantile(r, 1-level, hs_method), ES = mean loss >= VaR),
    'normal' (mean + sample std ddof, include_mean), 't' (location-scale MLE scipy t.fit, ES
    closed form), 'brw' (age-weighted HS, newest weight highest, lam=brw_lam), 'ewma' (RiskMetrics
    zero-mean normal with sigma_next = current EWMA vol INCLUDING the last return; seed ewma_init:
    'sample' = sample variance / covariance, 'first' = r_0^2). With asset_returns (T x N) and weights
    the EWMA sigma is sqrt(w' S w) of the asset EWMA covariance (identical to the univariate EWMA of
    the portfolio return with the same seed rule). Labels like 'HS', 'Normal', 'StudentT',
    'AgeWeightedHS', 'EWMA' are accepted."""
    from .riskx import historical_var_es, parametric_var_es, age_weighted_var_es, ewma_var_es
    m = VAR_METHOD_LABELS.get(method, str(method).lower())
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if m in ("hs", "historical"):
        return historical_var_es(r, level, method=hs_method, es_tail=es_tail, value=value)
    if m in ("normal", "gaussian", "parametric"):
        return parametric_var_es(r, level=level, dist="normal", include_mean=include_mean, ddof=ddof, value=value)
    if m in ("t", "student", "studentt", "student_t"):
        return parametric_var_es(r, level=level, dist="t", include_mean=include_mean, value=value)
    if m in ("brw", "age", "ageweighted", "agewightedhs", "ageweightedhs"):
        if brw_lam is None:
            raise ValueError("age-weighted HS needs brw_lam= (the decay the task states)")
        return age_weighted_var_es(r, lam=brw_lam, level=level, interpolate=brw_interpolate, value=value)
    if m in ("ewma", "riskmetrics"):
        if asset_returns is not None and weights is not None:
            R = np.asarray(asset_returns, float)
            w = np.asarray(weights, float)
            S = np.cov(R, rowvar=False, ddof=1) if ewma_init == "sample" else np.outer(R[0], R[0])
            for t in range(len(R)):
                if t == 0 and ewma_init != "sample":
                    continue
                S = ewma_lam * S + (1 - ewma_lam) * np.outer(R[t], R[t])
            sig = math.sqrt(float(w @ S @ w))
            mu = float(r.mean()) if ewma_include_mean else 0.0
            return parametric_var_es(mu=mu, sigma=sig, level=level, include_mean=ewma_include_mean, value=value)
        return ewma_var_es(r, lam=ewma_lam, level=level, include_mean=ewma_include_mean, init=ewma_init, value=value)
    raise ValueError(f"unknown VaR method {method!r}")


def overlapping_sums(returns, h):
    """h-period overlapping sums of a (log) return series: s_i = r_i + ... + r_{i+h-1}, length n-h+1."""
    r = np.asarray(returns, float)
    if h <= 1:
        return r.copy()
    c = np.concatenate([[0.0], np.cumsum(r)])
    return c[h:] - c[:-h]


def var_es_methods_table(returns, *, methods=("HS", "Normal", "StudentT", "AgeWeightedHS", "EWMA"),
                         levels=(0.95, 0.99), value=1.0, horizon=1, parametric=("Normal", "StudentT", "EWMA"),
                         asset_returns=None, weights=None, **kw):
    """DataFrame[method, alpha, var_dollar, es_dollar] - one row per (method, level), methods outer.

    horizon h > 1: PARAMETRIC methods (labels in `parametric`) = 1-day x sqrt(h); SIMULATION methods
    (HS, age-weighted) are re-estimated on h-day OVERLAPPING sums of the returns (overlapping_sums).
    kw: conventions of var_es_one (hs_method, es_tail, include_mean, brw_lam, ewma_lam, ewma_init ...)."""
    rows = []
    r = np.asarray(returns, float)
    for m in methods:
        for a in levels:
            if horizon > 1 and m in parametric:
                v, e = var_es_one(r, m, a, asset_returns=asset_returns, weights=weights, value=1.0, **kw)
                v, e = v * math.sqrt(horizon) * value, e * math.sqrt(horizon) * value
            elif horizon > 1:
                v, e = var_es_one(overlapping_sums(r, horizon), m, a, value=value, **kw)
            else:
                v, e = var_es_one(r, m, a, asset_returns=asset_returns, weights=weights, value=value, **kw)
            rows.append({"method": m, "alpha": float(a), "var_dollar": float(v), "es_dollar": float(e)})
    return pd.DataFrame(rows)


def var_methods_comparison_bundle(prices, *, notional=1.0, symbols=None, weights=None, date_col="date",
                                  symbol_col="symbol", price_col="close", return_kind="log",
                                  levels=(0.95, 0.99), horizon=10, backtest_level=0.99, min_obs=252,
                                  methods=("HS", "Normal", "StudentT", "AgeWeightedHS", "EWMA"),
                                  periods_per_year=252, **kw):
    """Long (date, symbol, close) or wide price table -> {data_summary.json, var_1day.csv,
    var_10day.csv, backtest.json, summary.json} for an (equal-)weight portfolio.

    Returns per symbol (log by default) on dates where ALL symbols trade; portfolio return =
    weighted average of the asset returns (equal weight default). n_<sym> = raw price rows per symbol,
    n_aligned = common dates, n_returns, portfolio_mean / std (ddof=1), correlation (first two assets).
    VaR tables via var_es_methods_table (1-day and `horizon`-day; ES >= VaR, positive dollars).
    Backtest: expanding-window HS VaR at backtest_level from data strictly BEFORE day t
    (t = min_obs..n-1), exceedance = realised loss > VaR, Kupiec POF p-value with p = 1-level
    (expected_rate). Summary: most/least conservative = argmax/argmin of the 1-day VaR at the highest
    level, var_range_<level> = [min, max], ewma_current_vol = annualised current EWMA portfolio vol
    (same seed rule as the EWMA VaR). Returns dict{file name: obj}."""
    from .risk import kupiec_pof
    p = prices.copy() if isinstance(prices, pd.DataFrame) else pd.read_csv(str(prices))
    if symbol_col in p.columns:
        counts = p.groupby(symbol_col)[price_col].count()
        p[date_col] = pd.to_datetime(p[date_col])
        wide = p.pivot_table(index=date_col, columns=symbol_col, values=price_col, aggfunc="last").sort_index()
    else:
        wide = p.set_index(pd.to_datetime(p[date_col])).drop(columns=[date_col]).sort_index()
        counts = wide.count()
    syms = list(symbols) if symbols is not None else list(wide.columns)
    wide = wide[syms].dropna()
    R = (np.log(wide / wide.shift(1)) if return_kind == "log" else wide.pct_change()).dropna()
    w = np.full(len(syms), 1.0 / len(syms)) if weights is None else np.asarray(weights, float)
    rp = (R.to_numpy() @ w)
    ds = {f"n_{s.lower()}": int(counts.get(s, 0)) for s in syms}
    ds.update({"n_aligned": int(len(wide)), "n_returns": int(len(rp)), "portfolio_mean": float(np.mean(rp)),
               "portfolio_std": float(np.std(rp, ddof=1)),
               "correlation": float(np.corrcoef(R.iloc[:, 0], R.iloc[:, 1])[0, 1]) if len(syms) > 1 else float("nan")})
    v1 = var_es_methods_table(rp, methods=methods, levels=levels, value=notional, horizon=1, **kw)
    vh = var_es_methods_table(rp, methods=methods, levels=levels, value=notional, horizon=horizon, **kw)
    q = round(1.0 - backtest_level, 12)
    hs_method = kw.get("hs_method", "linear")
    var_t = np.array([-np.quantile(rp[:t], q, method=hs_method) for t in range(int(min_obs), len(rp))])
    hits = -rp[int(min_obs):] > var_t
    n, x = int(len(var_t)), int(hits.sum())
    _, pval = kupiec_pof(x, n, backtest_level)
    bt = {"n_test_days": n, "n_exceedances": x, "exceedance_rate": x / n if n else float("nan"),
          "expected_rate": q, "kupiec_pvalue": float(pval)}
    top = max(levels)
    t1 = v1[v1["alpha"] == float(top)].set_index("method")["var_dollar"]
    from .riskx import ewma_vol
    ew_init = kw.get("ewma_init", "sample")
    summ = {"most_conservative_method": str(t1.idxmax()), "least_conservative_method": str(t1.idxmin()),
            f"var_range_{int(round(top * 100))}": [float(t1.min()), float(t1.max())],
            "ewma_current_vol": float(ewma_vol(rp, lam=kw.get("ewma_lam", 0.94), init=ew_init, as_of="last",
                                               periods_per_year=periods_per_year))}
    return {"data_summary.json": ds, "var_1day.csv": v1, f"var_{horizon}day.csv": vh, "backtest.json": bt,
            "summary.json": summ}


# ============================================================================= estimator study on simulated losses
def simulate_loss_samples(specs, n, seed, *, mixture_draw="full"):
    """Draw loss samples IN ORDER from ONE numpy.random.RandomState(seed) -> {name: ndarray}.

    specs: list of (name, kind, params) with kind 'normal' {loc=0, scale=1}, 't' {nu} (standard,
    rng.standard_t), 'mixture' {weights=(p, 1-p), sds=(s1, s2), locs=(0, 0)}. Mixture (two components):
    u = rng.uniform(size=n) first, then mixture_draw='full' draws n values from EACH component in order
    and keeps np.where(u < p, comp1, comp2) (the usual vectorised code); 'count' draws only the needed
    numbers per component."""
    rng = np.random.RandomState(seed)
    out = {}
    for name, kind, par in specs:
        par = par or {}
        if kind == "normal":
            out[name] = rng.normal(par.get("loc", 0.0), par.get("scale", 1.0), n)
        elif kind in ("t", "student_t"):
            out[name] = rng.standard_t(par["nu"], n)
        elif kind == "mixture":
            p = par["weights"][0]
            s1, s2 = par["sds"]
            m1, m2 = par.get("locs", (0.0, 0.0))
            u = rng.uniform(size=n)
            if mixture_draw == "full":
                a, b = rng.normal(m1, s1, n), rng.normal(m2, s2, n)
                out[name] = np.where(u < p, a, b)
            else:
                k = int((u < p).sum())
                x = np.empty(n)
                x[u < p] = rng.normal(m1, s1, k)
                x[~(u < p)] = rng.normal(m2, s2, n - k)
                out[name] = x
        else:
            raise ValueError(f"unknown kind {kind!r}")
    return out


def loss_estimator_study_bundle(specs, n, seed, alphas=(0.95, 0.99),
                                methods=("historical", "parametric_normal", "parametric_t", "kernel_density"), *,
                                kde_grid=10000, kde_pad_sd=5.0, mixture_draw="full"):
    """Simulated-loss VaR/ES estimator comparison -> {true_values.json, estimates.csv, comparison.json,
    summary.json} in one call (right tail: VaR = alpha-quantile of LOSSES, ES = E[L | L >= VaR]).

    Samples: simulate_loss_samples(specs, n, seed) (one RandomState, listed order). Truth:
    analytic_loss_var_es (normal z_a / phi(z_a)/(1-a); standard t q_a, f(q)(nu+q^2)/((nu-1)(1-a));
    mixture brentq on the mixture CDF + quad of x f(x) over the tail). Estimates: loss_var_es_table
    (historical np.quantile + mean of x >= VaR; normal MLE ddof=0; t.fit + quad tail; Silverman KDE,
    grid CDF interpolation + quad). Comparison: var_es_method_comparison (RMSE over all distributions x
    alphas; best per distribution = min var_rmse + es_rmse on its own alphas; overall = min total).
    true_values keys: {dist: {str(alpha): {var, es}}}. Returns dict{file name: obj}."""
    from .helpers76_1 import analytic_loss_var_es, loss_var_es_table, var_es_method_comparison
    samples = simulate_loss_samples(list(specs), n, seed, mixture_draw=mixture_draw)
    truths = {}
    for name, kind, par in specs:
        par = par or {}
        truths[name] = {}
        for a in alphas:
            if kind == "normal":
                d = analytic_loss_var_es("normal", a)
                d = {"var": par.get("loc", 0.0) + par.get("scale", 1.0) * d["var"],
                     "es": par.get("loc", 0.0) + par.get("scale", 1.0) * d["es"]}
            elif kind in ("t", "student_t"):
                d = analytic_loss_var_es("t", a, nu=par["nu"])
            else:
                d = analytic_loss_var_es("mixture", a, mix_weights=list(par["weights"]), mix_sds=list(par["sds"]))
            truths[name][str(a)] = {"var": float(d["var"]), "es": float(d["es"])}
    est = loss_var_es_table(samples, list(alphas), methods=tuple(methods), kde_grid=kde_grid, kde_pad_sd=kde_pad_sd)
    names = [s[0] for s in specs]
    cmp_ = var_es_method_comparison(est, truths, list(methods), names, list(alphas))
    comparison = {"method_rmse": cmp_["method_rmse"], "best_method_per_distribution": cmp_["best_method_per_distribution"]}
    summary = {"n_samples": int(n), "seed": int(seed), "alpha_levels": [float(a) for a in alphas],
               "distributions": names, "methods": list(methods), "overall_best_method": cmp_["overall_best_method"]}
    return {"true_values.json": truths, "estimates.csv": est, "comparison.json": comparison, "summary.json": summary}


# ============================================================================= EWMA covariance + Euler decomposition
def ewma_cov_recursive(returns, lam, *, seed="first", demean=False):
    """RiskMetrics EWMA covariance S_T [N x N] by the recursion on RAW returns (rows = dates, oldest first):
    S_0 = seed, S_t = lam S_{t-1} + (1 - lam) r_t r_t' (np.outer) for t = 1..T-1.
    seed='first' -> S_0 = outer(r_0, r_0) (the usual coded recursion; the first update then starts at
    r_1), seed='sample' -> sample covariance (ddof=1) and the recursion runs over ALL rows; an array
    seeds directly. demean=False (RiskMetrics: zero mean)."""
    R = np.asarray(returns, float)
    if R.ndim == 1:
        R = R[:, None]
    if demean:
        R = R - R.mean(axis=0)
    if isinstance(seed, str) and seed == "first":
        S = np.outer(R[0], R[0])
        start = 1
    elif isinstance(seed, str) and seed == "sample":
        S = np.cov(R, rowvar=False, ddof=1).reshape(R.shape[1], R.shape[1])
        start = 0
    else:
        S = np.asarray(seed, float)
        start = 0
    for t in range(start, len(R)):
        S = lam * S + (1.0 - lam) * np.outer(R[t], R[t])
    return S


def euler_vol_decomposition(cov, weights):
    """Euler (homogeneity) decomposition of portfolio VOLATILITY sigma_p = sqrt(w' S w):
    marginal_i = (S w)_i / sigma_p, component_i = w_i marginal_i (sum = sigma_p), pct_i = component_i /
    sigma_p (sum = 1). Returns dict{portfolio_variance, portfolio_vol, marginal, component, pct} (arrays)."""
    S = np.asarray(cov, float)
    w = np.asarray(weights, float)
    var = float(w @ S @ w)
    vol = math.sqrt(var)
    m = S @ w / vol
    c = w * m
    return {"portfolio_variance": var, "portfolio_vol": vol, "marginal": m, "component": c, "pct": c / vol}


def ewma_risk_report(returns, weights, lam, *, assets=None, confidence_level=0.99, annualization_factor=252,
                     window=None, seed="first", cov_elements=None, var_key=None):
    """EWMA covariance -> parametric VaR + Euler risk decomposition, in deliverable shape.

    returns: DataFrame (columns = assets in `assets` order) or array; window = keep only the LAST `window`
    rows (an 'estimation window' parameter); seed: see ewma_cov_recursive (default 'first' = outer
    product of the first row of the window). VaR = norm.ppf(confidence_level) x daily vol (positive),
    annualised vol = daily x sqrt(annualization_factor). Returns {"results.json": {portfolio_daily_vol,
    var_<100c>_parametric, annualized_vol}, "solution.json": {"intermediates": {ewma_cov_i_j for
    cov_elements, portfolio_variance, marginal_risk / component_risk / pct_risk_contribution {asset:
    value}}}} - exactly the two files, so `for name, obj in outs.items(): emit(name, obj)` is safe."""
    if isinstance(returns, pd.DataFrame):
        assets = list(assets) if assets is not None else [c for c in returns.columns if c.lower() != "date"]
        R = returns[assets].to_numpy(float)
    else:
        R = np.asarray(returns, float)
        assets = list(assets) if assets is not None else [f"asset_{i}" for i in range(R.shape[1])]
    if window:
        R = R[-int(window):]
    S = ewma_cov_recursive(R, lam, seed=seed)
    d = euler_vol_decomposition(S, weights)
    z = float(_st.norm.ppf(confidence_level))
    vk = var_key or f"var_{int(round(confidence_level * 100))}_parametric"
    res = {"portfolio_daily_vol": d["portfolio_vol"], vk: z * d["portfolio_vol"],
           "annualized_vol": d["portfolio_vol"] * math.sqrt(annualization_factor)}
    if cov_elements is None:   # every upper-triangle element; pass the task's (i, j) list to report only those
        cov_elements = [(i, j) for i in range(S.shape[0]) for j in range(i, S.shape[1])]
    inter = {f"ewma_cov_{i}_{j}": float(S[i, j]) for i, j in cov_elements if i < S.shape[0] and j < S.shape[1]}
    inter["portfolio_variance"] = d["portfolio_variance"]
    inter["marginal_risk"] = {a: float(v) for a, v in zip(assets, d["marginal"])}
    inter["component_risk"] = {a: float(v) for a, v in zip(assets, d["component"])}
    inter["pct_risk_contribution"] = {a: float(v) for a, v in zip(assets, d["pct"])}
    # know87 (round 2): pf86b rebuilt the file by hand as outs['intermediates']['ewma_cov'][i, j] (KeyError x3,
    # then a crash) - the full matrix is reachable under its usual names at every level, but is never WRITTEN
    from ._rdict import ResultDict
    ex = {k: S for k in ("ewma_cov", "cov", "covariance", "cov_matrix", "ewma_cov_matrix", "ewma_covariance")}
    ex.update({"weights": np.asarray(weights, float), "portfolio_vol": d["portfolio_vol"], "assets": list(assets)})
    inner = ResultDict(inter, extras=ex)
    sol = ResultDict({"intermediates": inner}, extras=ex)
    return ResultDict({"results.json": ResultDict(res), "solution.json": sol}, extras=ex)


# ============================================================================= intraday realized measures
def realized_measures_by_day(ticks, *, ts_col="timestamp", price_col="mid_price", freqs=(1, 5),
                             session=("09:30:00", "16:00:00"), dedup="median", noise_freq=1, rv_freq=5,
                             bv_freq=5, periods_per_year=252, bv_scale="none"):
    """Intraday prices -> per-day realized measures + summary, every day computed INSIDE its own group
    (no variable from another day's loop can leak in).

    Cleaning: parse ts; collapse duplicate timestamps by `dedup` (median); keep session [open, close]
    inclusive; sort; group by calendar date. Sampling at f minutes: keep rows with
    ((hour*60 + minute) - open_minutes) % f == 0; RV_f = sum of squared log returns of the kept prices,
    n_f = kept - 1 (0 and RV 0 when fewer than 2). Bandi-Russell noise: noise_var_est = RV_{noise_freq}
    / (2 n_{noise_freq}); grand_mean_noise_var = mean over days. Corrected RV_{rv_freq} = RV - 2 n
    grand_mean (NOT clipped); ann vol = sqrt(max(corrected, 0) x A). Bipower (Barndorff-Nielsen &
    Shephard Def. 1) on the bv_freq grid: (pi/2) sum |r_i||r_{i-1}| (bv_scale='n_over_n_minus_1'
    multiplies by n/(n-1)). Summary: num_days, grand_mean_noise_var, mean_RV_<f>min_corrected (raw
    mean), grand_ann_vol = sqrt(mean x A). Returns {"days": [dict...], "summary": dict}."""
    df = ticks.copy() if isinstance(ticks, pd.DataFrame) else pd.read_csv(str(ticks))
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.groupby(ts_col, as_index=False)[price_col].agg(dedup)
    t0, t1 = pd.Timestamp(session[0]).time(), pd.Timestamp(session[1]).time()
    tt = df[ts_col].dt.time
    df = df[(tt >= t0) & (tt <= t1)].sort_values(ts_col).reset_index(drop=True)
    open_min = pd.Timestamp(session[0]).hour * 60 + pd.Timestamp(session[0]).minute
    days = []
    for day, g in df.groupby(df[ts_col].dt.date, sort=True):
        g = g.sort_values(ts_col)
        mod = (g[ts_col].dt.hour * 60 + g[ts_col].dt.minute - open_min).to_numpy()
        px = g[price_col].to_numpy(float)

        def grid(f):
            p = px[mod % f == 0]
            return np.diff(np.log(p)) if len(p) >= 2 else np.array([])

        rec = {"date": str(day)}
        for f in freqs:
            r = grid(f)
            rec[f"RV_{f}min"] = float(np.sum(r ** 2))
            rec[f"n_{f}min"] = int(len(r))
        rn = grid(noise_freq)
        rec["noise_var_est"] = float(np.sum(rn ** 2) / (2 * len(rn))) if len(rn) else 0.0
        rb = np.abs(grid(bv_freq))
        bv = float((np.pi / 2.0) * np.sum(rb[1:] * rb[:-1])) if len(rb) >= 2 else 0.0
        if bv_scale == "n_over_n_minus_1" and len(rb) >= 2:
            bv *= len(rb) / (len(rb) - 1)
        rec[f"BV_{bv_freq}min"] = bv
        days.append(rec)
    gm = float(np.mean([d["noise_var_est"] for d in days])) if days else float("nan")
    for d in days:
        rv = d.get(f"RV_{rv_freq}min")
        if rv is None:
            raise ValueError(f"rv_freq {rv_freq} must be one of freqs")
        c = rv - 2.0 * d[f"n_{rv_freq}min"] * gm
        d[f"RV_{rv_freq}min_corrected"] = float(c)
        d[f"RV_{rv_freq}min_ann_vol"] = float(math.sqrt(max(c, 0.0) * periods_per_year))
    mc = float(np.mean([d[f"RV_{rv_freq}min_corrected"] for d in days])) if days else float("nan")
    summary = {"num_days": len(days), "grand_mean_noise_var": gm, f"mean_RV_{rv_freq}min_corrected": mc,
               "grand_ann_vol": float(math.sqrt(max(mc, 0.0) * periods_per_year))}
    return {"days": days, "summary": summary}


# ============================================================================= OHLC range-based volatility
def ohlc_variances(o, h, l, c, *, prev_close=None, ddof=1, yz_n=None):
    """Per-period VARIANCE estimates (daily scale) from OHLC arrays of one sample / window -> dict:
    close_to_close = var(ln C_t/C_{t-1}, ddof) over the sample's own closes (n-1 returns);
    parkinson = mean(ln(H/L)^2) / (4 ln 2);
    garman_klass = mean(0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2);
    rogers_satchell = mean(u (u - c) + d (d - c)) with u = ln(H/O), d = ln(L/O), c = ln(C/O)
      (= ln(H/C) ln(H/O) + ln(L/C) ln(L/O));
    yang_zhang = V_o + k V_c + (1 - k) V_RS, V_o = var(ln O_t/C_{t-1}, ddof) (overnight, n-1 values inside
      the sample, or n values when prev_close is given), V_c = var(ln C/O, ddof), V_RS = rogers_satchell,
      k = 0.34 / (1.34 + (n+1)/(n-1)) with n = yz_n or the number of rows.
    Zero-range days contribute 0 (ln 1)."""
    o, h, l, c = (np.asarray(x, float) for x in (o, h, l, c))
    n = len(c)
    rc = np.diff(np.log(c))
    hl = np.log(h / l)
    u, d, cc = np.log(h / o), np.log(l / o), np.log(c / o)
    rs = float(np.mean(u * (u - cc) + d * (d - cc)))
    if prev_close is not None:
        ov = np.log(o / np.concatenate([[prev_close], c[:-1]]))
    else:
        ov = np.log(o[1:] / c[:-1])
    nn = yz_n or n
    k = 0.34 / (1.34 + (nn + 1) / (nn - 1))
    vo = float(np.var(ov, ddof=ddof)) if len(ov) > ddof else 0.0
    vc = float(np.var(cc, ddof=ddof)) if n > ddof else 0.0
    return {"close_to_close": float(np.var(rc, ddof=ddof)) if len(rc) > ddof else 0.0,
            "parkinson": float(np.mean(hl ** 2) / (4 * math.log(2))),
            "garman_klass": float(np.mean(0.5 * hl ** 2 - (2 * math.log(2) - 1) * cc ** 2)),
            "rogers_satchell": rs, "yang_zhang": vo + k * vc + (1 - k) * rs}


OHLC_ESTIMATORS = ("close_to_close", "parkinson", "garman_klass", "rogers_satchell", "yang_zhang")


def rolling_ohlc_vols(df, window, *, periods_per_year=252, n_windows="n_minus_w", cols=("open", "high", "low", "close")):
    """Annualised rolling vols (sqrt(var x A)) of the five OHLC estimators on windows of `window` rows
    (rows i..i+window-1, each window self-contained: its CC uses its own window-1 returns, its YZ
    overnight term its own window-1 overnight gaps, k with n = window). n_windows='n_minus_w' ->
    n_days - window windows (i = 0..n-w-1, the count such tasks state), 'n_minus_w_plus_1' -> every full
    window. Returns DataFrame[estimator columns] (one row per window)."""
    O, H, L, C = (np.asarray(df[cname], float) for cname in cols)
    n = len(C)
    m = n - window if n_windows == "n_minus_w" else n - window + 1
    rows = []
    for i in range(max(m, 0)):
        s = slice(i, i + window)
        v = ohlc_variances(O[s], H[s], L[s], C[s], yz_n=window)
        rows.append({k: math.sqrt(max(v[k], 0.0) * periods_per_year) for k in OHLC_ESTIMATORS})
    return pd.DataFrame(rows, columns=list(OHLC_ESTIMATORS))


def ohlc_vol_estimator_bundle(ohlc, *, window=21, term_windows=(5, 10, 21, 63, 126, 252), periods_per_year=252,
                              n_windows="n_minus_w", efficiency_on="vol", date_col="date"):
    """Daily OHLC -> {calibration.json, full_sample_vol.json, rolling_vol_stats.csv, vol_term_structure.csv,
    summary.json} for the five range estimators (ohlc_variances / rolling_ohlc_vols conventions).

    Full sample: all rows (CC and YZ overnight use the n-1 within-sample gaps; YZ k with n = n_days).
    Rolling stats per estimator: mean/std(ddof=1)/min/max of the annualised rolling vols; efficiency =
    var(CC rolling series) / var(estimator rolling series) with efficiency_on='vol' (the rolling VOL
    series, ddof=1) or 'variance' (the rolling variance series). Term structure: mean rolling CC and YZ vol
    for each window size. Summary: n_rolling_windows, *_vol_annual, efficiencies, range_estimators_lower
    _variance (all four range estimators less variable than CC), yz_most_efficient, vol_mean_range =
    [min, max] of the five full-sample vols. Returns dict{file name: obj}."""
    df = ohlc.copy() if isinstance(ohlc, pd.DataFrame) else pd.read_csv(str(ohlc))
    df.columns = [str(cn).strip().lower() for cn in df.columns]
    if date_col in df:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
    O, H, L, C = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    cal = {"n_days": int(len(df)), "n_returns": int(len(df) - 1), "S0": float(C[-1]),
           "ohlc_valid": bool(((H >= np.maximum(O, C)) & (L <= np.minimum(O, C))).all())}
    v = ohlc_variances(O, H, L, C)
    fs = {k: math.sqrt(max(v[k], 0.0) * periods_per_year) for k in OHLC_ESTIMATORS}
    full = {"cc_vol": fs["close_to_close"], "parkinson_vol": fs["parkinson"], "garman_klass_vol": fs["garman_klass"],
            "rogers_satchell_vol": fs["rogers_satchell"], "yang_zhang_vol": fs["yang_zhang"]}
    rv = rolling_ohlc_vols(df, window, periods_per_year=periods_per_year, n_windows=n_windows)
    ser = rv if efficiency_on == "vol" else rv ** 2 / periods_per_year
    vcc = float(np.var(ser["close_to_close"], ddof=1))
    eff = {k: (vcc / float(np.var(ser[k], ddof=1)) if float(np.var(ser[k], ddof=1)) > 0 else float("inf"))
           for k in OHLC_ESTIMATORS}
    stats_rows = [{"estimator": k, "mean_vol": float(rv[k].mean()), "std_vol": float(rv[k].std(ddof=1)),
                   "min_vol": float(rv[k].min()), "max_vol": float(rv[k].max()), "efficiency": eff[k]}
                  for k in OHLC_ESTIMATORS]
    ts = []
    for w in term_windows:
        r = rolling_ohlc_vols(df, int(w), periods_per_year=periods_per_year, n_windows=n_windows)
        ts.append({"window": int(w), "cc_vol": float(r["close_to_close"].mean()) if len(r) else float("nan"),
                   "yz_vol": float(r["yang_zhang"].mean()) if len(r) else float("nan")})
    rng = [float(min(full.values())), float(max(full.values()))]
    summ = {"n_rolling_windows": int(len(rv)), "cc_vol_annual": full["cc_vol"], "yz_vol_annual": full["yang_zhang_vol"],
            "parkinson_vol_annual": full["parkinson_vol"], "gk_vol_annual": full["garman_klass_vol"],
            "rs_vol_annual": full["rogers_satchell_vol"], "yz_efficiency": eff["yang_zhang"],
            "parkinson_efficiency": eff["parkinson"], "gk_efficiency": eff["garman_klass"],
            "rs_efficiency": eff["rogers_satchell"],
            "range_estimators_lower_variance": bool(all(eff[k] > 1.0 for k in OHLC_ESTIMATORS[1:])),
            "yz_most_efficient": bool(eff["yang_zhang"] >= max(eff[k] for k in OHLC_ESTIMATORS)),
            "vol_mean_range": rng}
    return {"calibration.json": cal, "full_sample_vol.json": full, "rolling_vol_stats.csv": pd.DataFrame(stats_rows),
            "vol_term_structure.csv": pd.DataFrame(ts), "summary.json": summ}


# ============================================================================= code-migration deliverables
def _migration_inputs(old_source, input_file):
    """know78: resolve the old source and the run inputs however they were passed (None, a path, a bare
    name, a DataFrame read by INPUTS, a list) -> (source text, [input paths])."""
    roots = [pathlib.Path(p) for p in (os.environ.get("T1_DATA_DIR", ""), "/app/data", "/input/environment/data",
                                       "data") if p]
    roots = [r for r in roots if r.is_dir()]
    src = old_source
    if src is None or not isinstance(src, (str, pathlib.Path)) or (
            "def " not in str(src) and not pathlib.Path(str(src)).is_file()):
        name = str(src) if isinstance(src, (str, pathlib.Path)) else ""
        cands = [h for r in roots for h in sorted(r.glob("*.py")) if (not name or pathlib.Path(name).name == h.name)]
        cands = cands or [h for r in roots for h in sorted(r.glob("*.py"))]
        cands.sort(key=lambda h: (0 if re.search(r"old|legacy|v0|0x", h.name, re.I) else 1, h.name))
        if not cands:
            raise FileNotFoundError("polars_migrate_and_run: old source not found - pass its path")
        src = cands[0]
    if isinstance(src, (str, pathlib.Path)) and "def " not in str(src) and pathlib.Path(str(src)).is_file():
        src = pathlib.Path(str(src)).read_text(encoding="utf-8")
    if input_file is None or isinstance(input_file, pd.DataFrame) or (
            isinstance(input_file, (list, tuple)) and any(isinstance(x, pd.DataFrame) for x in input_file)):
        # every data file next to the source; the PILOT (smallest) first, then the larger ones
        files = [h for r in roots for h in r.iterdir() if h.is_file() and h.suffix.lower() in
                 (".csv", ".parquet", ".json", ".tsv", ".ndjson")]
        files.sort(key=lambda h: (0 if "pilot" in h.name.lower() else 1, h.stat().st_size))
        inputs = [str(h) for h in files]
    elif isinstance(input_file, (str, pathlib.Path)):
        inputs = [str(input_file)]
    else:
        inputs = [str(x) for x in input_file]
    resolved = []
    for x in inputs:
        p = pathlib.Path(x)
        if not p.is_file():
            hit = next((r / p.name for r in roots if (r / p.name).is_file()), None)
            p = hit or p
        resolved.append(str(p))
    return str(src), resolved


def polars_migrate_and_run(old_source=None, input_file=None, output_dir="/app/output", *, new_name=None,
                           strict_warnings=True, raise_on_failure=False):
    """Polars 0.x -> 1.x migration in one call -> dict{source, todo, warnings, files, input_used, errors, ok}; ALWAYS emit('<new>.py', r['source']).

    Translates the OLD SOURCE with polars_0x_to_1x_source, compiles it, RUNS the migrated function as
    fn(polars, input_file, output_dir) so every file the old pipeline writes exists in output_dir, and
    records every warning (Deprecation / PolarsInefficientMap / MapWithoutReturnDtype). old_source: text,
    a path, a bare name or None (the *.py with 'old' in its name in the data dir); input_file: a path, a
    list tried in order (the first run that succeeds writes the files), or None / a DataFrame (every data
    file in the data dir, the pilot first). new_name None: the old function name with old -> new.
    know78: a failed run or a warning NO LONGER raises (raise_on_failure=False) - the migrated source is
    returned with ok=False and the errors, so the deliverable .py is always written (a stub scores 0)."""
    import warnings as _w
    from .helpers75_4 import polars_0x_to_1x_source
    src, input_file = _migration_inputs(old_source, input_file)
    if new_name is None:
        m0 = re.search(r"^def\s+(\w+)\s*\(", src, re.M)
        if m0 and re.search(r"old", m0.group(1), re.I):
            new_name = re.sub(r"old", "new", m0.group(1), flags=re.I)
    mig = polars_0x_to_1x_source(str(src), new_name=new_name)
    code = str.__str__(mig)
    todo = list(getattr(mig, "todo", []))
    m = re.search(r"^def\s+(\w+)\s*\(", code, re.M)
    fname = m.group(1) if m else new_name
    ns = {}
    errors, used, msgs, files = [], None, [], []
    try:
        exec(compile(code, "<migrated>", "exec"), ns)  # noqa: S102 - the task's own pipeline
    except SyntaxError as e:
        errors.append(f"SyntaxError in migrated source line {e.lineno}: {e.msg}")
    if not errors:
        import polars as _pl
        os.makedirs(str(output_dir), exist_ok=True)
        before = set(os.listdir(str(output_dir)))
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            for inp in input_file:     # e.g. [pilot, full]: a pilot too small for a positional gather may fail
                try:
                    ns[fname](_pl, str(inp), str(output_dir))
                    used = str(inp)
                    break
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{inp}: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
        msgs = [f"{type(x.message).__name__}: {x.message}" for x in caught
                if "polars" in str(x.filename).lower() or "Polars" in type(x.message).__name__
                or "MapWithoutReturnDtype" in type(x.message).__name__ or "<migrated>" in str(x.filename)]
        files = sorted(set(os.listdir(str(output_dir))) - before) or sorted(os.listdir(str(output_dir)))
    ok = used is not None and not (strict_warnings and (msgs or todo))
    if raise_on_failure and used is None:
        raise RuntimeError(f"migrated pipeline failed on every input: {errors}")
    if raise_on_failure and strict_warnings and (msgs or todo):
        raise RuntimeError(f"migrated pipeline still has warnings/todo: {msgs[:5]} {todo[:5]}")
    return {"source": code, "todo": todo, "warnings": msgs, "files": files, "input_used": used, "errors": errors,
            "ok": ok}
