"""AUTO-FIX (9 Oct, lapis tengah): the most frequent crashes of the model's cells are patched by
Python and the cell is run again - no model round trip, no new 4,000-token reply to be cut.

Every rule is a (recogniser on the error text, rewrite of the cell source). A rewrite must be
safe when it does not apply (it then leaves the code unchanged, and the original error is
reported as before). Helpers the rewrites rely on are injected into the kernel namespace by
`install(ns)`. The census behind the list: s12_40..45 and k14_45 (House), lab43 (Colab).
"""
from __future__ import annotations

import difflib
import re

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- helpers injected into the cell namespace
def _fmt_date(x, fmt="%Y-%m-%d"):
    """strftime that accepts Timestamp / datetime64 / date / str and returns '' for NaT.
    know91: an Index / Series / array is formatted ELEMENT-WISE (same container type back); before, the whole
    container fell through to str(x) and a date column became the text 'RangeIndex(start=0, stop=217, step=1)'
    on every row. Row numbers (an integer index that is not epoch stamps) raise ValueError - they are not
    dates, the program must index by its date column."""
    if isinstance(x, (pd.Index, pd.Series, np.ndarray, list, tuple)):
        vals = pd.Index(x) if not isinstance(x, pd.Series) else x
        if pd.api.types.is_integer_dtype(vals) or isinstance(vals, pd.RangeIndex):
            arr = np.asarray(vals, dtype=float)
            if not len(arr) or np.nanmax(np.abs(arr)) < 1e8:
                raise ValueError(
                    "date formatting of an integer index: these are row NUMBERS, not dates - format the "
                    "date column itself (df['date'].dt.strftime(...)) or index the frame by it first "
                    "(df.set_index('<date column>'))")
        dt = pd.to_datetime(x, errors="coerce")
        if isinstance(dt, pd.Series):
            return dt.dt.strftime(fmt).fillna("")
        out = pd.Index(dt).strftime(fmt)
        out = pd.Index(["" if v is None or (isinstance(v, float) and np.isnan(v)) else v for v in out])
        return out if isinstance(x, pd.Index) else np.asarray(out, dtype=object)
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        ts = pd.Timestamp(x)
        return "" if pd.isna(ts) else ts.strftime(fmt)
    except Exception:  # noqa: BLE001
        return str(x)


def _scalar(x):
    """float() that accepts a 1-element array / Series / list (and takes the first element of a
    longer one, which is what float(x) of an array of identical values meant)."""
    try:
        return float(x)
    except TypeError:
        arr = np.asarray(x, dtype=float).ravel()
        if arr.size == 0:
            return float("nan")
        return float(arr[0])


def _as_int(x):
    try:
        return int(x)
    except TypeError:
        arr = np.asarray(x).ravel()
        return int(arr[0]) if arr.size else 0


def _to_naive(x, *args, **kwargs):
    """pd.to_datetime that always returns tz-NAIVE stamps: text parsed with utc=True then the zone
    dropped; integer epochs taken as milliseconds (seconds when < 1e11); other arguments passed on."""
    import numbers
    kwargs.pop("utc", None)
    arr = x
    try:
        probe = np.asarray(x).ravel()
        if probe.size and np.issubdtype(probe.dtype, np.number) and not np.issubdtype(probe.dtype, np.bool_):
            unit = "s" if np.nanmax(np.abs(probe.astype(float))) < 1e11 else "ms"
            kwargs.setdefault("unit", unit)
    except Exception:  # noqa: BLE001
        pass
    out = pd.to_datetime(arr, *args, utc=True, **kwargs)
    try:
        if isinstance(out, pd.DatetimeIndex):
            return out.tz_convert("UTC").tz_localize(None)
        if isinstance(out, pd.Series):
            return out.dt.tz_convert("UTC").dt.tz_localize(None)
        if isinstance(out, pd.Timestamp):
            return out.tz_convert("UTC").tz_localize(None)
    except Exception:  # noqa: BLE001
        pass
    return out


def _flat(x):
    """1-D view of a (n,1)/(1,n) array, Series or list of 1-element rows; other inputs unchanged."""
    try:
        arr = np.asarray(x)
        if arr.ndim == 2 and 1 in arr.shape:
            return arr.ravel()
    except Exception:  # noqa: BLE001
        pass
    return x


def _resolve_input(path: str):
    """The real path of an input the program named by a wrong (relative) path, by basename."""
    import json as _json, os as _os
    try:
        files = _json.loads(_os.environ.get("T1_INPUT_FILES") or "[]")
    except ValueError:
        files = []
    base = _os.path.basename(path)
    for f in files:
        if _os.path.basename(f) == base:
            return f
    return None


def install(ns: dict) -> None:
    ns.setdefault("_fmt_date", _fmt_date)
    ns.setdefault("_scalar", _scalar)
    ns.setdefault("_as_int", _as_int)
    ns.setdefault("_to_naive", _to_naive)
    ns.setdefault("_flat", _flat)


# ----------------------------------------------------------------- rewrites
_STRFTIME = re.compile(r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*|\[[^\]]*\]|\([^()]*\))*)\.strftime\(")
_PD_NP = re.compile(r"\bpd\.(column_stack|zeros_like|ones_like|random|linspace|arange|sqrt|exp|log|where|vstack|hstack|concatenate|mean|std|array)\b")


def _columns_in_state(state_text: str) -> list[str]:
    cols = []
    for m in re.finditer(r"columns=\[([^\]]*)\]", state_text or ""):
        cols.extend(c.strip().strip("'\"") for c in m.group(1).split(","))
    return [c for c in cols if c]


def _names_in_state(state_text: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r"^\s*([A-Za-z_]\w*):", state_text or "", re.M)]


# know78: convention overrides the model passes to a whole-task helper that contradict the textbook default the
# helper already uses. Each is dropped BEFORE the cell runs, unless the task's own text asks for that override
# (guard regex on the instruction). (helper-call regex, argument regex, guard regex, note)
_PRERUN = [
    (r"ipca_report\(", r",?\s*\bnormalize\s*=\s*False\b", r"plain als|unnormali[sz]ed|without (?:the )?normali[sz]",
     "ipca_report(normalize=False) dropped: the task text decides (know80 ipca_identification_rule)"),
    (r"sentiment_momentum_report\(", r",?\s*\btokenizer\s*=\s*(?:'split'|\"split\")", r"whitespace|split on spaces",
     "tokenizer='split' dropped: posts carry punctuation, tokens are [a-z]+ runs"),
    (r"localvol_barrier_report\(", r",?\s*\btime_interp\s*=\s*(?:'bucket'|\"bucket\")", r"next expiry|piecewise[- ]constant",
     "time_interp='bucket' dropped: local vol is interpolated between expiry slices"),
    (r"polars_migrate_and_run\(", r"\b(old_source|input_file)\s*=\s*INPUTS\[[^\]]+\]", None,
     "polars_migrate_and_run(...=INPUTS[...]) -> None: the helper finds the paths itself"),
    # know79 group 2 (selftest79_2.py)
    (r"svi_dupire_report\(", r",?\s*\ballow_single_type\s*=\s*True\b",
     r"calls? only|only calls|puts? only|only puts|single option type",
     "svi_dupire_report(allow_single_type=True) dropped: pass the quote table as loaded (calls AND puts)"),
    (r"heston2_surface_(?:deliverables|tables)\(", r",?\s*\bkk_scheme\s*=\s*(?:'gradient2?'|\"gradient2?\")",
     r"(?:np|numpy)\.gradient.{0,80}twice|gradient.{0,40}twice",
     "kk_scheme='gradient2' dropped: d2C/dK2 is the central three-point difference on interior strikes"),
    # know79 group 1 (selftest79_1.py)
    (r"mtm_xccy_swap_bundle\(", r",?\s*\bois_pay_months\s*=\s*\d+",
     r"annual(?:ly)? (?:fixed|payment|coupon)|fixed leg.{0,40}annual|pays? annually|annual fixed leg",
     "ois_pay_months dropped: no fixed-leg frequency stated, each OIS quote is one simple-rate period"),
    # know81 group 1: guard added - a task that says 'native units / do not rescale' keeps the raw fit
    (r"cta_ima_capital_report\(", r",?\s*\bfit_scale\s*=\s*1(?:\.0*)?\b",
     r"native units|do not rescale|disable (?:that|the) rescaling|rescale\s*=\s*false",
     "fit_scale=1 dropped: native-unit results come from the MLE fit (fit_scale='mle'); arch stalls on raw returns"),
    # know81 group 1 (selftest81_1.py): structured_note_report's defaults are the conventions-table readings
    # (garch_pricing_vol: long-run vol when the task reports the long-run variance; garch_mean: zero mean for a
    # daily-return risk model) - a pin of the other reading is dropped unless the task's words ask for it
    (r"structured_note_report\(", r",?\s*\bvol_source\s*=\s*(?:'forecast'|\"forecast\")",
     r"one[- ]step(?:[- ]ahead)? (?:forecast|variance)|next[- ]day (?:forecast|variance|vol)|current conditional vol",
     "structured_note_report(vol_source='forecast') dropped: the task names no forecast vol (garch_pricing_vol)"),
    (r"structured_note_report\(", r",?\s*\bgarch_mean\s*=\s*(?:'constant'|\"constant\")",
     r"constant[- ]mean|\bmu\b|mean equation|drift",
     "structured_note_report(garch_mean='constant') dropped: the task names no mean term (garch_mean)"),
    # know80 group 1: the fx carry bundle's hedge window is the trading days the book is held, its spot hedge the
    # offsetting trade - drop a hand-pinned other reading unless the task states it
    (r"fx_carry_hedge_bundle\(", r",?\s*\bwindow_basis\s*=\s*(?:'tenor'|\"tenor\")",
     r"hedge window (?:is|equals|=) the (?:forward )?tenor|calendar days to delivery",
     "fx_carry_hedge_bundle(window_basis='tenor') dropped: the hedge window is the holding period between rolls"),
    (r"fx_carry_hedge_bundle\(", r",?\s*\bhedge_amount\s*=\s*(?:'delta'|\"delta\")",
     r"hedge amount (?:is|equals|=) (?:\+)?delta|long delta",
     "fx_carry_hedge_bundle(hedge_amount='delta') dropped: the spot hedge offsets the option delta"),
    # know79 group 3 (selftest79_3.py): text-settled defaults (None = read the task text) must not be pinned
    (r"absorption_riskparity_report\(", r",?\s*\bfinal_partial_month\s*=\s*True\b",
     r"partial month.{0,80}rebalanc|final (?:row|day) (?:is|counts as) a rebalance",
     "final_partial_month=True dropped: the task text decides (month_end_sample_rule; know81 default True)"),
    (r"ipca_report\(", r",?\s*\bsweep_sign\s*=\s*True\b", r"(?:each|every) (?:iteration|sweep)[^.]{0,80}sign",
     "ipca_report(sweep_sign=True) dropped: the task text decides (know87 ipca_sweep_sign_rule)"),
    # know87 group C (selftest87_C.py): with the KPS identification in the sweep the sign convention follows every
    # eigen-rotation (ipca package ALS step) - a pinned sweep_sign=False is dropped unless the task forbids it
    (r"ipca_report\(", r",?\s*\bsweep_sign\s*=\s*False\b",
     r"(?:do not|don't|never|must not|should not)\s+(?:flip|fix|apply|enforce|normali[sz]e)[^.]{0,60}sign[^.]{0,60}"
     r"(?:each|every|inside|within)\s+(?:the\s+)?(?:als\s+)?(?:iteration|sweep|step)|sign[^.]{0,40}only once\b",
     "ipca_report(sweep_sign=False) dropped: the sign convention follows each eigen-rotation of the sweep "
     "(ipca_sweep_sign_rule)"),
    (r"cta_regime_report\(", r",?\s*\bfirst_row\s*=\s*(?:'zero'|\"zero\")", r"first (?:row|day) (?:return )?(?:is|=) 0",
     "cta_regime_report(first_row='zero') dropped: it is the know81 default already (NaN lagged weight -> 0)"),
    # know80 group 2 (selftest80_2.py): ipca_report reads identification and cs-R2 centering from the task text
    (r"ipca_report\(", r",?\s*\bnormalize\s*=\s*True\b",
     r"(?:each|every) (?:als )?(?:iteration|sweep)[^.]{0,120}(?:orthonormal|identity|identif|normali[sz])",
     "ipca_report(normalize=True) dropped: the task text decides where the identification is imposed"),
    (r"ipca_report\(", r",?\s*\bcs_r2\s*=\s*(?:'[a-z]+'|\"[a-z]+\")",
     r"(?:un-?)?cent(?:ere|re)d (?:average )?cross[- ]sectional|cross[- ]sectional r\S{0,3}\s*\((?:un-?)?cent",
     "ipca_report(cs_r2=...) dropped: the qualifier attached to the cross-sectional R2 decides"),
    # know80 group 3 (selftest80_3.py): the GARCH mean model of cta_regime_report is read from the task text
    (r"cta_regime_report\(", r",?\s*\bgarch_mean\s*=\s*(?:'[Cc]onstant'|\"[Cc]onstant\")",
     r"constant[- ]mean|mean[- ]model|with (?:a|an) (?:constant |estimated )?mean\b|mean\s*=\s*['\"]?constant",
     "cta_regime_report(garch_mean='Constant') dropped: garch_mean_rule decides (know81: silent task -> Constant)"),
    # know81 group 3 (selftest81_3.py): the three text-settled defaults above were reversed to the textbook reading
    # (conventions table: garch_mean, backtest_first_return, month_end_rebalance); a pin of the withdrawn reading is
    # dropped unless the task's own words ask for it (the entries above stay: dropping them now lands on the same
    # value the default gives)
    (r"absorption_riskparity_report\(", r",?\s*\bfinal_partial_month\s*=\s*False\b",
     r"(?:complete|full) (?:calendar )?months? only|only (?:complete|full) (?:calendar )?months|"
     r"(?:drop|exclude|discard|ignore)s? (?:the |any )?(?:incomplete|partial) (?:final |last )?month",
     "final_partial_month=False dropped: the last observed trading day of each month of the sample is a rebalance"),
    (r"cta_regime_report\(", r",?\s*\bfirst_row\s*=\s*(?:'drop'|\"drop\")",
     r"drop the first (?:return|row|day)|(?:series|backtest) starts? (?:at|on) the second",
     "cta_regime_report(first_row='drop') dropped: the first return day has a zero (NaN -> 0) lagged weight"),
    (r"cta_regime_report\(", r",?\s*\bgarch_mean\s*=\s*(?:'[Zz]ero'|\"[Zz]ero\")",
     r"zero[- ]mean|no mean term|without (?:a )?mean|mean\s*=\s*['\"]?zero|\bvar\b|value[- ]at[- ]risk|"
     r"expected shortfall|risk capital|capital charge",
     "cta_regime_report(garch_mean='Zero') dropped: the task names no mean model - constant mean (garch_mean_rule)"),
]


# know78: a value the model passes that a task of this kind (instruction regex) settles the other way:
# (helper-call regex, argument regex, replacement, instruction regex, note)
_PRERUN_SET = [
    # know81 group 1: the two know78 structured-note forcings (forecast vol, constant mean) are withdrawn - see the
    # _PRERUN drops above; the CTA warm-up rewrite no longer fires when the task's signal has no flat state
    (r"cta_(?:ima_capital_report|ema_vol_target_returns)\(", r"\bwarmup\s*=\s*(['\"])(?:short|mask)\1", "warmup='zero'",
     r"^(?![\s\S]*(?:[-−]1 otherwise|no [\"'“]?no position[\"'”]? state))[\s\S]*(?:ema[- ]crossover|trend[- ]following)",
     "CTA warm-up rows carry weight 0"),
    (r"bond_hedge_immunization_bundle\(", r"\bnode_interp\s*=\s*(['\"])joint\1", "node_interp='linear'",
     r"bootstrap the supplied maturities in order|bootstrap(?:ped)? (?:\w+ ){0,3}in (?:maturity )?order",
     "sequential bootstrap: each node linear back to the previous one"),
    # know79 group 3 K1: a report's max_drawdown is already the positive magnitude the task asks for
    (r"max_drawdown", r"-\s*((?:float\()?\s*\w+\[\s*['\"]max_drawdown['\"]\s*\])", r"\1",
     r"positive magnitude", "max_drawdown kept as the report's positive magnitude"),
    # know79 group 1 (selftest79_1.py): a whole-task bundle is the single source of its files.
    # (a) a cell that calls a *_bundle but emits only ONE of its files emits them all (the per-file cells ran out
    #     of model requests after a few files and left the rest as stubs);
    (r"^(?![\s\S]*\.items\(\))[\s\S]*\b[A-Za-z_]\w*\s*=\s*\w+_bundle\(",
     r"(?m)^([ \t]*)emit\(\s*(['\"])([^'\"\n]+)\2\s*,\s*([A-Za-z_]\w*)\[\s*\2\3\2\s*\]\s*\)[ \t]*$",
     "\\1for _k79, _v79 in (\\4.items() if getattr(\\4, '_t1_bundle', False) else ()):\n"
     "\\1    emit(_k79, _v79) if '.' in str(_k79) else None\n\\1emit(\\2\\3\\2, \\4[\\2\\3\\2])",
     r"", "bundle cell emits every file of the bundle"),
    # (b) a later cell that rebuilds a file by hand (no bundle call, no file-name subscript) emits the bundle's
    #     object for that file when an earlier bundle produced it (hand-typed intermediates / a swapped helper
    #     overwrote correct files); a file no bundle produced is emitted as written.
    (r"^(?![\s\S]*_bundle\()(?![\s\S]*\w\[\s*['\"][^'\"\n]+\.(?:json|csv|parquet|tsv|txt|md)['\"]\s*\])[\s\S]*\bemit\(",
     r"\bemit\(\s*(['\"])([^'\"\n]+\.(?:json|csv|parquet|tsv|txt|md))\1\s*,\s*([A-Za-z_]\w*)\s*\)",
     "emit(\\1\\2\\1, __import__('qf_toolkit.helpers79_1', fromlist=['_']).bundle_lookup(\\1\\2\\1, \\3, globals()))",
     r"", "hand-rebuilt file -> the bundle's object when a bundle produced it"),
]


# know87: silent unit / definition errors around a documented helper (no crash, wrong numbers). Each rewrite is
# a textbook identity of the helper it touches, fires only on the exact misuse and leaves other code unchanged.
_OPERAND = r"[A-Za-z_]\w*(?:\.\w+)*(?:\[[^\[\]\n]*\])*"
_DEC4 = r"(?:1e4|1e\+4|1\.0e4|10000(?:\.0*)?|10_000|100(?:\.0*)?\s*\*\*\s*2)"


_BOUND: dict[str, set] = {}     # know87: helper-result names bound in EARLIER cells of this kernel (cells share ns)


def _assigned_from(code: str, call: str) -> list[str]:
    """Names bound to the result of `call(` (plain or tuple-unpacked assignment targets) in this code, plus the
    names an earlier cell of the same kernel bound to it (a later cell reads them from the shared namespace)."""
    names = []
    for m in re.finditer(r"(?m)^[ \t]*([A-Za-z_][\w \t,]*?)\s*=\s*(?:[\w.]+\.)?" + call + r"\(", code):
        names += [n.strip() for n in m.group(1).split(",") if re.fullmatch(r"\s*[A-Za-z_]\w*\s*", n)]
    seen = _BOUND.setdefault(call, set())
    out = names + [n for n in sorted(seen) if n not in names]
    seen.update(names)
    return out


def _gpd_on_excesses(code: str, text: str) -> tuple[str, str]:
    """POT: with floc=0 the GPD is the law of the EXCESS X - u over the threshold (Pickands-Balkema-de Haan;
    McNeil, Frey & Embrechts 2015 QRM s.5.2; Coles 2001 s.4.2). genpareto.fit(EXCEEDANCES, floc=0) on values
    x > u (not x - u) is rewritten to genpareto.fit(EXCEEDANCES - u, floc=0)."""
    if "genpareto" not in code or not re.search(r"floc\s*=\s*0(?:\.0*)?\b", code):
        return code, ""
    new = code
    # inline: genpareto.fit(L[L > u], floc=0)
    new = re.sub(r"(genpareto\.fit\(\s*)([A-Za-z_]\w*)\[\s*\(?\s*\2\s*>\s*(" + _OPERAND + r")\s*\)?\s*\](\s*,\s*floc\s*=\s*0(?:\.0*)?\b)",
                 lambda m: f"{m.group(1)}{m.group(2)}[{m.group(2)} > {m.group(3)}] - {m.group(3)}{m.group(4)}", new)
    # named: X = L[L > u] (the whole right-hand side) ... genpareto.fit(X, floc=0)
    for m in re.finditer(r"(?m)^[ \t]*([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\[\s*\(?\s*\2\s*>\s*(" + _OPERAND +
                         r")\s*\)?\s*\][ \t]*(?:#.*)?$", code):
        x, u = m.group(1), m.group(3)
        later = code[m.end():]
        if re.search(r"(?m)^[ \t]*" + re.escape(x) + r"\s*(?:-=|=\s*" + re.escape(x) + r"\s*-)", later):
            continue                       # the program turns them into excesses itself
        new = re.sub(r"(genpareto\.fit\(\s*)" + re.escape(x) + r"(\s*,\s*floc\s*=\s*0(?:\.0*)?\b)",
                     lambda k: f"{k.group(1)}{x} - {u}{k.group(2)}", new)
    return (new, "genpareto.fit(exceedances, floc=0) -> fit on the excesses x - u (POT)") if new != code else (code, "")


def _garch_evt_already_scaled(code: str, text: str) -> tuple[str, str]:
    """garch_evt_var_es returns (VaR, ES) already scaled by the conditional vol (sigma_T x the standardised GPD
    tail, McNeil & Frey 2000). A second multiplication of its values by a vol is dropped."""
    names = _assigned_from(code, "garch_evt_var_es")
    if not names:
        return code, ""
    new = code
    volish = re.compile(r"vol|sigma", re.I)
    inner = re.compile(r"""\[\s*['"](?:pot|garch)['"]""")     # the standardised fit / the GARCH dict stay as they are

    def keep(ref_txt, other):
        return bool(volish.search(other)) and not volish.search(ref_txt) and not inner.search(ref_txt)

    for nm in set(names):
        ref = re.escape(nm) + r"(?:\[[^\[\]\n]*\])*"
        new = re.sub(r"(?<![\w.])(" + ref + r")\s*\*\s*(" + _OPERAND + r")",
                     lambda m: m.group(1) if keep(m.group(1), m.group(2)) else m.group(0), new)
        new = re.sub(r"(?<![\w.\]])(" + _OPERAND + r")\s*\*\s*(" + ref + r")(?![\w\[])",
                     lambda m: m.group(2) if keep(m.group(2), m.group(1)) else m.group(0), new)
    return (new, "garch_evt_var_es values are already in loss units: the extra x vol was dropped") if new != code else (code, "")


def _garch_forecast_decimal(code: str, text: str) -> tuple[str, str]:
    """garch11_forecast(scale=100, the default) returns omega, long_run_variance and sigma2_next already in DECIMAL
    units. A second descaling hidden in a product denominator (g['omega'] / (1e4 * (1 - a - b))) is removed; and
    when the task defines the vol-of-vol as the standard deviation of the first differences of the in-sample
    annualised conditional volatility, a hand-built value passed to delta_gamma_vega_var is replaced by the
    fitted series' own g['vol_of_vol_daily'] (the same definition, from the arch conditional volatility)."""
    calls = re.findall(r"garch11_forecast\(([^()]*(?:\([^()]*\)[^()]*)*)\)", code)
    if any(re.search(r"\bscale\s*=\s*(?!100(?:\.0*)?\b)", c) for c in calls):
        for n in _assigned_from(code, "garch11_forecast"):
            _BOUND.get("garch11_forecast", set()).discard(n)
        return code, ""                    # a native-unit fit: its outputs are what they are
    names = [n for n in _assigned_from(code, "garch11_forecast")]
    if not names:
        return code, ""
    new, notes = code, []
    for nm in set(names):
        key = re.escape(nm) + r"\[\s*(?P<q>['\"])(?:omega|long_run_variance|sigma2_next)(?P=q)\s*\]"
        new2 = re.sub(r"(?P<k>" + key + r")\s*/\s*\(\s*" + _DEC4 + r"\s*\*\s*", r"\g<k> / (", new)
        new2 = re.sub(r"(?P<k>" + key + r")\s*/\s*" + _DEC4 + r"(?![\w.*])", r"\g<k>", new2)
        if new2 != new:
            notes.append("decimal GARCH output not descaled twice")
            new = new2
    t = (text or "").lower()
    if re.search(r"vol(?:atility)?[- ]of[- ]vol", t) and re.search(r"first differences?", t) \
            and "vol_of_vol" not in new and "delta_gamma_vega_var(" in new:
        g = names[0]
        m = re.search(r"delta_gamma_vega_var\(", new)
        args = _call_args(new, m.end())
        kw = re.search(r"delta_gamma_vega_var\([^)]*\bvov_daily\s*=\s*([A-Za-z_]\w*)", new)
        vv = kw.group(1) if kw else None
        if vv is None and args is not None and len(args[0]) >= 7 and "=" not in args[0][6].split("(")[0]:
            vv = args[0][6].strip()
        if vv:
            if re.fullmatch(r"[A-Za-z_]\w*", vv):
                new3 = re.sub(r"(?m)^([ \t]*)" + re.escape(vv) + r"\s*=\s*(?!.*\bvol_of_vol).*$",
                              lambda k: f"{k.group(1)}{vv} = {g}['vol_of_vol_daily']", new, count=0)
                if new3 != new:
                    new = new3
                    notes.append("vol-of-vol from the fitted conditional-vol series (g['vol_of_vol_daily'])")
    return (new, "; ".join(notes)) if new != code else (code, "")


def _call_args(code: str, start: int):
    """Top-level comma-separated arguments of the call whose '(' ends at `start` -> (list, (start, end))."""
    depth, cur, out, i = 0, "", [], start
    while i < len(code):
        ch = code[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                out.append(cur)
                return out, (start, i)
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
        i += 1
    return None


_COMPONENT_DEF = re.compile(r"arange\(|gammaln\(|factorial|poisson|\.pmf\(|\.logpmf\(")


def _mixture_axis(code: str, text: str) -> tuple[str, str]:
    """know87 (round 2): a finite-mixture log-likelihood sums the components FOR EACH observation,
    log p(x_i) = logsumexp_n [log w_n + log phi_n(x_i)], then sums over i (Titterington, Smith & Makov 1985
    s.1.2; the Poisson-mixture density of jump-diffusion log returns, Press 1967 / Ball & Torous 1983).
    When the per-component log-weights are laid along axis 1 (W[None, :], W built from arange / gammaln /
    a Poisson pmf) and the observations along axis 0 (x[:, None]), logsumexp(..., axis=0) sums over the
    OBSERVATIONS - the wrong axis; it becomes axis=1 (and the mirror case W[:, None] with axis=1 -> axis=0)."""
    if "logsumexp(" not in code:
        return code, ""
    comp = set()
    for m in re.finditer(r"(?m)^[ \t]*([A-Za-z_]\w*)\s*=\s*(.+)$", code):
        if _COMPONENT_DEF.search(m.group(2)) or any(re.search(r"(?<![\w.])" + re.escape(c) + r"\b", m.group(2)) for c in comp):
            comp.add(m.group(1))
    if not comp:
        return code, ""
    row = r"\[\s*(?:None|np\.newaxis)\s*,\s*:\s*\]"     # [None, :]  -> a vector along axis 1
    col = r"\[\s*:\s*,\s*(?:None|np\.newaxis)\s*\]"     # [:, None]  -> a vector along axis 0
    out, pos, changed = [], 0, False
    for m in re.finditer(r"logsumexp\(", code):
        if m.start() < pos:
            continue
        args = _call_args(code, m.end())
        if args is None:
            continue
        parts, (s, e) = args
        expr = parts[0]
        ax = None
        for i, p in enumerate(parts[1:], 1):
            k = re.fullmatch(r"\s*axis\s*=\s*(-?\d)\s*", p)
            if k:
                ax = (i, int(k.group(1)))
        if ax is None:
            continue
        names_row = {n for n in re.findall(r"([A-Za-z_]\w*)\s*" + row, expr)}
        names_col = {n for n in re.findall(r"([A-Za-z_]\w*)\s*" + col, expr)}
        rest = code.replace(expr, "")
        new_ax = None
        if ax[1] == 0 and names_row & comp and not names_col and re.search(col, rest):
            new_ax = 1
        elif ax[1] == 1 and names_col & comp and not names_row and re.search(row, rest):
            new_ax = 0
        if new_ax is None:
            continue
        parts = list(parts)
        parts[ax[0]] = re.sub(r"(axis\s*=\s*)-?\d", r"\g<1>" + str(new_ax), parts[ax[0]])
        out.append(code[pos:m.end()] + ",".join(parts))
        pos, changed = e, True
    if not changed:
        return code, ""
    out.append(code[pos:])
    return "".join(out), "mixture log-likelihood: logsumexp over the COMPONENT axis (per observation), not over the observations"


_POS_MDD = re.compile(r"(?:\w*max\w*draw_?down\w*|maximum drawdown)`?[^.\n]{0,40}?\b(?:positive magnitude|"
                      r"non-negative number|positive number|as a positive)", re.I)


def _mdd_call_positive(code: str, text: str) -> tuple[str, str]:
    """know89 (guard C, flip census): the toolkit's max_drawdown(r) returns a NEGATIVE float (its API line); when
    the TASK states the maximum drawdown as a positive magnitude, a call that the program does not negate itself
    becomes abs(max_drawdown(...)) - the stated convention reached through a helper CALL, not only through a
    report subscript (the _PRERUN_SET rule). Untouched: a local def / variable named max_drawdown, .max_drawdown(
    methods, calls already inside abs( or after a unary minus."""
    if "max_drawdown(" not in code or not _POS_MDD.search(text or ""):
        return code, ""
    if re.search(r"^\s*def\s+max_drawdown\b|^\s*max_drawdown\s*=", code, re.M):
        return code, ""
    out, pos, changed = [], 0, False
    for m in re.finditer(r"(?<![\w.'\"])max_drawdown\(", code):
        if m.start() < pos:
            continue
        before = code[max(0, m.start() - 12):m.start()]
        if re.search(r"(?:abs\(|-)\s*(?:qf_toolkit\.)?$", before):
            continue
        args = _call_args(code, m.end())
        if args is None:
            continue
        _parts, (_s, e) = args
        out.append(code[pos:m.start()] + "abs(" + code[m.start():e + 1] + ")")
        pos, changed = e + 1, True
    if not changed:
        return code, ""
    out.append(code[pos:])
    return "".join(out), "max_drawdown(...) -> abs(max_drawdown(...)): the task states a positive magnitude"


_PRERUN_FN = [_gpd_on_excesses, _garch_evt_already_scaled, _garch_forecast_decimal, _mixture_axis, _mdd_call_positive]


def prerun(code: str, instruction: str | None = None, fn_only: bool = False) -> tuple[str, list[str]]:
    """(code with the contradicting overrides dropped, notes). Never raises. fn_only (know87, whole programs run
    by soft_run): only the helper-unit identities of _PRERUN_FN, not the cell-oriented drops and settings."""
    notes: list[str] = []
    try:
        import os
        text = (instruction if instruction is not None else os.environ.get("T1_INSTRUCTION", "")).lower()
        for call, arg, guard, note in (() if fn_only else _PRERUN):
            if not re.search(call, code):
                continue
            if guard and re.search(guard, text):
                continue
            if "INPUTS" in arg:
                new = re.sub(arg, r"\1=None", code)
            else:
                new = re.sub(r"(" + call + r"[^)]*?)" + arg, r"\1", code, flags=re.S)
            if new != code:
                code = new
                notes.append(note)
        import sys as _sys
        _h79 = _sys.modules.get("qf_toolkit.helpers79_1")
        bundle_seen = bool(getattr(_h79, "_CALLS", 0))   # know79: a whole-task bundle ran in this kernel
        for call, arg, repl, need, note in (() if fn_only else _PRERUN_SET):
            if note.startswith("hand-rebuilt file") and not bundle_seen:
                continue
            if re.search(call, code) and re.search(need, text):
                new = re.sub(arg, repl, code)
                if new != code:
                    code = new
                    notes.append(note)
        for fn in _PRERUN_FN:              # know87: silent unit / definition errors around documented helpers
            try:
                new, note = fn(code, text)
            except Exception:  # noqa: BLE001
                continue
            if new != code:
                try:
                    compile(new, "<prerun>", "exec")
                except SyntaxError:
                    continue
                code = new
                notes.append(note)
    except Exception:  # noqa: BLE001 - a rewrite aid, never a failure cause
        pass
    return code, notes


def apply(code: str, error: str, state_text: str = "") -> tuple[str, str]:
    """(patched code, note) - the code unchanged and '' when no rule applies."""
    err = error or ""
    new, note = code, ""

    if "does not support strftime" in err or "has no attribute 'strftime'" in err:
        new = _STRFTIME.sub(r"_fmt_date(\1, ", code)
        note = "strftime -> _fmt_date (NaT / datetime64 safe)"
    elif ("only 0-dimensional arrays can be converted" in err or "only length-1 arrays" in err or "only size-1 arrays" in err
          or re.search(r"(?:float|int)\(\) argument must be a string.*not '(?:Series|DataFrame|list|ndarray)'", err)):
        new = re.sub(r"(?<![\w.])float\(", "_scalar(", code)
        new = re.sub(r"(?<![\w.])int\(", "_as_int(", new)
        note = "float()/int() of an array -> first element"
    elif "'module' object is not callable" in err:
        # know71: `import datetime` then datetime(...) / `from scipy import stats` then stats(...)
        new = re.sub(r"(?<![\w.])datetime\(", "datetime.datetime(", code)
        new = re.sub(r"(?<![\w.])date\(", "datetime.date(", new) if "import datetime" in code and "from datetime" not in code else new
        new = re.sub(r"(?<![\w.])timedelta\(", "datetime.timedelta(", new) if "from datetime" not in code else new
    elif "'numpy.ndarray' object has no attribute 'values'" in err:
        new = re.sub(r"\.values\b", "", code)
        note = "ndarray.values -> the array itself"
    elif re.search(r"module 'pandas' has no attribute '(\w+)'", err):
        attr = re.search(r"module 'pandas' has no attribute '(\w+)'", err).group(1)
        if hasattr(np, attr) or attr == "random":
            new = re.sub(r"\bpd\." + attr + r"\b", "np." + attr, code)
            note = f"pd.{attr} -> np.{attr}"
    elif "Cannot convert non-finite values (NA or inf) to integer" in err or "IntCastingNaNError" in err:
        new = re.sub(r"\.astype\(\s*(int|'int'|\"int\"|np\.int64|'int64'|\"int64\")\s*\)", r".fillna(0).astype(\1)", code)
        note = "astype(int) -> fillna(0).astype(int)"
    # ---- know72 rules (14 Oct census of 67 failed units) ----
    elif ("Invalid comparison between dtype=datetime64" in err or "Cannot compare tz-naive and tz-aware" in err
          or "Cannot convert input [" in err or "with DatetimeArray is no longer supported" in err) and "pd.to_datetime(" in code:
        # every pd.to_datetime(x) -> _to_naive(x): parse with utc=True and drop the zone; epoch ints
        # become timestamps; comparisons then happen naive against naive.
        new = re.sub(r"\bpd\.to_datetime\(", "_to_naive(", code)
        note = "pd.to_datetime -> _to_naive (utc=True, tz dropped, epoch ms parsed)"
    elif ("Data must be 1-dimensional" in err or "Buffer has wrong number of dimensions" in err
          or "Per-column arrays must each be 1-dimensional" in err):
        new = re.sub(r"\bpd\.Series\(\s*([A-Za-z_][\w.\[\]]*)\s*([,)])", r"pd.Series(_flat(\1)\2", code)
        note = "pd.Series(x) -> pd.Series(_flat(x)) (2-D (n,1) arrays flattened)"
    elif "got an unexpected keyword argument 'utc'" in err and "pd.Timestamp(" in code:
        new = re.sub(r"\bpd\.Timestamp\(([^()]+?),\s*utc\s*=\s*True\)", r"pd.to_datetime(\1, utc=True)", code)
        note = "pd.Timestamp(x, utc=True) -> pd.to_datetime(x, utc=True)"
    elif re.search(r"^\s*import pandas as np\s*$", code, re.M):
        # know76: 'import pandas as np' shadowed numpy (ou-jump, mc-greek and geske runs)
        new = re.sub(r"^(\s*)import pandas as np\s*$", r"\1import numpy as np", code, flags=re.M)
        note = "import pandas as np -> import numpy as np"
    elif re.search(r"argument of type 'method' is not iterable|Invalid file path or buffer object type", err) \
            and re.search(r"pd\.read_(?:json|csv|parquet)\(\s*INPUTS\[", code):
        new = re.sub(r"pd\.read_(?:json|csv|parquet)\(\s*(INPUTS\[[^\]]+\])[^)]*\)", r"\1", code)
        note = "pd.read_*(INPUTS[...]) -> INPUTS[...] (already parsed)"
    elif "mergesot" in code:
        new = re.sub(r"kind\s*=\s*(['\"])mergesot\1", 'kind="mergesort"', code)
        note = "kind='mergesot' -> 'mergesort'"
    elif re.search(r"between\(\) got an unexpected keyword argument '(neither|both|left|right)'", err):
        new = re.sub(r"\.between\(([^()]*?),\s*\b(neither|both|left|right)\s*=\s*True\s*\)", r'.between(\1, inclusive="\2")', code)
        new = re.sub(r"\.between\(([^(),]+),([^(),]+),\s*(['\"])(neither|both|left|right)\3\s*\)", r'.between(\1,\2, inclusive="\4")', new)
        note = "between(neither=True) -> between(inclusive='neither')"
    elif "SeedSequence expects int" in err:
        new = re.sub(r"default_rng\(\s*([^()]+?)\s*\)", r"default_rng(int(\1))", code)
        note = "default_rng(seed) -> default_rng(int(seed))"
    elif "module 'numpy' has no attribute 'special'" in err:
        new = re.sub(r"\bnp\.special\.", "scipy.special.", code)
        if new != code and "import scipy.special" not in new:
            new = "import scipy.special\n" + new
        note = "np.special -> scipy.special"
    elif re.search(r"KeyError: '[\w ]+ per unit time'", err):
        new = re.sub(r"\[\s*(['\"])(\w+) per unit time\1\s*\]", r"[\1\2\1]", code)
        note = "'<key> per unit time' -> '<key>'"
    elif "'Series' object has no attribute 'days'" in err or "'Series' object has no attribute 'seconds'" in err:
        # know73: a timedelta Series needs the .dt accessor
        new = re.sub(r"(?<!\.dt)\.(days|seconds|total_seconds)\b", r".dt.\1", code)
        note = "Series.days -> Series.dt.days"
    elif "Cannot convert input [" in err and "pd.Timestamp(" in code:
        # know73: pd.Timestamp() of a column -> a parsed, tz-naive column
        new = re.sub(r"\bpd\.Timestamp\(", "_to_naive(", code)
        note = "pd.Timestamp(column) -> _to_naive(column)"
    elif "cannot import name 'quad' from 'scipy.optimize'" in err:
        new = code.replace("from scipy.optimize import quad", "from scipy.integrate import quad")
        new = re.sub(r"\boptimize\.quad\b", "integrate.quad", new)
        note = "quad lives in scipy.integrate"
    elif "has no attribute 'add_const'" in err or "KeyError: 'add_const'" in err or "'add_const'" in err:
        new = re.sub(r"\badd_const\b", "add_constant", code)
        note = "statsmodels add_const -> add_constant"
    elif re.search(r"module 'math' has no attribute|must be real number, not (?:numpy\.ndarray|Series|list)|only (?:size-1|length-1) arrays", err) and "math." in code:
        new = re.sub(r"\bmath\.(exp|log|sqrt|sin|cos|tan|fabs|floor|ceil|log1p|expm1)\(", r"np.\1(", code)
        note = "math.<fn> on arrays -> np.<fn>"
    elif "No such file or directory" in err:
        m = re.search(r"No such file or directory: '([^']+)'", err)
        found = _resolve_input(m.group(1)) if m else None
        if found:
            new = code.replace(f"'{m.group(1)}'", f"'{found}'").replace(f'"{m.group(1)}"', f'"{found}"')
            note = f"path {m.group(1)!r} -> input {found!r}"
    elif "the JSON object must be str" in err:
        new = re.sub(r"\bjson\.loads\(\s*INPUTS\[([^\]]+)\]\s*\)", r"INPUTS[\1]", code)
        note = "json.loads(INPUTS[...]) -> INPUTS[...]"
    elif "argument should be a str or an os.PathLike" in err:
        new = re.sub(r"\b(pd\.read_csv|pd\.read_json|open|read_table|read_json)\(\s*INPUTS\[([^\]]+)\]\s*[,)]",
                     lambda m: f"({m.group(0)[len(m.group(1)) + 1:].rstrip(',)')})" + (m.group(0)[-1] if m.group(0)[-1] == ")" else ","), code)
        note = "read of INPUTS[...] -> the loaded object"
    else:
        m = re.search(r"KeyError: ['\"]([^'\"]+)['\"]", err)
        if m:
            cols = _columns_in_state(state_text)
            close = difflib.get_close_matches(m.group(1), cols, n=1, cutoff=0.8)
            if close and close[0] != m.group(1):
                new = code.replace(f"'{m.group(1)}'", f"'{close[0]}'").replace(f'"{m.group(1)}"', f'"{close[0]}"')
                note = f"column {m.group(1)!r} -> {close[0]!r} (closest existing)"
        m2 = re.search(r"name '(\w+)' is not defined", err)
        if m2 and not note:
            names = _names_in_state(state_text)
            close = difflib.get_close_matches(m2.group(1), names, n=1, cutoff=0.85)
            if close and close[0] != m2.group(1):
                new = re.sub(r"\b" + re.escape(m2.group(1)) + r"\b", close[0], code)
                note = f"name {m2.group(1)!r} -> {close[0]!r} (closest defined)"
    if new == code:
        return code, ""
    try:
        compile(new, "<autofix>", "exec")
    except SyntaxError:
        return code, ""
    return new, note
