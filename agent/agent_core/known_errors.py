"""Known errors -> the concrete fix, written by Python into the repair request (6 Oct).

Every entry comes from a measured failure: the 2 Oct autopsy of 252 runs and the error census of
the s12_40..42 batches (209 crashes). The model repeated the same error on the same cell up to
three times when it only saw the traceback; here the traceback is paired with the fix that worked,
and, where the live namespace is known, with the real column names and shapes.
"""
from __future__ import annotations

import difflib
import re

#: (pattern, fix). {g1}, {g2} are the pattern's groups.
RULES: list[tuple[str, str]] = [
    (r"only 0-dimensional arrays can be converted to Python scalars|only (?:length-1|size-1) arrays can be converted",
     "float()/int() received an array with several elements. Reduce it first: x.item() for a 1-element array, "
     "np.asarray(x).ravel()[0] for the first value, or keep it as an array and emit a table."),
    (r"Length mismatch: Expected axis has 1 elements, new values have (\d+)",
     "The table was read as ONE column: the file is not comma-separated. Use INPUTS[<name>] (delimiter already "
     "detected) or pd.read_csv(path, sep=';' / '\\t' / '|'); never assign {g1} column names to a 1-column frame."),
    (r"Cannot convert input \[|Tz-aware datetime|tz-aware|Cannot compare tz-naive and tz-aware",
     "Timestamps are tz-aware or epoch numbers. Parse once: pd.to_datetime(col, utc=True).dt.tz_localize(None) "
     "for text, pd.to_datetime(col, unit='ms') for epoch milliseconds; compare only naive with naive. INPUTS "
     "already did this for date-like columns (see RUNTIME STATE dtypes)."),
    (r"EmptyDataError|No columns to parse from file",
     "The file read is EMPTY or was never written - it is one of YOUR deliverables, not an input. Use the "
     "in-memory variable; emit() writes deliverables, do not read them back."),
    (r"KeyError: (\d+)\b",
     "Integer label {g1} is not in the index (the frame was filtered or sorted, so the RangeIndex has gaps). "
     "Use .iloc[i] for positions, or .reset_index(drop=True) first."),
    (r"KeyError: ['\"]([^'\"]+)['\"]",
     "Column/key '{g1}' does not exist. Use a name from the RUNTIME STATE columns exactly{near}."),
    (r"['\"]?([^'\"]+)['\"]? is not in list",
     "list.index('{g1}') on a label that is absent. Build a dict {{label: i}} and look up with .get, or "
     "filter with `if x in labels`; for rating labels include every state the matrix carries (NR, D, SD, WD)."),
    (r"operands could not be broadcast together with shapes \(([\d, ]+)\) \(([\d, ]+)\)",
     "Shapes ({g1}) and ({g2}) do not align. Make the smaller one explicit: w[:, None] to broadcast over "
     "columns, w[None, :] over rows, or .reshape; check which axis is paths and which is time/assets."),
    (r"shape mismatch: objects cannot be broadcast to a single shape",
     "Two arrays of different shapes were combined element-wise; align them on purpose (np.broadcast_to, "
     "[:, None]) or compute per column with a comprehension."),
    (r"object too deep for desired array|setting an array element with a sequence|inhomogeneous shape",
     "A list of unequal-length rows (or a column of lists) was turned into an array. Use np.vstack on rows of "
     "equal length, pd.DataFrame(list_of_dicts), or np.concatenate for ragged data."),
    (r"output array is read-only|assignment destination is read-only",
     "The array is a read-only view from pandas: use arr = frame[col].to_numpy().copy() before writing into it."),
    (r"ufunc '\w+' did not contain a loop with signature matching types|not supported for dtype 'str'|unsupported operand type\(s\) for [-+*/]: 'str'",
     "A numeric column is text (blanks, commas, %, letters). Convert: pd.to_numeric(col.astype(str).str.replace(',', '').str.rstrip('%'), errors='coerce'); INPUTS converts obvious cases, this one needs an explicit map."),
    (r"invalid literal for int\(\) with base 10: '([^']*)'",
     "'{g1}' is not an integer - the column holds codes/labels. Map labels to numbers with a dict, or keep them as categories."),
    (r"the JSON object must be str, bytes or bytearray, not (dict|list)",
     "INPUTS[<file>.json] is already a {g1} - use it directly, no json.loads / open."),
    (r"argument should be a str or an os.PathLike object.*not '(dict|list|DataFrame)'",
     "A loaded object ({g1}) was passed where a path was expected. INPUTS[name] is the loaded content; "
     "do not open() or read_csv() it again."),
    (r"got an unexpected keyword argument '([^']+)'",
     "Keyword '{g1}' is not a parameter of that function. The toolkit drops unknown keywords and records it; "
     "for other libraries read the signature in TOOLKIT / the error and pass only its parameters."),
    (r"takes from (\d+) to (\d+) positional arguments? but (\d+) were given|takes (\d+) positional arguments? but (\d+) were given",
     "Too many positional values for that signature: pass the extras by keyword, or drop them (the toolkit "
     "functions return tuples/frames documented in TOOLKIT)."),
    (r"datetime\.date\([\d, ]+\) is not in list|Timestamp\('[^']+'\) is not in list",
     "Date types are mixed (datetime.date vs Timestamp vs string). Normalise both sides: pd.Timestamp(x).normalize(), "
     "then look up with a dict or .loc on a DatetimeIndex."),
    (r"'(\w+_gen)' object has no attribute '(\w+)'",
     "'{g2}' is a module-level function, not a method of the distribution: call scipy.stats.{g2}(...){where}."),
    (r"module '(\w+)' has no attribute '(\w+)'",
     "{g1}.{g2} does not exist{where}. Check the spelling against the imports already bound (np, pd, stats, optimize)."),
    (r"name '(\w+)' is not defined",
     "'{g1}' was never defined in this session (a cut reply, or a helper from an earlier draft). Define it in this "
     "cell or use the library name{where}; np, pd, math, json, scipy.stats/optimize and every qf_toolkit "
     "function are already imported."),
    (r"closing parenthesis '.' does not match opening parenthesis|unexpected EOF while parsing|was never closed|SyntaxError",
     "The code was cut mid-statement or a bracket is unbalanced: write a SHORTER cell (one deliverable, under 40 lines) and close every bracket."),
    (r"unknown money-market basis '(\w+)'",
     "'{g1}' is a currency, not a basis: pass basis=money_market_basis('{g1}') (360 or 365) or 'ACT/360' / 'ACT/365'."),
    (r"maturities must be \[",
     "The quotes are not on a regular k/freq grid: use par_curve(maturities, par_rates, freq), which interpolates, instead of bootstrap_par."),
    (r"Reindexing only valid with uniquely valued Index|cannot reindex on an axis with duplicate labels",
     "The index has duplicate labels (dates repeated). Drop duplicates or aggregate first: frame.groupby(level=0).last() or .drop_duplicates(subset=[date_col])."),
    (r"Can only use \.dt accessor with datetimelike values",
     "The column is text, not datetime: pd.to_datetime(col, utc=True).dt.tz_localize(None) first."),
    (r"could not convert string to float: '([^']*)'",
     "'{g1}' sits in a numeric column: strip units/blanks and pd.to_numeric(..., errors='coerce'); if it is a label, it belongs to a key column, not a value column."),
    (r"NaTType does not support strftime|'numpy\.datetime64' object has no attribute 'strftime'",
     "Format dates through pandas: pd.Timestamp(x).strftime(...) after dropping NaT (s.dropna()), or "
     "s.dt.strftime('%Y-%m-%d') on a Series; never call strftime on numpy.datetime64 or NaT."),
    (r"'module' object is not callable",
     "A MODULE was called like a function (e.g. riskx(...) or arch(...)). Call the function inside it: "
     "look up the name in TOOLKIT (from qf_toolkit import <function>) or arch.arch_model(...)."),
    (r"'str' object has no attribute '(iloc|loc|columns|values)'",
     "INPUTS[<file>] is TEXT here (a .txt/.md/.html input), not a table. Parse it (split lines, "
     "pd.read_csv(io.StringIO(text), sep=...)) or use the table input instead."),
    (r"'DataFrame' object has no attribute '(dt|str|cat)'",
     "df['<name>'] returned a DataFrame, not a column: the frame has TWO columns with that name (typical after "
     "reset_index() on an index that is also a column, concat or merge). Keep one: df = df.loc[:, "
     "~df.columns.duplicated()] (or drop the column before reset_index), then df['<name>'].{g1} works."),
    (r"Invalid comparison between dtype=datetime64|'<' not supported between instances of 'datetime\.datetime' and 'int'",
     "A datetime column is compared with an integer (epoch ms). Convert the integer: pd.to_datetime(x, unit='ms') "
     "(unit='s' for 10-digit epochs), then compare datetimes with datetimes."),
    (r"Cannot subtract tz-naive and tz-aware|Cannot compare tz-naive and tz-aware",
     "Mixed time zones: make both sides naive UTC - s.dt.tz_convert('UTC').dt.tz_localize(None) for tz-aware, "
     "pd.Timestamp(x).tz_localize(None) for a scalar."),
    (r"Cannot compare Timestamp with datetime\.date",
     "Compare like with like: pd.Timestamp(d) == ts, or ts.normalize() == pd.Timestamp(d)."),
    (r"einstein sum|einsum",
     "The einsum subscripts do not match the operands' ndim. Write the product explicitly: (w @ Sigma @ w) for a "
     "quadratic form, (returns * w).sum(axis=1) for portfolio returns, np.matmul for batches."),
    (r"Input operand 1 does not have enough dimensions|matmul: Input operand",
     "A scalar/float was used where a vector was needed: weights must be np.asarray(CFG['weights'], float) with "
     "shape (n,) - check CFG for the list and the asset order."),
    (r"SVD did not converge|Eigenvalues did not converge",
     "Non-finite values or an ill-conditioned matrix reached the decomposition: drop NaN rows first, then add a "
     "ridge (Sigma + 1e-8*np.eye(n)) or use np.linalg.pinv."),
    (r"f\(a\) and f\(b\) must have different signs",
     "brentq needs a bracket: scan a grid for a sign change first (e.g. np.linspace over the parameter), then "
     "brentq on the bracketing pair; if no sign change, use minimize_scalar on |f|."),
    (r"Cannot convert non-finite values \(NA or inf\) to integer|IntCastingNaNError",
     "NaN cannot become int: fill or drop them (s.fillna(0).astype(int)) and use 'Int64' for nullable ints."),
    (r"Unalignable boolean Series provided as indexer",
     "The boolean mask has a different index than the frame: use mask.to_numpy() or build it from the same frame "
     "(frame[frame[col] > x])."),
    (r"The truth value of a Series is ambiguous",
     "A Series was used in `if`: reduce it first (s.any(), s.all(), s.iloc[0]) or use np.where for element-wise."),
    (r"kernel reply unreadable",
     "The cell printed too much (a whole frame or array). Print shapes and .head(3) only; never print large objects."),
    (r"Data must be 1-dimensional",
     "A 2-D array was given where a Series/1-D was expected (e.g. arch_model(returns) with a matrix): fit per column "
     "in a loop, or pass frame[col]."),
    (r"'ARCHModelResult' object has no attribute",
     "arch results expose .params (a Series: 'omega', 'alpha[1]', 'beta[1]'), .conditional_volatility, .loglikelihood; "
     "or use the toolkit fit_garch11(returns) -> (omega, alpha, beta, cond_var)."),
    (r"MemoryError|Unable to allocate",
     "The array is too large: simulate in chunks of paths (e.g. 20,000 at a time) and accumulate sums / quantile samples."),
    (r"LinAlgError|Singular matrix|not positive definite",
     "The covariance/matrix is singular: add a small ridge (eps*I), use np.linalg.pinv, or drop perfectly collinear columns; check for constant columns and NaNs first."),
    # ---- know72 (14 Oct census of the 67 units that had not passed) ----
    (r"Invalid comparison between dtype=datetime64\[[^\]]*\] and (int|Timestamp|str)",
     "A datetime column was compared with a {g1}: parse BOTH sides once with _to_naive(x) (tz dropped, epoch ints read as "
     "milliseconds) and compare timestamps with timestamps; never compare a datetime column with an integer or a tz-aware stamp."),
    (r"Buffer has wrong number of dimensions|Per-column arrays must each be 1-dimensional",
     "A 2-D (n,1) array was passed where a 1-D Series was expected: use _flat(x) / x.ravel() / frame.iloc[:, 0]."),
    (r"matmul: Input operand \d+ has a mismatch in its core dimension",
     "Matrix shapes do not chain (A @ B needs A.shape[1] == B.shape[0]): print the shapes, transpose the one that is "
     "(assets x time) instead of (time x assets), and align on the same asset order before multiplying."),
    (r"weights length must equal number of price columns",
     "The toolkit backtest got weights for a different set of assets than the price frame carries: build weights as a "
     "dict/Series keyed by the price columns (missing assets = 0) and pass prices[list(weights)]."),
    (r"severity_dist\('gamma'\): 'beta' is ambiguous",
     "Name the gamma parameter explicitly: severity_dist('gamma', shape=k, scale=theta) (a 'rate' is 1/scale); the task's "
     "words 'scale' or 'rate' decide which one it gives."),
    (r"too many values to unpack \(expected (\d+)\)",
     "The function returns more items than you unpacked ({g1}): assign the result to one name and index it, or read the "
     "toolkit signature in TOOLKIT and unpack exactly its tuple."),
    (r"could not get source code",
     "inspect.getsource cannot see code defined in a cell. A deliverable that is a .py file must be written as TEXT: build "
     "the function source as a string (triple-quoted) and emit('name.py', text); then exec it from the file to run it."),
    (r"no input named '([^']+\.py)'",
     "'{g1}' is a helper SCRIPT shipped with the task, not a data table: load it by path with importlib.util.spec_from_file_location "
     "and call its functions; INPUTS carries data files only."),
    (r"IndexError: index (\d+) is out of bounds for axis \d+ with size (\d+)",
     "Index {g1} exceeds the size {g2}: the array is one element shorter than assumed (a diff/shift dropped a row, or maturities "
     "are 1..N not 0..N). Loop over range(len(x)) and align by label, not by a hard-coded count."),
    (r"Boolean index has wrong length: (\d+) instead of (\d+)",
     "The mask ({g1}) was built from a different array than the one it filters ({g2}): build the mask from the same frame/array "
     "(or .reindex it) before filtering."),
    (r"Invalid value '[^']*' for dtype 'int64'",
     "A fractional value was written into an integer column: cast the column to float first (frame[col] = frame[col].astype(float))."),
    (r"TimeoutError: the cell ran longer",
     "Re-size, do not retry: paths/steps/grid 10x smaller, numpy vectorised over paths, no per-path or per-day Python loop; "
     "variables already computed are still in the kernel."),
    # ---- know76 (per-unit gap analysis of the know75 full run: swinging + never-passed) ----
    (r"(not enough|too many) values to unpack \(expected (\d+)(?:, got \d+)?\)",
     "Unpack exactly what the helper returns ({g1} values, expected {g2}). Several toolkit helpers return TUPLES: "
     "panel_to_array -> (dates, ids, array); newton_iv_vec -> (sigma, converged, iterations); spot_bump_krd -> (krd, P, "
     "P_bumped); coupon_schedule -> (previous_date, future_dates); clean_venue_snapshots -> (clean, counts, pairs); "
     "kupiec_pof -> (lr, p); hold_weights_returns -> (returns, day_index). Read the API line; when a *_bundle / *_report "
     "helper covers the task, call it once and emit its outputs."),
    (r"Length of values \(3\) does not match length of index",
     "A 3-tuple was assigned as a column (newton_iv_vec returns (sigma, converged, iterations)): iv, ok, n = newton_iv_vec(...); df['implied_vol'] = iv."),
    (r"float\(\) argument must be a string or a real number, not 'dict'",
     "A helper returned a dict, not a number: index it (e.g. prices[opt]['mc_price'], guidance_midpoint_musd(text)['midpoint']); "
     "print(sorted(result)) once to see its keys."),
    (r"unsupported operand type\(s\) for [-+*/]: '\w+' and 'tuple'",
     "One operand is a TUPLE returned by a helper (e.g. hold_weights_returns -> (returns, day_index)): unpack it first."),
    (r"All arrays must be of the same length",
     "One column has fewer entries than the others: build each block separately and pd.concat them, or repeat the short "
     "column explicitly (np.concatenate([x, x]))."),
    (r"positional argument follows keyword argument",
     "Python call syntax: every positional argument must come before the first name=value argument (or name them all)."),
    (r"AxisError: axis 1 is out of bounds for array of dimension 1",
     "A 1-D array was used where the [T x n] matrix of ALL assets is needed: pass prices[asset_cols].to_numpy(), not one column."),
    (r"argument of type 'method' is not iterable|module 'pandas' has no attribute 'jsonl'|Invalid file path or buffer object type: <class 'pandas",
     "INPUTS[...] is already parsed (.csv/.jsonl -> DataFrame, .json -> dict): use it directly, never pass it to a pandas reader."),
    (r"cannot insert (\w+), already exists",
     "The frame already has a '{g1}' column: skip the insert (if '{g1}' not in df.columns: df.insert(...))."),
    (r"StopIteration",
     "A selection was EMPTY (next() on nothing): take labels from df[col].unique() and match them exactly; print them first."),
    (r"invalid index to scalar variable",
     "A scalar was indexed like an array: check which argument is a number and which a series (window ints vs MA arrays)."),
    (r"can't multiply sequence by non-int of type '(str|float)'",
     "A list was multiplied like a number: convert with np.asarray(x, dtype=float), or a parameter set given as a dict was read as a list."),
    # ---- know75 (per-unit gap analysis of the know74 full run) ----
    (r"embedded null byte|a bytes-like object is required, not 'str'|'str' object has no attribute 'iloc'|Error tokenizing data\. C error: Expected \d+ fields",
     "INPUTS[<file>] for a .zip/.xlsx/banner file is raw content, not a path or table: open it by PATH "
     "(zipfile.ZipFile(path), ixbrl_main_html(path), read_banner_table(path, header_words) for banner rows above the header)."),
    (r"KeyError: ['\"\[]*'?CIK'?['\"\]]*",
     "That table has no CIK column: in SEC 13F data the filer CIK sits in SUBMISSION; join tables on ACCESSION_NUMBER "
     "(sec13f_filings(sec13f_load()))."),
    (r"first_obs and last_obs produce an empty array",
     "arch_model got an empty or all-NaN series: build returns only where lagged weights exist, assert len > window and "
     "all finite before fitting."),
    (r"between\(\) got an unexpected keyword argument '(neither|both|left|right)'",
     "Series.between(lo, hi, inclusive='{g1}') - inclusive takes the string; or (x > lo) & (x < hi)."),
    (r"KeyError: '(\w+_(?:pct|flag|musd|amount))'",
     "'{g1}' exists only for some rows/types: keep one dict per item with EVERY key (np.nan when not applicable) and "
     "read values from that dict, not from a column that may be absent."),
    (r"severity_dist\('gamma'\): 'beta' is ambiguous",
     "'scale beta = X' means severity_dist('gamma', shape=alpha, scale=X); only 'rate beta' means rate=. Do not retry beta=."),
    (r"truth value of an? (?:Series|array)[\s\S]{0,400}?in implied_vol|implied_vol\([\s\S]{0,200}truth value",
     "qf_toolkit.implied_vol is SCALAR. For a column of prices call implied_vol_vec(price, S, K, T, r, q, kind) "
     "(Series/arrays in, array out, NaN outside the no-arbitrage bounds)."),
    (r"too many indices for array: array is 1-dimensional, but (\d+) were indexed",
     "A 1-D array was indexed with {g1} axes: use x[:, None] for a (T,1) column and check .ndim/.shape before slicing "
     "(for HMM EM call gaussian_hmm_fit)."),
    (r"__new__\(\) got an unexpected keyword argument 'utc'",
     "pd.Timestamp has no utc= keyword: use pd.to_datetime(x, utc=True) or pd.Timestamp(x).tz_convert('UTC')."),
    (r"SeedSequence expects int or sequence of ints for entropy not ([\d.]+)",
     "The seed was read from JSON as a float ({g1}): np.random.default_rng(int(seed))."),
    (r"KeyError: (-\d+)\b",
     "Label {g1} is not in the index: s[-1] is a LABEL lookup in pandas. Use s.iloc[-1] or s.to_numpy()[-1]."),
    (r"Unable to coerce to Series, length must be (\d+): given (\d+)",
     "A pandas row of length {g1} was combined with an array of length {g2}: use .to_numpy() on both sides over the "
     "same ordered column list before arithmetic."),
    (r"could not broadcast input array from shape \((\d+),\) into shape \((\d+),\)",
     "A series of length {g1} was written into an array of length {g2}: price-based series have one more row than "
     "returns - compute on the full price column, then align to return dates with [1:]."),
    (r"'function' object is not subscriptable|'numpy\.ndarray' object is not callable",
     "A name was rebound (an array stored under a function's name, or a function indexed like an array): use "
     "distinct names (V for prices, bs_price for the function)."),
    (r"string indices must be integers",
     "A loop reached a STRING where it expected a dict: JSON objects often carry a '_comment' or a date string next "
     "to the data. Iterate the nested dict, skip keys starting with '_' and values that are not dicts; quotes stored "
     "as [bid, offer] lists unpack with bid, offer = d[pair]."),
    (r"KeyError: '[^']*;[^']*;[^']*'",
     "The file is ';'-separated and was read as ONE column: pd.read_csv(path, sep=';', index_col=0) or "
     "read_delimited_matrix(path)."),
    (r"legs .* do not triangulate to|KeyError: '[A-Z]{3}/[A-Z]{3}'",
     "Triangulate with cross_via_usd(target, {pair: (bid, ask)}): it finds the X/USD and USD/Y legs in either "
     "orientation (an inverted leg is (1/ask, 1/bid))."),
    (r"'(?:Expr|DataFrame|LazyFrame)' object has no attribute '(?:apply|groupby|with_row_count|melt|cumsum|map_dict)'"
     r"|UDF called without return type|PolarsInefficientMapWarning|MapWithoutReturnDtypeWarning",
     "Polars 1.x API: new, todo = polars_0x_to_1x_source(old_src), then rewrite each todo lambda as a native "
     "expression (pl.col(...) arithmetic, .abs(), .cast); never map_elements."),
    (r"KeyError: '\d+(?:\.\d+)?'",
     "That maturity is not a node of the curve: interpolate (zero_fn(t) from bootstrap_par_linear_zero, or "
     "loglinear_interp on discount factors) instead of looking the key up."),
    (r"JSONDecodeError: Extra data: line 1 column \d+",
     "A .jsonl file holds one JSON object per line: [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()] "
     "(INPUTS already loads .jsonl as a table)."),
    (r"KeyError: \"\['value'\] not in index\"|KeyError: 'value'",
     "Print df.columns: long-format vendor files name the value column e.g. field_value (with field_name); use the "
     "exact names, never params.get(guess, default)."),
    (r"module 'numpy' has no attribute 'special'",
     "special functions live in scipy.special (import scipy.special), not numpy."),
    (r"SOFT_ASSERT_FAILED line \d+: .*(?:mismatch|does not sum|don't sum|expected)",
     "Delete self-checks that compare with a value or identity the task does not state; compare floats with "
     "math.isclose, never ==. Do not change the computation to satisfy an invented check."),
    # ---- know73 ----
    (r"You are trying to merge on (\w+) and (\w+) columns",
     "The merge keys have different types ({g1} vs {g2}): normalise BOTH to the same form first - "
     "pd.to_datetime(x).dt.strftime('%Y-%m-%d') for dates (SEC 'DD-MON-YYYY' included), astype(str) for ids."),
    (r"Cannot index with multidimensional key|Unalignable boolean Series",
     "The mask was built on a different index. A Series assigned into a frame with another index becomes all-NaN: "
     "use .to_numpy() or reindex to the same index; for event windows keep the DATE index on both returns."),
    (r"single positional indexer is out-of-bounds|index 0 is out of bounds for axis 0 with size 0",
     "A lookup returned NO rows (.loc[mask].iloc[0] on an empty selection): the key is absent in that form - e.g. the file "
     "quotes USD/JPY, not JPY/USD (invert: 1/ask, 1/bid). Print the unique values of the key column, build a dict "
     "{key: row} and handle the inverted or missing key explicitly."),
    (r"divide\(.*where=|invalid value encountered in divide",
     "np.divide(a, b, where=mask) needs out=np.zeros_like(a, dtype=float) as well, otherwise the masked cells hold garbage."),
    (r"cannot concatenate object of type '<class 'dict'>'",
     "pd.concat takes Series/DataFrames: wrap a dict first (pd.DataFrame([d]) for one row, pd.Series(d) for one column), "
     "or build a list of dicts and call pd.DataFrame(rows) once at the end."),
    (r"level > 0 or level < -1 only valid with MultiIndex",
     "The index is not a MultiIndex: drop the level= argument, or set_index([a, b]) first; after groupby(...).apply use "
     ".reset_index(drop=True)."),
    (r"could not convert string to float: '([^']*)'",
     "Text '{g1}' sits in a numeric column (a header/label row read as data, or a keep='first'-style argument passed "
     "positionally). Read with the right header row, filter label rows out, then pd.to_numeric(..., errors='coerce')."),
    (r"Invalid value '([^']*)' for dtype '(float64|int64)'",
     "The string '{g1}' was written into a numeric column ({g2}): you assigned a column NAME instead of its values "
     "(frame[col] = frame['{g1}'] not frame[col] = '{g1}')."),
    (r"shapes \((\d+),\) and \((\d+),(\d+)\) not aligned",
     "A 1-D vector ({g1}) was multiplied with a ({g2},{g3}) matrix on the wrong side: use M @ v when M is (k, n) and v has n "
     "entries, or transpose M; print shapes before every matrix product."),
    (r"unterminated string literal|unexpected EOF while parsing|expected an indented block",
     "The cell was cut mid-statement: write SHORTER cells (one step each), never a long triple-quoted block in one reply."),
    (r"takes from \d+ to \d+ positional arguments but \d+ were given",
     "Pass the extra values by keyword: toolkit functions take data positionally and every option as keyword "
     "(parametric_var_es(returns, level=..., dist=...))."),
    (r"value_name \([^)]*\) cannot match an element in the DataFrame columns",
     "pd.melt: choose value_name / var_name that are not existing column names (e.g. value_name='val')."),
    (r"Length of values \((\d+)\) does not match length of index \((\d+)\)",
     "Assigning {g1} values to a frame of {g2} rows: align by index (frame.loc[idx, col] = s) or compute the column from the same "
     "rows (after the same filter / dropna) instead of a parallel list."),
    # ---- know78 (per-unit gap analysis of the know77 never-passed lanes) ----
    (r"Object of type (DataFrame|Series|ndarray|Figure|Timestamp) is not JSON serializable",
     "A .json deliverable got an object that holds a {g1}. Usually a whole-task helper's output dict (keyed by file "
     "name) was emitted as ONE json file: emit each entry instead - bundle_files(outs, [file names], emit=emit) or "
     "`for name, obj in outs.items(): emit(name, obj)`. A table inside a json: .to_dict('records')."),
    (r"cannot unpack non-iterable (?:numpy\.)?(float64|float|int) object",
     "The helper returned ONE number ({g1}), not a tuple: read its API line (e.g. svi_raw -> w only; svi_derivatives "
     "-> (w, w_k, w_kk)). If a *_report / *_bundle already holds this file, emit rep['<stem>'] instead of recomputing."),
    (r"Can only use \.dt accessor with datetimelike values",
     "That column already holds strings (report outputs carry ISO 'YYYY-MM-DD' text): use it as is, or "
     "pd.to_datetime(col) before .dt."),
    (r"KeyError: 'filing_id'",
     "No filings were found from that argument: pass the filings DIRECTORY (or None) - e.g. "
     "sec8k_event_alpha_bundle('/app/data/filings', '/app/data/universe.csv', '/app/data/params.json'); the helper "
     "looks one folder down and accepts paths, names or a dict."),
    (r"No such file or directory \(os error 2\)|migrated pipeline failed on every input",
     "A PATH was expected but a parsed object (INPUTS[...], a DataFrame or text) was passed: give file paths or None, "
     "e.g. r = polars_migrate_and_run(None, None, '/app/output'); emit('<new>.py', r['source'])."),
    (r"agent_weights must give a weight for every agent",
     "Map every agent score to its weight from params: agent_weights={'<agent>_score': params['<agent>_weight'], ...}, "
     "or call committee_alpha_auto('/app/data') which reads the agents and weights itself."),
    (r"KeyError: '(quote_time|quote_unit)'",
     "These quote files have no {g1} column - they are already clean: drop the normalize_quote_book call and pass the "
     "loaded frames straight to the curve bundle (dual_curve_deliverables)."),
    (r"cerep_migration_study.*truth value of a DataFrame is ambiguous|truth value of a DataFrame is ambiguous.*cerep",
     "Call the whole study in one line: out = cerep_migration_report({n: INPUTS[n] for n in INPUTS.keys()}); "
     "for name, obj in out.items(): emit(name, obj). A plain list of tables needs years=[...]."),
    (r"'(market|smb|hml|mkt\w*)_beta_residual'",
     "alpha_signal_backtest returns <factor>_beta_residual under the factor names AND market_/smb_/hml_beta_residual; "
     "or write every file with for name, obj in json_deliverables(o).items(): emit(name, obj)."),
    (r"committee_process_alpha_bundle: pass \[",
     "Pass params=json-loaded params file (its keys fill these settings) or the named keywords - the task states them."),
    # ---- know79 group 2 (selftest79_2.py) ----
    (r"in localvol_mc_barrier|localvol_mc_barrier\(\) (?:missing|got)|localvol_mc_barrier: ",
     "Do not hand-wire the barrier MC: rep2 = localvol_barrier_report(raw_snapshot, task_rules, barrier_contract, "
     "snapshot_meta) prices it from the raw snapshot and returns barrier_results / summary in the contract keys - "
     "emit('barrier_results.json', rep2['barrier_results']); emit('summary.json', rep2['summary']). The engine "
     "itself takes local_vol as a callable or the local-vol table, and barrier= (or H=)."),
    (r"svi_dupire_report: every quote carries the one type label",
     "Call svi_dupire_report on the option quotes table exactly as loaded (INPUTS[...] / read_table, no parity "
     "conversion, no relabelling, no column subset): it filters, converts and labels itself."),
]

_COMPILED = [(re.compile(p, re.I), fix) for p, fix in RULES]
_COLS = re.compile(r"columns=\[([^\]]*)\]")


def _columns_in_state(state_text: str) -> list[str]:
    cols = []
    for m in _COLS.finditer(state_text or ""):
        cols.extend(c.strip().strip("'\"") for c in m.group(1).split(","))
    return [c for c in cols if c]


def hint(error_text: str, state_text: str = "") -> str:
    """The fix lines for every known pattern in `error_text` ('' when none matches)."""
    text = error_text or ""
    out = []
    for pat, fix in _COMPILED:
        m = pat.search(text)
        if not m:
            continue
        groups = {f"g{i}": (g or "") for i, g in enumerate(m.groups(), 1)}
        near, where = "", ""
        if "{near}" in fix:
            cols = _columns_in_state(state_text)
            close = difflib.get_close_matches(groups.get("g1", ""), cols, n=3, cutoff=0.5) if cols else []
            near = f" (closest existing: {', '.join(close)})" if close else ""
        if "{where}" in fix:
            try:
                from .genrt import prep
                attr = groups.get("g2") or groups.get("g1") or ""
                found = prep.where_is(attr) if attr else ""
                where = f" (it exists as {found})" if found else ""
            except Exception:  # noqa: BLE001
                where = ""
        try:
            line = fix.format(**groups, near=near, where=where)
        except (KeyError, IndexError):
            line = fix
        if line not in out:
            out.append(line)
        if len(out) >= 3:
            break
    # know80 (helpers80_4): a missing COLUMN gets a per-table report - which tables have it, which name the
    # same role differently (with the rename), which are static (know79 multimodal: 3 identical
    # KeyError: 'asof_date' cells; rebalance / bulletin tables name their date column differently)
    if out and re.search(r"KeyError: ['\"]|None of \[Index", text):
        try:
            from qf_toolkit.helpers80_4 import keyerror_column_report
            report = keyerror_column_report(text, state_text)
        except Exception:  # noqa: BLE001 - a hint must never raise
            report = ""
        if report:
            out.append("PER TABLE:\n  " + report.replace("\n", "\n  "))
    return ("KNOWN ERROR - APPLY THIS FIX:\n- " + "\n- ".join(out)) if out else ""
