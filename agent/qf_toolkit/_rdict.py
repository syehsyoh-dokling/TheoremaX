"""know77: result dicts that explain themselves.

know76 lost whole request budgets to KeyErrors on guessed helper keys ('beta', 'hw_a', 'a',
'annualized_volatility', 'se' ...). A ResultDict behaves exactly like a dict (json.dump, ** unpacking,
isinstance(x, dict)), but a missing key
  1. resolves a declared ALIAS (e.g. 'annualized_volatility' -> 'annualized_vol'),
  2. else, for a bundle {file name: obj}, finds the key inside exactly one nested dict and returns it,
  3. else raises KeyError naming the real keys and the closest ones.
Private module (not exported); helpers import it lazily.

know89 (flip census: the model's RANDOM slips around helper results):
  * a positional read `x = res[1]` / `float(res[0])` of a result (a dict, not a tuple) returns the field the
    receiving variable NAMES (exact key or one unambiguous near-miss); otherwise TypeError listing the keys -
    never the value that happens to sit at that position (know77-88 returned list(values)[i] silently);
  * a tuple unpacking `a, b, c = res` assigns the fields the target names name (all or nothing); otherwise
    TypeError listing the keys (a plain dict hands out its KEY STRINGS);
  * a near-miss resolution (head / tail / nested match, not a declared alias) is announced on stderr.
  Exact keys, aliases, extras, `in`, len, iteration in a for loop, json.dump: unchanged.
"""
from __future__ import annotations

import difflib
import re
import sys

_UNPACK = None
_RECENT: list = []          # know89 guard F: weak references to the latest toolkit results (newest last)


def register(obj):
    """Remember a toolkit result (weakly) so emit can restore a contract key a reshaped copy dropped."""
    try:
        import weakref
        _RECENT.append(weakref.ref(obj))
        if len(_RECENT) > 64:
            del _RECENT[:-64]
    except TypeError:
        pass
    return obj


def recent_results() -> list:
    return [o for o in (r() for r in reversed(_RECENT)) if o is not None]
_STORES = ("STORE_NAME", "STORE_FAST", "STORE_GLOBAL", "STORE_DEREF")
_CAST = r"(?:float|int|bool|str|np\.float64|np\.asarray|np\.array|abs|round)"


def _note(msg: str) -> None:
    try:
        print(f"[resultdict] {msg}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - a note, never a failure cause
        pass


def _store_names(frame, n: int) -> list:
    """Names of the next n STORE instructions after the frame's current one ([] when the value is not stored
    straight into plain names)."""
    import dis
    out, started = [], False
    for inst in dis.get_instructions(frame.f_code):
        if inst.offset <= frame.f_lasti:
            continue
        if inst.opname in _STORES:
            out.append(str(inst.argval))
            started = True
            if len(out) == n:
                return out
            continue
        if not started and inst.opname in ("CACHE", "EXTENDED_ARG", "NOP"):
            continue
        return []
    return []


def _line_targets(frame, index: int) -> list:
    """`name = res[i]` / `name = float(res[i])` on the caller's source line -> [name] (the source of a kernel cell
    is registered in linecache by the runtime)."""
    try:
        import linecache
        ln = linecache.getline(frame.f_code.co_filename, frame.f_lineno)
    except Exception:  # noqa: BLE001
        return []
    m = re.match(r"^\s*([A-Za-z_]\w*)\s*(?::[^=]+)?=\s*(?:" + _CAST + r"\(\s*)*[A-Za-z_][\w.]*(?:\[[^\[\]]*\])*"
                 r"\[\s*-?" + str(abs(int(index))) + r"\s*\]", ln or "")
    return [m.group(1)] if m else []


class ResultDict(dict):
    """dict with alias resolution, unique nested lookup and an informative KeyError."""

    _aliases: dict = {}
    _extras: dict = {}

    def __init__(self, *args, aliases=None, extras=None, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_aliases", dict(aliases or {}))
        # know87 (round 2): values a caller may reach by subscript (a full matrix next to its reported
        # elements, e.g. res['ewma_cov'][i, j]) that are NOT items - json.dump / emit never write them
        object.__setattr__(self, "_extras", dict(extras or {}))
        register(self)

    # ------------------------------------------------------------------ know89: names, not positions
    def _resolve_name(self, name):
        """A variable name -> the real key it names: exact, alias, or one unambiguous near-miss; None."""
        if not isinstance(name, str):
            return None
        if dict.__contains__(self, name):
            return name
        if name in self._aliases and dict.__contains__(self, self._aliases[name]):
            return self._aliases[name]
        low = name.lower().strip("_")
        keys = [k for k in self.keys() if isinstance(k, str)]
        for k in keys:
            if k.lower() == low:
                return k
        own = [k for k in keys if k.lower().endswith("_" + low) or k.lower().startswith(low + "_")]
        if len(own) == 1:
            return own[0]
        part = [k for k in keys if low.endswith("_" + k.lower()) or low.startswith(k.lower() + "_")]
        if len(part) == 1:
            return part[0]
        return None

    def _positional(self, key):
        names = []
        try:
            f = sys._getframe(2)
            names = _store_names(f, 1) or _line_targets(f, key)
        except Exception:  # noqa: BLE001
            names = []
        real = self._resolve_name(names[0]) if names else None
        if real is not None:
            _note(f"positional read [{key}] of a dict result: returned [{real!r}] (the name {names[0]!r} asks for it)")
            return dict.__getitem__(self, real)
        raise TypeError(f"res[{key}]: this toolkit result is a DICT, not a tuple - read it by NAME. "
                        f"Keys: {[str(k) for k in self.keys()][:40]}")

    def __iter__(self):
        try:
            global _UNPACK
            if _UNPACK is None:
                import dis
                _UNPACK = dis.opmap.get("UNPACK_SEQUENCE", -1)
            f = sys._getframe(1)
            code = f.f_code.co_code
            if 0 <= f.f_lasti < len(code) and code[f.f_lasti] == _UNPACK:
                n = code[f.f_lasti + 1]
                names = _store_names(f, n)
                real = [self._resolve_name(nm) for nm in names]
                if len(real) == n and all(r is not None for r in real) and len(set(real)) == n:
                    _note(f"tuple unpacking of a dict result: {names} read by name as {real}")
                    return iter([dict.__getitem__(self, r) for r in real])
                raise TypeError(f"unpacking {n} names from a toolkit result: it is a DICT, not a tuple - read it by "
                                f"NAME (res['key']). Keys: {[str(k) for k in self.keys()][:40]}")
        except TypeError:
            raise
        except Exception:  # noqa: BLE001 - detection is an aid; plain iteration otherwise
            pass
        return dict.__iter__(self)

    def __missing__(self, key):
        if key in self._aliases and self._aliases[key] in self:
            return dict.__getitem__(self, self._aliases[key])
        if isinstance(key, str) and key in self._extras:
            return self._extras[key]
        if isinstance(key, int) and not isinstance(key, bool) and -len(self) <= key < len(self):
            return self._positional(key)
        hits = [(name, sub) for name, sub in self.items() if isinstance(sub, dict) and key in sub]
        if len(hits) == 1:
            return hits[0][1][key]
        # know82: a short key that is the unique tail / head of a real key ('mean_ou' -> 'stationary_mean_ou',
        # 'portfolio_daily_vol' inside 'results.json') resolves to it; ambiguous matches still raise
        sk = str(key)
        for alt in (sk[:-4] if sk.endswith("_ols") else None,
                    "ann_" + sk[len("annualized_"):] if sk.startswith("annualized_") else None,
                    "annualized_" + sk[len("ann_"):] if sk.startswith("ann_") else None):
            if alt and dict.__contains__(self, alt):
                _note(f"[{sk!r}] read as [{alt!r}]")
                return dict.__getitem__(self, alt)
        own = [k for k in self.keys() if isinstance(k, str) and k != sk and (k.endswith("_" + sk) or k.startswith(sk + "_"))]
        if len(own) == 1:
            _note(f"[{sk!r}] read as [{own[0]!r}]")
            return dict.__getitem__(self, own[0])
        deep = [(n, k) for n, sub in self.items() if isinstance(sub, dict)
                for k in sub if isinstance(k, str) and (k == sk or k.endswith("_" + sk))]
        if len(deep) == 1:
            _note(f"[{sk!r}] read as [{deep[0][0]!r}][{deep[0][1]!r}]")
            return self[deep[0][0]][deep[0][1]]
        keys = [str(k) for k in self.keys()]
        close = difflib.get_close_matches(str(key), keys, n=4, cutoff=0.5)
        nested = sorted({f"[{n!r}][{k!r}]" for n, sub in self.items() if isinstance(sub, dict)
                         for k in sub if difflib.SequenceMatcher(None, str(key), str(k)).ratio() > 0.6})[:4]
        raise KeyError(f"{key!r} is not a key of this toolkit result. Keys: {keys[:40]}; closest: {close}"
                       + (f"; nested candidates: {nested}" if nested else "")
                       + ". Copy a listed key (print(sorted(res)) once) instead of guessing.")

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, TypeError):
            return default

    def __contains__(self, key):
        return dict.__contains__(self, key) or (key in self._aliases and dict.__contains__(self, self._aliases[key]))


def rdict(d, aliases=None):
    """Wrap a plain dict (shallow) as a ResultDict."""
    return ResultDict(d, aliases=aliases)


def deep_rdict(obj, _seen=None, _depth=0):
    """know89: a whole-task helper's plain-dict result -> ResultDict, nested plain dicts too (the same inner
    object stays ONE object). Non-dict values, and dicts that already are a dict subclass, are kept as they are
    (their nested plain dicts are still wrapped in place of the originals only when the parent is rebuilt here)."""
    if _seen is None:
        _seen = {}
    if type(obj) is not dict or _depth > 4:
        return obj
    if id(obj) in _seen:
        return _seen[id(obj)]
    new = ResultDict()
    _seen[id(obj)] = new
    for k, v in obj.items():
        dict.__setitem__(new, k, deep_rdict(v, _seen, _depth + 1))
    return new
