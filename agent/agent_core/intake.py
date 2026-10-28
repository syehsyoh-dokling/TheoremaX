"""INTAKE: what Python can take out of the task before the model sees anything (8 Oct, lapis tengah).

`config_flatten` turns every JSON input into flat, named variables (`signal_lag`,
`quotes.EUR/CHF.tenor_6M`, `weights` as a list) so no key is guessed: 6 of the 62 failing units died
on KeyError / NoneType / float-vs-vector reads of a nested config. `task_graph` lists the task's
rule sentences (must / only / exactly / before / after), its parameters with units, its dates and
any example JSON shown, so the brief can quote the exact sentence next to the cell that needs it.
Offline, deterministic, never reads checks/.
"""
from __future__ import annotations

import json
import pathlib
import re
from typing import Any

RULE_WORDS = re.compile(r"\b(must|only|exactly|never|before|after|in this order|at least|at most|"
                        r"non-negative|positive|rounded?|decimal places|iff|if and only if)\b", re.I)
PARAM = re.compile(r"(?P<name>[A-Za-z][A-Za-z0-9_ ]{1,40}?)\s*(?:=|:|of|is|at)\s*(?P<num>-?\d+(?:\.\d+)?)\s*"
                   r"(?P<unit>%|bps|bp|basis points|years?|days?|months?|business days|trading days|paths|"
                   r"simulations|draws|steps|dp)?", re.I)
DATE = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")


def config_flatten(obj: Any, prefix: str = "", out: dict | None = None, depth: int = 0) -> dict:
    """{dotted.path: value}; lists of scalars stay lists (a weights vector), lists of dicts are
    indexed (`obligors[0].pd`), scalars at the top keep their own name."""
    out = {} if out is None else out
    if depth > 6:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict) or (isinstance(v, list) and v and isinstance(v[0], (dict, list))):
                out[key] = v if isinstance(v, list) else {"__keys__": list(v.keys())[:40]}
                config_flatten(v, key, out, depth + 1)
            else:
                out[key] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:50]):
            config_flatten(v, f"{prefix}[{i}]", out, depth + 1)
    return out


def load_configs(input_root: str | pathlib.Path, data_files) -> tuple[dict, list[str]]:
    """Every JSON input flattened into one namespace: short keys first (top-level names), dotted
    paths for the rest; a clash keeps the first file's value and is reported."""
    cfg: dict = {}
    notes: list[str] = []
    root = pathlib.Path(input_root)
    for rel in data_files:
        p = root / rel
        if p.suffix.lower() != ".json" or not p.is_file() or p.name == "manifest.json":
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except ValueError as exc:
            notes.append(f"{rel}: not JSON ({exc})")
            continue
        flat = config_flatten(obj)
        stem = p.stem
        for k, v in flat.items():
            if k in cfg and cfg[k] != v:
                cfg[f"{stem}.{k}"] = v
                notes.append(f"key {k!r} in several files; see {stem}.{k}")
            else:
                cfg.setdefault(k, v)
        cfg.setdefault(stem, obj)
    return cfg, notes


def cfg_listing(cfg: dict, limit: int = 60) -> str:
    """The brief's view of CFG: one line per scalar/list key, nested dicts as their key lists."""
    lines = []
    for k, v in list(cfg.items())[:limit]:
        if isinstance(v, dict):
            keys = v.get("__keys__") if "__keys__" in v else list(v.keys())[:12]
            lines.append(f"  CFG[{k!r}]: dict with keys {keys}")
        elif isinstance(v, list):
            head = v[:6]
            lines.append(f"  CFG[{k!r}]: list len={len(v)} {head}{' ...' if len(v) > 6 else ''}")
        else:
            lines.append(f"  CFG[{k!r}] = {v!r}"[:160])
    if len(cfg) > limit:
        lines.append(f"  ... {len(cfg) - limit} more keys (print(sorted(CFG)) to see them)")
    return "\n".join(lines)


def task_graph(instruction: str) -> dict:
    """rules, params, dates, example JSON blocks from the instruction text."""
    text = re.sub(r"<!--.*?-->", " ", instruction or "", flags=re.S)   # know71: no comment/canary lines as rules
    sentences = [s.strip() for s in re.split(r"(?<=[.;:\n])\s+", text) if s.strip()]
    rules = [s[:300] for s in sentences if RULE_WORDS.search(s) and not s.lstrip().startswith("<!--")][:60]
    params = []
    for m in PARAM.finditer(text):
        name = m.group("name").strip().lower()
        if len(name) < 3 or name in ("the", "and", "with", "for", "from"):
            continue
        params.append({"name": name[:40], "value": float(m.group("num")), "unit": (m.group("unit") or "").lower()})
    dates = sorted(set(m.group(0) for m in DATE.finditer(text)))[:30]
    examples = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)[:6]
    return {"rules": rules, "params": params[:80], "dates": dates, "examples": [e[:1200] for e in examples]}


#: know71: Monte Carlo / randomness settings a task states. A result off by MC noise fails a
#: tolerance check, so the seed, path count and step count the TASK gives are enforced verbatim.
_MC_KEYS = (
    ("seed", r"(?:random[ _]?)?seed|random_state|rng[ _]seed"),
    ("n_paths", r"n_paths|num_paths|n_sims|n_simulations|num_simulations|n_mc_paths|mc_paths|"
                r"(?:monte carlo )?(?:paths|simulations|scenarios|draws|samples)"),
    ("n_steps", r"n_steps|num_steps|time[ _]steps|steps(?: per path)?"),
)


def mc_settings(instruction: str, cfg: dict | None = None) -> dict:
    """{'seed': 42, 'n_paths': 50000, ...} from the task text ('seed 42', 'seed=42', '50,000 paths',
    'n_paths = 50000') and from flattened config keys; the config wins on a conflict."""
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", (instruction or "").replace("**", "").replace("`", ""))
    out: dict = {}
    for key, pat in _MC_KEYS:
        found = []
        # "<key> = 42", "<key>: 42", "seed(42)", "Number of samples: N = 100000"
        for m in re.finditer(r"\b(?:" + pat + r")\b\s*(?:=|:|of|is|to|\()?\s*(?:[A-Za-z]\s*=\s*)?(\d{1,9})\b", text, re.I):
            found.append(int(m.group(1)))
        # "100000 OU-jump paths", "50000 Monte Carlo simulations" (up to two words in between)
        if key != "seed":
            for m in re.finditer(r"\b(\d{2,9})\s+(?:[A-Za-z][\w\-]*\s+){0,2}?(?:" + pat + r")\b", text, re.I):
                found.append(int(m.group(1)))
            found = [v for v in found if v >= 10]
        if found:
            # the seed is stated once; a path / step count is the LARGEST stated (a "first 1000 rows"
            # preview is smaller than the simulation size)
            out[key] = found[0] if key == "seed" else max(found)
    for k, v in (cfg or {}).items():
        leaf = str(k).split(".")[-1].lower()
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        for key, pat in _MC_KEYS:
            if re.fullmatch(pat, leaf, re.I) or (key == "seed" and "seed" in leaf):
                out[key] = int(v)
    return out


def mc_block(instruction: str, cfg: dict | None = None) -> str:
    s = mc_settings(instruction, cfg)
    if not s:
        return ""
    return ("MONTE CARLO SETTINGS STATED BY THE TASK (use exactly these values - never fewer paths "
            "or steps to save time; seed the generator the task names, else np.random.default_rng(seed), once): "
            + ", ".join(f"{k}={v}" for k, v in s.items()))


def rules_for(graph: dict, names: list[str], limit: int = 8) -> list[str]:
    """Rule sentences that mention any of `names` (a deliverable, a column, a quantity)."""
    low = [n.lower() for n in names if n and len(n) >= 3]
    out = []
    for s in graph.get("rules", []):
        sl = s.lower()
        if any(n in sl for n in low):
            out.append(s)
        if len(out) >= limit:
            break
    return out
