"""A persistent Python kernel for cell-by-cell solving (6 Oct).

The model's program arrives as cells (`# %% title` markers). Each cell runs here, in ONE
namespace that survives between requests, so a reply the 4,000-token limit cuts loses nothing:
the complete cells have already run and their variables are cached in this process. After every
cell the kernel reports the new or changed variables (type, shape, columns, head) - the "runtime
state" the model reads before it writes the next cell, instead of guessing shapes.

Protocol (one JSON object per line on stdin / stdout):
  {"cmd": "run", "code": "..."}  -> {"ok": bool, "stdout": str, "error": str, "vars": [str], "soft": [str]}
  {"cmd": "vars"}                -> {"vars": [str]}
  {"cmd": "exit"}
Asserts are recorded, not raised (see soft_run.py); INPUTS and emit are predefined there too.
"""
import ast
import io
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import soft_run  # noqa: E402

SKIP_TYPES = ("module", "function", "type", "builtin_function_or_method", "method")


def describe(value) -> str:
    try:
        name = type(value).__name__
        if hasattr(value, "columns") and hasattr(value, "index"):
            cols = list(value.columns)[:30]
            head = value.head(2).to_string(max_cols=12)[:300].replace("\n", " | ")
            dtypes = {str(c): str(t) for c, t in list(value.dtypes.items())[:12]}
            notes = value.attrs.get("prep_notes") if hasattr(value, "attrs") else None
            return (f"{name} shape={tuple(value.shape)} columns={cols!r} dtypes={dtypes} head: {head}"
                    + (f" | prepared: {'; '.join(notes)[:200]}" if notes else ""))
        shape = getattr(value, "shape", None)
        if shape is not None and not callable(shape):
            flat = repr(value)[:120].replace("\n", " ")
            return f"{name} shape={tuple(shape)} dtype={getattr(value, 'dtype', None)} {flat}"
        if isinstance(value, dict):
            return f"dict len={len(value)} keys={list(value)[:30]!r}"
        if isinstance(value, (list, tuple, set)):
            return f"{name} len={len(value)} {repr(value)[:120]}"
        if isinstance(value, (int, float, str, bool)) or value is None:
            return f"{name} = {repr(value)[:120]}"
        return f"{name} {repr(value)[:100]}"
    except Exception:  # noqa: BLE001
        return "<unprintable>"


def snapshot(ns: dict) -> dict:
    out = {}
    for k, v in ns.items():
        if k.startswith("_") or type(v).__name__ in SKIP_TYPES or k in ("INPUTS", "emit"):
            continue
        try:
            out[k] = (id(v), describe(v))
        except Exception:  # noqa: BLE001
            pass
    return out


def _autofix_once(exc, code_text: str, ns: dict, fixes: list) -> bool:
    """Try one Python-side patch of the cell for this exception; True when a patched version
    should be run (recorded in `fixes` as (note, new_code)). At most 2 patches per cell."""
    if len(fixes) >= 2:
        return False
    try:
        import autofix
    except Exception:  # noqa: BLE001
        return False
    err = f"{type(exc).__name__}: {exc}"
    state_text = "\n".join(f"  {k}: {d}" for k, (_i, d) in snapshot(ns).items())
    new, note = autofix.apply(code_text, err, state_text)
    if not note or new == code_text:
        return False
    fixes.append((note, new))
    return True


def main() -> int:
    try:
        paths = json.loads(os.environ.get("T1_INPUT_FILES") or "[]")
    except ValueError:
        paths = []
    ns = {"__name__": "__main__", "__builtins__": __builtins__,
          "__soft_assert_record__": soft_run._record, "__t1_rows__": soft_run._rows,
          "INPUTS": soft_run._Inputs(paths), "emit": soft_run._emit}
    # The session opens with the usual names bound (know39: 4 cells died on `read_table` /
    # `float64` not defined - a cell that forgot an import the model had in mind).
    try:
        exec("import numpy as np\nimport pandas as pd\nimport math, json\n"
             "from qf_toolkit import *\n", ns)
    except Exception:  # noqa: BLE001
        pass
    try:
        import prep
        prep.install(ns)      # resolving readers, lenient json.loads / toolkit, common names
    except Exception:  # noqa: BLE001
        prep = None
    try:
        import autofix
        autofix.install(ns)   # _fmt_date, _scalar, _as_int used by the auto-fix rewrites
    except Exception:  # noqa: BLE001
        pass
    before = snapshot(ns)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        cmd = msg.get("cmd")
        if cmd == "exit":
            break
        if cmd == "vars":
            print(json.dumps({"vars": [f"{k}: {d}" for k, (_i, d) in snapshot(ns).items()][:40]}), flush=True)
            continue
        if cmd != "run":
            continue
        soft_run._FAILED.clear()
        try:
            soft_run._EMITTED_CELL.clear()
        except AttributeError:
            pass
        buf = io.StringIO()
        real_stdout, real_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = buf
        ok, error = True, ""
        resolved = []
        code_text = msg.get("code", "")
        fixes: list[str] = []
        pre_notes: list[str] = []
        try:
            import autofix
            code_text, pre_notes = autofix.prerun(code_text)   # know78: guarded convention-override drops
        except Exception:  # noqa: BLE001
            pass
        try:
            for _attempt in range(6):
                tree = soft_run._Soften().visit(ast.parse(code_text, filename="<cell>"))
                ast.fix_missing_locations(tree)
                code_obj = compile(tree, "<cell>", "exec")
                try:
                    # know89: the cell's source is readable by line (a positional read of a toolkit result is
                    # resolved by the receiving variable's name on that line)
                    import linecache
                    linecache.cache["<cell>"] = (len(code_text), None, code_text.splitlines(True), "<cell>")
                except Exception:  # noqa: BLE001
                    pass
                try:
                    exec(code_obj, ns)
                    break
                except NameError as exc:
                    # a name the model remembers but did not import: bind it and run the cell again
                    # (the cell has not emitted anything yet when it dies on a name)
                    name = getattr(exc, "name", None) or (str(exc).split("'")[1] if "'" in str(exc) else "")
                    value, where = (prep.resolve_name(name) if (prep and name) else (None, ""))
                    if value is not None and name not in ns:
                        ns[name] = value
                        resolved.append(f"{name} <- {where}")
                        buf.seek(0); buf.truncate()
                        continue
                    if not _autofix_once(exc, code_text, ns, fixes):
                        raise
                    code_text = fixes[-1][1]
                    buf.seek(0); buf.truncate()
                except Exception as exc:  # noqa: BLE001 - 9 Oct: Python patches the usual crashes itself
                    if not _autofix_once(exc, code_text, ns, fixes):
                        raise
                    code_text = fixes[-1][1]
                    buf.seek(0); buf.truncate()
        except BaseException as exc:  # noqa: BLE001 - every failure is reported, never fatal
            ok = False
            tb = traceback.format_exc()
            error = f"{type(exc).__name__}: {exc}\n--- traceback tail ---\n{tb[-1200:]}"
        finally:
            sys.stdout, sys.stderr = real_stdout, real_stderr
        # know94: the program's OWN checks that failed in a cell that emitted deliverables belong to those
        # deliverables (a double-sort run asserted net = gross - cost x turnover, the assertion failed - it had
        # passed cost 0 - and the cell was accepted and `done` wrote the wrong file); they join the emit verdicts
        # the cells loop will not accept `done` over
        try:
            own = [m for m in soft_run._FAILED if "emit(" not in m]
            if own:
                for base in dict.fromkeys(getattr(soft_run, "_EMITTED_CELL", [])):
                    if getattr(soft_run, "_EMIT_FROM_BUNDLE", {}).get(base):
                        continue          # know95: a bundle's own object - the failed check is the suspect
                    soft_run._EMIT_FLAGS[base] = list(soft_run._EMIT_FLAGS.get(base, [])) + \
                        [f"{m} (the program's own check, in the cell that emitted it)" for m in own[:2]]
        except Exception:  # noqa: BLE001 - an aid, never a failure cause
            pass
        notes = (prep.take_notes() if prep else []) + [f"resolved name {r}" for r in resolved]
        notes += [f"auto-fixed: {n}" for n, _c in fixes] + [f"pre-run: {n}" for n in pre_notes]
        notes += [ln.strip()[:300] for ln in buf.getvalue().splitlines() if ln.startswith("CONTRACT_NOTE")][:4]
        after = snapshot(ns)
        changed = [f"{k}: {d}" for k, (i, d) in after.items() if before.get(k, (None, None))[0] != i
                   or before.get(k, (None, None))[1] != d]
        before = after
        print(json.dumps({"ok": ok, "stdout": buf.getvalue()[-1500:], "error": error,
                          "vars": changed[:30], "soft": list(soft_run._FAILED)[:10],
                          "notes": notes[:8],
                          "emit_flags": {k: v[:2] for k, v in getattr(soft_run, "_EMIT_FLAGS", {}).items() if v}}),
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
