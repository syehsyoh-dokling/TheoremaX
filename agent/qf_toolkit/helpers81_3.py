"""know81 group 3 (regime portfolios / factor models): one engine aid for template JSON deliverables.

Task pattern: "report helper -> intermediates already shaped {name: {'value': x}} -> a later cell re-types the
template block as {name: {'value': intermediates[name]}}". The know77 and know79 riskparity programs did exactly
that and wrote {"value": {"value": x}} for every intermediate (a WRONG verdict no number can fix). The instruction's
own template ("num_rebalances": {"value": <int>}) puts a SCALAR under "value"; a dict there is never meant.
collapse_value_nesting(obj) lifts every {"value": {"value": x, ...}} back to {"value": x, ...} (keys of the inner
dict such as 'note' are kept when the outer dict lacks them). The engine's emit() applies it to every JSON dict.

No task names, no data values. Self-tests: proven_work/fail72/selftest81_3.py.

API lines (one per function):
     collapse_value_nesting(obj) -> obj with {"value": {"value": x}} lifted to {"value": x} at any depth (engine aid; emit applies it)
"""
from __future__ import annotations

_ENGINE_ONLY = ("collapse_value_nesting",)


def _lift(node, path, fixed):
    if isinstance(node, dict):
        while isinstance(node.get("value"), dict) and "value" in node["value"]:
            inner = node["value"]
            node["value"] = inner["value"]
            for k, v in inner.items():
                if k != "value" and k not in node:
                    node[k] = v
            fixed.append(".".join(path) or "<root>")
        for k in list(node):
            _lift(node[k], path + [str(k)], fixed)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _lift(v, path + [str(i)], fixed)


def collapse_value_nesting(obj):
    """obj (a dict or list about to be written as JSON) with every {"value": {"value": x, ...}} collapsed to
    {"value": x, ...}, in place on a copy; prints one NOTE naming the keys it fixed. Anything else is unchanged."""
    import copy
    try:
        if not isinstance(obj, (dict, list)):
            return obj
        out = copy.deepcopy(obj)
        fixed = []
        _lift(out, [], fixed)
        if not fixed:
            return obj
        print(f"NOTE: collapsed a doubled {{'value': {{'value': x}}}} at {fixed[:8]} - the template puts the scalar "
              "under 'value'", flush=True)
        return out
    except Exception:  # noqa: BLE001 - an aid, never a failure cause
        return obj
