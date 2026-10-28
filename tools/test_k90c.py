"""know90c checks: the object-repr guard flags a str(RangeIndex) date column and passes real dates; the
duplicate-column hint fires for "'DataFrame' object has no attribute 'dt'"."""
import sys

sys.path.insert(0, "/ctx_rt")
sys.path.insert(0, "/agent")
import pandas as pd
import soft_run
from agent_core import known_errors

soft_run._FAILED.clear()
bad = pd.DataFrame({"date": [str(pd.RangeIndex(3))] * 3, "x": [1.0, 2.0, 3.0]})
soft_run._guard_sanity("decomposition_results.csv", bad)
assert any("PRINTED FORM" in f for f in soft_run._FAILED), soft_run._FAILED
soft_run._FAILED.clear()
good = pd.DataFrame({"date": ["2006-12-01", "2007-01-01", "2007-02-01"], "name": ["Index fund", "a", "b"],
                     "x": [1.0, 2.0, 3.0]})
soft_run._guard_sanity("decomposition_results.csv", good)
assert not soft_run._FAILED, soft_run._FAILED
h = known_errors.hint("AttributeError: 'DataFrame' object has no attribute 'dt'. Did you mean: 'at'?")
assert "TWO columns" in h and ".dt works" in h, h
print("K90C OK")
