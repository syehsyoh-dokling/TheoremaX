"""Duplicate (date, identifier) rows in the residual-momentum input panel (data only)."""
import glob

import pandas as pd

for f in glob.glob("/data/*.parquet") + glob.glob("/data/*.pqt"):
    df = pd.read_parquet(f)
    cols = [c for c in ("date", "identifier") if c in df.columns]
    print(f, df.shape, cols, "duplicate key rows:", int(df.duplicated(cols).sum()) if len(cols) == 2 else "n/a",
          "| full-row duplicates:", int(df.duplicated().sum()))
