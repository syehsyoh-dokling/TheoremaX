method: polars-0x-to-1x-migration
family: var-backtesting-and-data-prep
keywords: polars migration, polars 1.x, polars 0.x, old api to new api, groupby to group_by, with_column, map_elements, return_dtype, polarsinefficientmapwarning, mapwithoutreturndtypewarning, deprecationwarning, lazy and eager
---
## Formulas
1. Renames 0.x -> 1.x: groupby -> group_by; groupby_dynamic -> group_by_dynamic; groupby_rolling -> rolling; with_column -> with_columns; apply -> map_elements(return_dtype=); map -> map_batches; cumsum etc -> cum_sum; pl.count() -> pl.len(); take -> gather; frame_equal -> equals; melt -> unpivot; pivot(columns=) -> pivot(on=); with_row_count -> with_row_index; str.lengths -> str.len_chars; str.strptime -> str.to_datetime; map_dict -> replace; sort(reverse=) -> descending=; min_periods -> min_samples; join 'outer' -> 'full'; read_csv dtypes= -> schema_overrides=.
2. Replace Python lambdas by native expressions (when/then, str, dt) to avoid map warnings.
## Variants to decide
- V1 column_names: changed default names. Options: (a) alias to the OLD names (count, row_nr); (b) keep new defaults (len, index). Default when silent: (a) - "preserve all outputs".
- V2 group_order: group_by order. Options: (a) maintain_order=True or explicit sort; (b) unordered. Default when silent: (a) - deterministic outputs.
- V3 translation_scope: migration. Options: (a) translate every statement, keep every write; (b) rewrite freely. Default when silent: (a) - Polars 1.0 upgrade guide.
## Invariants
- Same output files, columns, rows and dtypes as the old script; no warnings under warnings-as-errors.
## Pitfalls
- Inventing a new pipeline instead of translating the given file.
- Not writing the deliverable .py itself; hardcoded results.
- Emitting a function object or translating one statement; use polars_0x_to_1x_source, then hand-translate each lambda to native.
- The rewriter returns the source str itself (also unpacks to src, todo).
## Sources
- Polars 1.0 upgrade guide; Polars API reference (deprecations).
