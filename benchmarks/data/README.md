# Backend Benchmark Systems

This directory contains the 10 CIF inputs used by
`benchmarks/backend_startup_perstep.py`.

They were extracted from the tier2 CSP parquet set using the same selection
that the benchmark used before these fixtures were packaged:

1. Read `<MOL>.parquet`.
2. Filter `converged == True` and `n_atoms == N`.
3. Select `sample(n=1, random_state=42)`.
4. Save the row's `cif` text as `systems/<MOL>_<N>.cif`.

`systems_manifest.json` records the selected systems and source row counts.
The benchmark uses these packaged CIFs by default. Pass `--parquet-dir` to
sample from the original parquet files instead.
