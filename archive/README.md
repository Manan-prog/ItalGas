# archive/

Code of experiments that did not become part of the working chain (moved here 9 Nov 2026, unchanged, unmaintained).
Every experiment is logged in `../EXPERIMENTS.md` with its config and metric, and the decisions it led to are in
`../DECISIONS.md`; this folder keeps those entries reproducible.

- `pipeline/`  experiment modules (injection tests, learned emission, Viterbi tracking, polarisation, end-point detector,
  junction pins, apex agreement, multi-pass registration, town-plan overlay, ...). They import from `../pipeline/`;
  run them with both folders on `sys.path`, from this folder: `PYTHONPATH=../pipeline python <script> ...`.
  Paths inside are the ones of the original machine (`/Users/.../ItalGas`); they were not converted to `paths.py`.
- `configs/`   their YAML configs.
- `figures/`   their figure scripts and PNGs (the INTERNAL town-plan overlay is excluded from git).
- `web/`       the first two web pages (three-street page, city page template v1).
- `tests/`     unit tests of archived modules (run with `archive/pipeline` on `sys.path`).
