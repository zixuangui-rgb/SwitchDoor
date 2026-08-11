# Repository invariants

- `src/switchdoor/core.py` is the sole executable source of transition truth.
- Generators, wrappers, datasets, and tests must call the core transition API;
  do not reimplement C0–C3 elsewhere.
- Keep the package dependency-free at runtime. Optional integrations belong in
  downstream projects unless they are broadly useful and remain optional.
- A semantic change requires a version change, an update to
  `docs/environment-spec.md`, and regression tests.
- Do not couple the environment to JEPA, an LLM, a projector, a particular
  training split, an absolute path, or a GPU/runtime configuration.
- Preserve deterministic generation and rendering for a fixed configuration.
