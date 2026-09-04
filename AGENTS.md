# Repository invariants

- `config/experiment.json` is the sole machine-readable environment contract.
- `src/switchdoor_three_level/model.py` is the sole executable source of
  transition truth. Offline generation and the live environment must reuse it.
- Offline and Web observations must reuse the registered renderer; do not
  create a second physics or rendering implementation.
- Preserve the public, supervision, and private information boundary defined
  in `EXPERIMENT_PROTOCOL.zh.md`.
- Keep the package dependency-free at runtime.
- A semantic change requires a package version change, protocol/config updates,
  and regression tests.
- Do not couple the environment to JEPA, an LLM, training code, an absolute
  path, or a GPU/runtime configuration.
- Preserve deterministic generation and rendering for a fixed configuration.
