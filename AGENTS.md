# Repository Guidelines

## Project Structure & Module Organization
- `OHRgoal/` contains object-hit-ratio offline algorithms (`FOO`, `PFOO-U`, `PFOO-L`, `Belady`, etc.), each with its own `Makefile`.
- `BHRgoal/` contains byte-hit-ratio implementations (`PFOO-L`, `Belady`, `BeladySplit`).
- Shared C++ utilities live in `lib/` (notably `lib/trace/` for OracleGeneral trace parsing and test-trace generation).
- `tests/` contains C++ unit tests (Catch2). Additional Python validation scripts are at repo root (for example, `test_solver_small.py`) and in `foo-jax/tests/`.
- GPU-oriented Python implementations are in `foo-jax/` (JAX) and `cuopt-python/` (NVIDIA cuOpt).

## Build, Test, and Development Commands
- Build a C++ algorithm from its directory:
  - `make -C OHRgoal/FOO`
  - `make -C BHRgoal/PFOO-L`
- Build and run C++ unit tests:
  - `make -C tests test` (builds `tests/runtests` and executes it).
- Generate a sample OracleGeneral trace:
  - `make -C lib/trace test-gen && ./lib/trace/generate_test_trace test_trace`
- Run Python tests for the JAX package:
  - `cd foo-jax && pip install -e .[dev] && pytest`

## Coding Style & Naming Conventions
- C++: use existing Makefile defaults (`-std=c++11`, `-Wall`); follow the surrounding style in each module.
- Python: follow PEP 8, 4-space indentation, `snake_case` for functions/modules, `PascalCase` for classes.
- Test files should follow `test_*.py` or `test_*.cpp` naming, matching current patterns.
- Keep changes minimal and focused (KISS/YAGNI). Reuse shared parsing logic in `lib/trace` (DRY).

## Testing Guidelines
- For C++, add or extend Catch2 tests under `tests/` when changing shared logic.
- For algorithm behavior, run the target binary on a deterministic trace and compare key metrics (for example OHR/BHR).
- For Python packages, add pytest cases under `foo-jax/tests/` and run `pytest -v --tb=short`.

## Commit & Pull Request Guidelines
- Prefer Conventional Commits seen in history, e.g. `feat(foo-jax): ...`, `chore: ...`, `feat!: ...` for breaking changes.
- Keep commits scoped to one logical change.
- PRs should include: purpose, affected modules/paths, reproducible test commands, and representative output/metrics.
- Link related issues and call out trace format or performance-impact changes explicitly.

## Security & Configuration Tips
- Do not commit proprietary trace data or large generated binaries.
- Keep local paths and credentials out of code; use CLI arguments for trace and cache-size inputs.
