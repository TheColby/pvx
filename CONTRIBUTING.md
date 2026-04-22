# Contributing to pvx

Thanks for contributing to `pvx`.

This project is research- and production-oriented. Changes should prioritize
audio quality first, speed second, and preserve command-line interface (CLI)
backward compatibility unless explicitly discussed.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -e ".[dev]"
```

Then install the pre-commit hooks (required — they enforce the lint and format
gates that CI also runs):

```bash
pre-commit install
```

## Lint and Format

`pvx` uses [ruff](https://docs.astral.sh/ruff/) for both linting and formatting.
The full configuration lives in `pyproject.toml` under `[tool.ruff]`.

```bash
# Check (matches CI)
ruff check src/ tests/ benchmarks/
ruff format --check src/ tests/ benchmarks/

# Auto-fix
ruff check --fix src/ tests/ benchmarks/
ruff format src/ tests/ benchmarks/
```

The pre-commit hook runs these on staged files automatically. Run
`pre-commit run --all-files` locally if you want to mirror the CI lint job.

## Required Checks Before Opening a Pull Request

```bash
pre-commit run --all-files       # ruff check + ruff format + housekeeping hooks
pytest tests/                    # unit tests
make docs                        # docs regeneration
make typecheck                   # mypy gate
```

If you do not use `make`, run the underlying commands directly.

## CI Gates

Every pull request must pass the jobs in `.github/workflows/ci.yml`:

| Job          | What it runs                                                          |
|--------------|------------------------------------------------------------------------|
| `test`       | `pytest` matrix across Python 3.10–3.13 on Linux + macOS               |
| `test-torch` | GPU transform tests with the optional `[torch]` extra                  |
| `lint`       | `ruff check`, `ruff format --check`, `pre-commit run --all-files`, mypy |
| `security`   | `bandit` static scan + `pip-audit` vulnerability scan                  |
| `build`      | `python -m build` + `twine check`                                      |
| `docs`       | Docs coverage tests (`tests/test_docs_coverage.py`)                    |

If a CI job fails, run the equivalent command locally to reproduce.

## Plugin Transforms

External packages can register augmentation transforms via setuptools entry
points without touching pvx itself:

```toml
# In your package's pyproject.toml
[project.entry-points."pvx.augment.transforms"]
my_reverb = "my_package.transforms:MyReverb"
```

After install, the transform is auto-discovered and available via
`pvx.augment.get_transform("my_reverb")` and any pipeline-by-name lookup.

## Pull Request Scope

- Keep changes focused and reviewable.
- Include tests for behavioral changes.
- Update documentation in the same pull request when behavior, parameters,
  outputs, or defaults change.
- Preserve existing command-line interfaces (`pvx`, `pvxvoc`, `pvxfreeze`, etc.).

## Commit Messages

Use clear imperative messages, for example:

- `Add transient region merge guard`
- `Fix CSV interpolation edge case at file end`
- `Regenerate docs after CLI flag update`

## DSP and Audio Quality Expectations

- Document algorithm assumptions and failure modes.
- Provide deterministic mode behavior for reproducibility.
- Avoid regressions in transient handling and stereo coherence.
- Validate objective metrics when changing core transforms or phase logic.

## Documentation Contract

When touching documentation generators, regenerate and commit generated outputs:

```bash
python3 scripts/scripts_generate_python_docs.py
python3 scripts/scripts_generate_theory_docs.py
python3 scripts/scripts_generate_docs_extras.py
python3 scripts/scripts_generate_html_docs.py
```

## Attribution


