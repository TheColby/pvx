# Release Process

`pvx` uses tag-based releases.

## Prerequisites

- `main` is green on required checks.
- `CHANGELOG.md` is updated.
- `pyproject.toml` version is updated.

## Pre-Tag Validation

Run release validation locally before creating the tag:

```bash
uv run python scripts/scripts_alpha_check.py
uv run pytest -q
uv run python benchmarks/run_augment_profile_suite.py --quick --gate --out-dir benchmarks/out_augment_profiles_release
```

For alpha releases, also review [docs/ALPHA_RELEASE.md](docs/ALPHA_RELEASE.md) and make sure the stable/beta/experimental CLI surface in `pyproject.toml` still matches the README and release notes.

`make alpha-check` remains as a convenience wrapper when local `make` is available.

If the Homebrew formula should track the tagged release instead of `--HEAD`, refresh it after tagging and publish the updated formula to [`TheColby/homebrew-pvx`](https://github.com/TheColby/homebrew-pvx):

```bash
./scripts/refresh_homebrew_formula.sh v0.1.0a1
```

Optional baseline refresh (only when benchmark behavior intentionally changed):

```bash
uv run python benchmarks/run_augment_profile_suite.py --quick --refresh-baselines --out-dir benchmarks/out_augment_profiles_refresh
```

## Create a Release Tag

```bash
git checkout main
git pull --rebase
git tag -a v0.1.0a1 -m "pvx v0.1.0a1 (alpha)"
git push origin v0.1.0a1
```

Pushing a `v*` tag triggers `.github/workflows/release.yml` to build artifacts
and publish a GitHub release.

## Attribution
