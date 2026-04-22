# Alpha Release Guide

This page is the release-facing contract for `pvx` alpha builds.

## Supported Surface

Stable entry points for `0.1.0a1`:
- `pvx`
- `pvxvoc`
- `pvxfreeze`
- `pvxwarp`
- `pvxformant`
- `pvxfilter`
- `pvxretune`
- `pvxanalysis`

Beta entry points:
- functional and documented, but flags may still move in minor releases

Experimental entry points:
- useful for exploration
- not promised stable for scripting

Compatibility shims:
- `pvxalgorithms`, `pvxalgorithms.base`, and `pvxalgorithms.registry` remain available only as deprecated migration shims

## Release Checklist

Before tagging an alpha build:

1. Run `uv run python scripts/scripts_alpha_check.py`
   Optional convenience wrapper: `make alpha-check` when local `make` is available
2. Run `uv run pytest -q`
3. Run `uv run python benchmarks/run_augment_profile_suite.py --quick --gate --out-dir benchmarks/out_augment_profiles_release`
4. Confirm `README.md`, `RELEASE.md`, and `pyproject.toml` agree on the stable/beta/experimental command surface
5. If publishing a tagged formula, run `./scripts/refresh_homebrew_formula.sh v0.1.0a1`, publish the refreshed formula to [`TheColby/homebrew-pvx`](https://github.com/TheColby/homebrew-pvx), and verify the tap instructions still match [docs/HOMEBREW.md](docs/HOMEBREW.md)
6. Regenerate source docs with `make docs`
7. Regenerate HTML/man outputs only when user-facing rendered docs changed or when preparing the tag with `make docs-generated`
8. Make sure deprecated compatibility layers still warn cleanly and still route to the canonical modules

## Alpha Principles

- prefer a smaller honest surface over a wider unstable one
- keep compatibility shims explicit and noisy rather than magical
- treat docs source files as the review artifact; generated outputs are release artifacts
- add direct seam tests when extracting modules from `pvx.py` or `pvx.core.voc`
