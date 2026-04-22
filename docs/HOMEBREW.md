# Homebrew Install

`pvx` ships with a Homebrew formula in `Formula/pvx.rb`, published through the tap at [`TheColby/homebrew-pvx`](https://github.com/TheColby/homebrew-pvx).

## Install From This Repo

Current development / pre-release flow:

```bash
brew tap TheColby/pvx
brew install --HEAD TheColby/pvx/pvx
```

The tap formula creates an isolated Python virtualenv, installs the runtime Python dependencies, and links the `pvx*` CLI entry points.

## Stable Tagged Release Flow

After a tag is published, stamp the formula with the tagged source tarball checksum. The helper updates the repo-local formula; then copy or commit that formula into the tap repository:

```bash
./scripts/refresh_homebrew_formula.sh v0.1.0a1
```

That updates `Formula/pvx.rb` with:
- the tagged GitHub source tarball URL
- the matching `sha256`
- the explicit formula version

Once the stamped formula is committed in [`TheColby/homebrew-pvx`](https://github.com/TheColby/homebrew-pvx), users can install the tagged formula without `--HEAD`:

```bash
brew tap TheColby/pvx
brew install TheColby/pvx/pvx
```

## Notes

- The formula currently targets `python@3.12`.
- Runtime audio I/O depends on Homebrew `libsndfile`.
- This formula is aimed at the project tap workflow, not Homebrew core submission.
- Homebrew 5.x rejects local/raw formula installs for this use case, so the tap is the supported install path.
