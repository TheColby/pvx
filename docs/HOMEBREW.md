# Installing and Managing pvx with Homebrew

Homebrew is the supported installation path for macOS users who want the `pvx` command suite without managing a project checkout or Python virtual environment. The formula is published through the [`TheColby/pvx`](https://github.com/TheColby/homebrew-pvx) tap because Homebrew requires third-party formulae to live in a tap. Installing `Formula/pvx.rb` directly from a local path is not supported by current Homebrew releases.

## Stable Installation

The stable formula installs the tagged alpha release. Homebrew fetches the release archive, verifies its SHA-256 checksum, installs `libsndfile`, creates an isolated Python environment, and links the command-line programs into the Homebrew prefix.

```bash
brew tap TheColby/pvx
brew install TheColby/pvx/pvx
```

The tap command only needs to be run once. After installation, verify both Homebrew's record and the primary command:

```bash
brew list --versions pvx
pvx --help
pvxvoc --help
```

Homebrew 6 requires trust for third-party tap content. Homebrew can trust the selected formula during installation; when it instead reports that the tap is not trusted, grant trust only to the pvx formula and retry:

```bash
brew trust --formula TheColby/pvx/pvx
brew install TheColby/pvx/pvx
```

Formula-level trust is narrower than trusting the entire tap. Inspect the formula in the tap repository before granting trust, particularly on a shared or security-sensitive machine.

The formula installs every command entry point shipped by the Python package. The alpha support promise remains deliberately narrower: `pvx`, `pvxvoc`, `pvxfreeze`, `pvxwarp`, `pvxformant`, `pvxfilter`, `pvxretune`, and `pvxanalysis` are the stable command-line surface documented for production scripts.

## Development Installation

The `HEAD` formula follows the latest commit on the repository's `main` branch. It is useful for testing an unreleased fix, but its behavior can move ahead of the guide and tagged release.

```bash
brew install --HEAD TheColby/pvx/pvx
```

Homebrew does not automatically replace a stable installation with `HEAD`. Reinstall explicitly when switching tracks:

```bash
brew uninstall pvx
brew install --HEAD TheColby/pvx/pvx
```

Return to the stable release in the same way:

```bash
brew uninstall pvx
brew install TheColby/pvx/pvx
```

## Upgrading

Homebrew updates the tap metadata before upgrading the package. These commands show the available formula and install a newer tagged version when one has been published:

```bash
brew update
brew info TheColby/pvx/pvx
brew upgrade pvx
```

A `HEAD` installation is rebuilt from the latest repository state with:

```bash
brew reinstall --HEAD TheColby/pvx/pvx
```

Existing audio files, render outputs, presets, and manifests are not stored inside the Homebrew cellar and are not removed by an upgrade.

## Uninstalling

Uninstalling removes the Homebrew-managed virtual environment and linked commands. It does not delete projects or rendered audio in user directories.

```bash
brew uninstall pvx
```

Remove the tap as well when no formula from it is needed:

```bash
brew untap TheColby/pvx
```

## What the Formula Installs

The formula uses Homebrew's `Language::Python::Virtualenv` support. pvx and its Python runtime dependencies live under the formula's private `libexec` directory, while small launcher scripts are linked into `$(brew --prefix)/bin`. This prevents pvx packages from modifying the system Python or an activated project environment.

`libsndfile` is a Homebrew dependency because pvx uses it for audio file input and output through the Python `soundfile` package. The stable formula currently uses Homebrew's `python@3.12`. Optional machine-learning integrations such as PyTorch and TensorFlow are intentionally excluded from the base formula because of their size and platform-specific installation requirements.

The formula also installs available manual pages from `man/man1`. Homebrew exposes them through its normal manual-page path, so a supported command can be inspected with a command such as:

```bash
man pvx
```

## Shell Path

Homebrew normally configures its binary directory during Homebrew installation. If the `brew` command works but `pvx` is not found after a successful install, inspect the active prefix and shell path:

```bash
brew --prefix
command -v brew
command -v pvx
printf '%s\n' "$PATH"
```

Apple Silicon Homebrew normally uses `/opt/homebrew`; Intel Homebrew normally uses `/usr/local`. Homebrew's recommended shell environment can be loaded on Apple Silicon with:

```bash
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Place the corresponding `brew shellenv` command in the shell startup file only when Homebrew itself is not already configuring the path.

## Troubleshooting

A failed install is easiest to diagnose by separating tap state, formula state, dependencies, and command linking. Begin with the following inspection commands:

```bash
brew update
brew tap
brew info TheColby/pvx/pvx
brew doctor
```

If Homebrew reports that a local `Formula/pvx.rb` is rejected, install through the tap instead:

```bash
brew tap TheColby/pvx
brew install TheColby/pvx/pvx
```

If an interrupted build left partial state, remove and reinstall the formula:

```bash
brew uninstall --force pvx
brew cleanup pvx
brew install TheColby/pvx/pvx
```

If Homebrew reports that the Xcode Command Line Tools are outdated, update them through System Settings under General and Software Update. Homebrew may require a newer Command Line Tools release even when the `clang` compiler already works for ordinary local builds. Complete that system update before retrying the pvx formula.

If audio files fail to open after installation, confirm that Homebrew installed and linked `libsndfile`, then run the pvx smoke test:

```bash
brew list --versions libsndfile
pvx smoke
```

The full Homebrew build log is available through Homebrew's usual diagnostic output. Include `brew config`, `brew info TheColby/pvx/pvx`, the failing command, and the relevant log tail in a bug report.

## Maintainer Release Flow

The repository keeps the source formula at `Formula/pvx.rb`. A tagged release must have a stable archive URL, its exact SHA-256 checksum, and an explicit version before the formula is copied to the tap. The refresh helper is repeatable, so it can update both a previously stamped formula and a newly prepared release.

```bash
./scripts/refresh_homebrew_formula.sh v0.1.0a1
ruby -c Formula/pvx.rb
python3 scripts/scripts_sync_homebrew_tap_formula.py ../homebrew-pvx
```

The tap checkout should then be tested as Homebrew sees it:

```bash
brew style TheColby/pvx/pvx
brew audit --strict TheColby/pvx/pvx
brew install --build-from-source TheColby/pvx/pvx
brew test TheColby/pvx/pvx
```

After a successful test, commit and push `Formula/pvx.rb` in [`TheColby/homebrew-pvx`](https://github.com/TheColby/homebrew-pvx). The tag-driven release workflow also validates Ruby syntax and uploads the stamped formula beside the Python distributions, providing one release artifact from which the tap can be synchronized.

## Formula Version Policy

The stable formula follows tagged pvx releases and never points at a moving branch. `HEAD` is the only branch-following installation. This separation keeps normal `brew upgrade` behavior reproducible while retaining an explicit route for development testing.

The Homebrew formula is a distribution mechanism, not a wider stability promise. Alpha, beta, and experimental command tiers remain governed by the supported-surface chapter and release notes. A successful installation confirms packaging and executable discovery; it does not promote experimental commands into the stable interface.
