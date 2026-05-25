# Supported Surface

This page is the short version of what `pvx` expects alpha users to rely on.

## Stable in `0.1.x`

These commands are the intended scripted/public alpha surface:

- `pvx`
- `pvxvoc`
- `pvxfreeze`
- `pvxwarp`
- `pvxformant`
- `pvxfilter`
- `pvxretune`
- `pvxanalysis`

## Beta in `0.1.x`

These are usable, but minor releases may still change flags or defaults:

- `pvxharmonize`
- `pvxconform`
- `pvxmorph`
- `pvxtransient`
- `pvxunison`
- `pvxdenoise`
- `pvxdeverb`
- `pvxlayer`
- `pvxresponse`
- `pvxenvelope`
- `pvxreshape`
- `pvxtvfilter`
- `pvxnoisefilter`
- `pvxbandamp`
- `pvxspeccompander`
- `pvxring`
- `pvxringfilter`
- `pvxringtvfilter`
- `pvxharmmap`

## Experimental in `0.1.x`

These are available for exploration, not for stable automation contracts:

- `hps-pitch-track`
- `pvxchordmapper`
- `pvxinharmonator`
- `pvxtrajectoryreverb`
- `pvxnoise`
- `pvxrir`
- `pvxcodec`
- `pvxspecaugment`
- `pvxgain`

## Compatibility Shims

The `pvxalgorithms*` import aliases remain available during `0.1.x` only to ease migration.

- Canonical imports: `pvx.algorithms*`
- Deprecated aliases: `pvxalgorithms`, `pvxalgorithms.base`, `pvxalgorithms.registry`
- Removal target: first `0.2.0` release unless the release notes say otherwise

## Docs Guide

If you only need the supported user-facing paths, start with:

- [README](../README.md)
- [Getting Started](GETTING_STARTED.md)
- [Homebrew Install](HOMEBREW.md)
- [Alpha Release Guide](ALPHA_RELEASE.md)

Use the generated references only when you need exhaustive flag/file inventory:

- [CLI Flags Reference](CLI_FLAGS_REFERENCE.md)
- [Python File Help](PYTHON_FILE_HELP.md)
