# Lessons from Paul Koonce's PVC for pvx

This note captures what remains valuable from the historical PVC package and how those ideas can continue to improve `pvx`.

Primary sources:
- Princeton archive page (Paul Koonce): [https://www.cs.princeton.edu/courses/archive/spr99/cs325/koonce.html](https://www.cs.princeton.edu/courses/archive/spr99/cs325/koonce.html)
- Linux Audio package index entry: [https://wiki.linuxaudio.org/apps/all/pvc](https://wiki.linuxaudio.org/apps/all/pvc)

## 1) What PVC Got Right

## 1.1 Composable command-line tools

PVC exposed many focused commands (`plainpv`, `twarp`, `harmonizer`, `ringfilter`, and others), each tuned for one processing goal.

Why it matters:
- small tools are easier to test and reason about
- shell pipelines naturally support complex workflows

`pvx` carry-forward:
- subcommands such as `voc`, `freeze`, `harmonize`, `conform`, `retune`, `morph`, `denoise`, `deverb`
- managed composition with `pvx chain`

## 1.2 Dynamic control from external files

PVC supported dynamic parameter control by reading function files and interpolating through time.

Why it matters:
- most real audio work needs parameter trajectories, not only constants

`pvx` carry-forward:
- CSV/JSON control-rate signals on core numeric parameters
- interpolation modes (`none`, `linear`, `nearest`, `cubic`, `polynomial`)

## 1.3 Terminal-first discoverability

PVC documentation emphasized "run routine name to see flags/defaults."

Why it matters:
- users working in terminals should not need to read source code first

`pvx` carry-forward:
- richer `--help` text with grouped argument sections
- `--example` and `--guided`
- script-level help epilogs with copy-paste commands

## 1.4 Explicit defaults and transparent controls

PVC exposed defaults directly in command help, including ranges and mode flags.

Why it matters:
- predictable defaults reduce trial-and-error and improve reproducibility

`pvx` carry-forward:
- explicit defaults in CLI help and docs
- deterministic central processing unit (CPU) mode
- reproducible benchmark scripts and quality gates

## 1.5 Script-based workflow reuse

PVC shipped shell scripts as reusable "poor-man's graphical user interface (GUI)" presets.

Why it matters:
- consistent processing recipes are as important as raw algorithms

`pvx` carry-forward:
- cookbook recipes (`docs/EXAMPLES.md`)
- `pvx examples <name>`
- benchmark, regression, and documentation generation scripts

## 2) Where pvx Can Improve Further (PVC-Inspired)

1. Keep adding "single-purpose tools with shared conventions" over monolithic flags.
2. Preserve plain-text control formats and avoid opaque binary automation formats.
3. Keep command help readable and practical before adding new flags.
4. Ensure each new advanced feature ships with at least one runnable example.
5. Maintain a strong terminal-first workflow for research and production use.

## 3) Suggested Ongoing Checklist

- For every new algorithm:
  - add one direct one-liner example in `--help`
  - add one entry in `docs/EXAMPLES.md`
  - add one "when not to use" note
- For every new dynamic parameter:
  - document scalar usage
  - document CSV control usage
  - document JSON control usage
  - document interpolation implications
- For every quality feature:
  - show baseline vs improved command pair
  - state key artifacts to listen for

