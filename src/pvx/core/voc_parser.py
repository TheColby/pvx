"""Parser and argument validation helpers for `pvx voc`."""

from __future__ import annotations

import argparse
from pathlib import Path

from pvx.core import voc as voc_core
from pvx.core.control_bus import parse_control_routes
from pvx.core.output_policy import (
    BIT_DEPTH_CHOICES,
    DITHER_CHOICES,
    METADATA_POLICY_CHOICES,
    validate_output_policy_args,
)
from pvx.core.presets import PRESET_CHOICES
from pvx.core.voc_console import EXAMPLE_CHOICES, add_console_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-vocoder CLI for multi-file, multi-channel time stretching and pitch shifting."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Beginner examples:\n"
            "  pvx voc input.wav --stretch 1.2 --output output.wav\n"
            "  pvx voc vocal.wav --preset vocal --pitch -2 --output vocal_tuned.wav\n"
            "  pvx voc speech.wav --transient-mode hybrid --stretch 1.25 --output speech_hybrid.wav\n"
            "  pvx voc stereo.wav --stereo-mode mid_side_lock --coherence-strength 0.9 --stretch 1.2 --output stereo_lock.wav\n"
            "  pvx pitch-track A.wav --emit pitch_to_stretch --output - | pvx voc B.wav --control-stdin --output B_follow.wav\n"
            "  pvx voc input.wav --stretch controls/stretch.csv --interp linear --output output.wav\n"
            "  pvx voc input.wav --example all\n"
        ),
    )

    parser.add_argument("inputs", nargs="*", help="Input audio files/globs or '-' for stdin")

    io_args_group = parser.add_argument_group("I/O")
    io_args_group.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files (default: same directory as each input)",
    )
    io_args_group.add_argument(
        "--suffix",
        default="_pv",
        help="Suffix appended to output filename stem (default: _pv)",
    )
    io_args_group.add_argument(
        "--output-format",
        default=None,
        help="Output format/extension (e.g. wav, flac, aiff). Default: keep input extension.",
    )
    io_args_group.add_argument(
        "--out",
        "--output",
        dest="output",
        type=Path,
        default=None,
        help="Explicit output file path (single-input mode only). Alias: --out",
    )
    io_args_group.add_argument(
        "--overwrite",
        "--force",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing outputs and other write targets",
    )
    io_args_group.add_argument(
        "--dry-run", action="store_true", help="Resolve settings without writing files"
    )
    io_args_group.add_argument(
        "--stdout",
        action="store_true",
        help="Write processed audio to stdout stream (for piping); requires exactly one input",
    )

    debug_group = parser.add_argument_group("Debug")
    add_console_args(debug_group, include_no_progress_alias=True)

    beginner_group = parser.add_argument_group("Beginner experience")
    beginner_group.add_argument(
        "--preset",
        choices=list(PRESET_CHOICES),
        default="none",
        help=(
            "High-level intent preset. Legacy: none/vocal/ambient/extreme. "
            "New: default/vocal_studio/drums_safe/extreme_ambient/stereo_coherent."
        ),
    )
    beginner_group.add_argument(
        "--example",
        choices=list(EXAMPLE_CHOICES),
        default=None,
        help="Print copy-paste example command(s) and exit.",
    )
    beginner_group.add_argument(
        "--guided",
        action="store_true",
        help="Interactive guided mode for first-time users.",
    )
    beginner_group.add_argument(
        "--stretch",
        type=str,
        default=None,
        help="Alias for --time-stretch. Accepts scalar or control file (.csv/.json).",
    )
    beginner_group.add_argument(
        "--gpu",
        action="store_true",
        help="Alias for --device cuda.",
    )
    beginner_group.add_argument(
        "--cpu",
        action="store_true",
        help="Alias for --device cpu.",
    )

    planning_group = parser.add_argument_group("Performance")
    planning_group.add_argument(
        "--quality-profile",
        choices=list(voc_core.QUALITY_PROFILE_CHOICES),
        default="neutral",
        help="Named tuning profile for vocoder defaults (default: neutral)",
    )
    planning_group.add_argument(
        "--auto-profile",
        action="store_true",
        help="Analyze input and choose a profile automatically (speech/music/percussion/ambient/extreme).",
    )
    planning_group.add_argument(
        "--auto-profile-lookahead-seconds",
        type=float,
        default=6.0,
        help="Seconds of audio used when estimating --auto-profile (default: 6.0).",
    )
    planning_group.add_argument(
        "--auto-transform",
        action="store_true",
        help="Allow automatic transform selection when --transform is not explicitly set.",
    )

    stft_group = parser.add_argument_group("Quality/Phase")
    stft_group.add_argument(
        "--n-fft",
        type=str,
        default=2048,
        help="FFT size (default: 2048). Accepts scalar or control file (.csv/.json).",
    )
    stft_group.add_argument(
        "--win-length",
        type=str,
        default=2048,
        help="Window length in samples (default: 2048). Accepts scalar or control file (.csv/.json).",
    )
    stft_group.add_argument(
        "--hop-size",
        type=str,
        default=512,
        help="Hop size in samples (default: 512). Accepts scalar or control file (.csv/.json).",
    )
    stft_group.add_argument(
        "--window",
        choices=list(voc_core.WINDOW_CHOICES),
        default="hann",
        help="Window type (default: hann)",
    )
    stft_group.add_argument(
        "--kaiser-beta",
        type=str,
        default=14.0,
        help="Kaiser window beta parameter used when --window kaiser (default: 14.0). Accepts scalar or control file (.csv/.json).",
    )
    stft_group.add_argument(
        "--transform",
        choices=list(voc_core.TRANSFORM_CHOICES),
        default="fft",
        help=(
            "Per-frame transform backend for STFT/ISTFT paths "
            "(default: fft; options: fft, dft, czt, dct, dst, hartley)"
        ),
    )
    stft_group.add_argument(
        "--no-center",
        action="store_true",
        help="Disable center padding in STFT/ISTFT",
    )
    stft_group.add_argument(
        "--phase-locking",
        choices=["off", "identity"],
        default="identity",
        help="Inter-bin phase locking mode for transient fidelity (default: identity)",
    )
    stft_group.add_argument(
        "--phase-engine",
        choices=list(voc_core.PHASE_ENGINE_CHOICES),
        default="propagate",
        help=(
            "Phase synthesis engine: propagate (classic phase vocoder), "
            "hybrid (propagated + stochastic blend), random (ambient stochastic phase)."
        ),
    )
    stft_group.add_argument(
        "--ambient-phase-mix",
        type=str,
        default=0.5,
        help=(
            "Random-phase blend when --phase-engine hybrid "
            "(0.0=propagated only, 1.0=random only; default: 0.5). "
            "Accepts scalar or control file (.csv/.json)."
        ),
    )
    stft_group.add_argument(
        "--phase-random-seed",
        type=int,
        default=None,
        help="Optional deterministic seed for random/hybrid phase generation.",
    )
    stft_group.add_argument(
        "--transient-preserve",
        action="store_true",
        help="Enable transient phase resets based on spectral flux",
    )
    stft_group.add_argument(
        "--transient-threshold",
        type=str,
        default=2.0,
        help="Spectral-flux multiplier for transient detection (default: 2.0). Accepts scalar or control file (.csv/.json).",
    )
    stft_group.add_argument(
        "--fourier-sync",
        action="store_true",
        help=(
            "Enable fundamental frame locking. Uses generic short-time Fourier "
            "transforms with per-frame FFT sizes locked to detected F0."
        ),
    )
    stft_group.add_argument(
        "--fourier-sync-min-fft",
        type=str,
        default=256,
        help="Minimum frame FFT size for --fourier-sync (default: 256). Accepts scalar or control file (.csv/.json).",
    )
    stft_group.add_argument(
        "--fourier-sync-max-fft",
        type=str,
        default=8192,
        help="Maximum frame FFT size for --fourier-sync (default: 8192). Accepts scalar or control file (.csv/.json).",
    )
    stft_group.add_argument(
        "--fourier-sync-smooth",
        type=str,
        default=5,
        help="Smoothing span (frames) for prescanned F0 track in --fourier-sync (default: 5). Accepts scalar or control file (.csv/.json).",
    )
    stft_group.add_argument(
        "--multires-fusion",
        action="store_true",
        help="Blend multiple FFT resolutions for each channel before pitch resampling.",
    )
    stft_group.add_argument(
        "--multires-ffts",
        type=str,
        default="1024,2048,4096",
        help="Comma-separated FFT sizes for --multires-fusion (default: 1024,2048,4096)",
    )
    stft_group.add_argument(
        "--multires-weights",
        type=str,
        default=None,
        help="Comma-separated fusion weights for --multires-fusion (defaults to equal weights).",
    )
    voc_core.add_runtime_args(stft_group)

    time_group = parser.add_argument_group("Time/Pitch")
    time_group.add_argument(
        "--time-stretch",
        "--time-stretch-factor",
        type=str,
        default=1.0,
        help="Final duration multiplier (1.0=unchanged, 2.0=2x longer). Accepts scalar or control file (.csv/.json).",
    )
    time_group.add_argument(
        "--target-duration",
        type=float,
        default=None,
        help="Absolute target duration in seconds (overrides --time-stretch)",
    )
    time_group.add_argument(
        "--stretch-mode",
        choices=["auto", "standard", "multistage"],
        default="auto",
        help=(
            "Stretch strategy: standard (single pass), multistage (chained moderate passes), "
            "or auto (multistage only for extreme ratios; default: auto)."
        ),
    )
    time_group.add_argument(
        "--extreme-time-stretch",
        action="store_true",
        help="Force multistage strategy even when ratio is moderate.",
    )
    time_group.add_argument(
        "--extreme-stretch-threshold",
        type=str,
        default=2.0,
        help="Auto-mode threshold for multistage activation (default: 2.0). Accepts scalar or control file (.csv/.json).",
    )
    time_group.add_argument(
        "--max-stage-stretch",
        type=str,
        default=1.8,
        help="Maximum per-stage ratio used in multistage mode (default: 1.8). Accepts scalar or control file (.csv/.json).",
    )
    time_group.add_argument(
        "--onset-time-credit",
        action="store_true",
        help=(
            "Enable onset-triggered time-credit scheduling to reduce transient smear "
            "during extreme stretching."
        ),
    )
    time_group.add_argument(
        "--onset-credit-pull",
        type=str,
        default=0.5,
        help=(
            "Fraction of per-frame read advance removable while onset credit exists "
            "(0.0..1.0, default: 0.5). Accepts scalar or control file (.csv/.json)."
        ),
    )
    time_group.add_argument(
        "--onset-credit-max",
        type=str,
        default=8.0,
        help="Maximum accumulated onset time credit in analysis-frame units (default: 8.0). Accepts scalar or control file (.csv/.json).",
    )
    time_group.add_argument(
        "--no-onset-realign",
        action="store_true",
        help=(
            "Disable fractional read-position realignment on onsets when "
            "--onset-time-credit is enabled."
        ),
    )
    time_group.add_argument(
        "--ambient-preset",
        action="store_true",
        help=(
            "Convenience preset for ambient extreme stretch "
            "(random phase engine, onset-time-credit, transient preserve, conservative staging)."
        ),
    )
    time_group.add_argument(
        "--auto-segment-seconds",
        type=float,
        default=0.0,
        help=(
            "Optional segment size in seconds for long jobs. "
            "When >0, processing runs per segment with crossfade assembly."
        ),
    )
    time_group.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory used to cache per-segment checkpoint chunks for resume workflows.",
    )
    time_group.add_argument(
        "--checkpoint-id",
        type=str,
        default=None,
        help="Optional checkpoint run identifier (default: hash of input/settings).",
    )
    time_group.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing checkpoint chunks from --checkpoint-dir when available.",
    )
    time_group.add_argument(
        "--interp",
        choices=list(voc_core.CONTROL_INTERP_CHOICES),
        default="linear",
        help=(
            "Interpolation mode for time-varying control signals loaded from CSV/JSON "
            "(default: linear)."
        ),
    )
    time_group.add_argument(
        "--order",
        type=int,
        default=3,
        help=(
            "Polynomial order for --interp polynomial (default: 3). "
            "Accepts any integer >= 1; effective fit degree is min(order, control_points-1)."
        ),
    )

    transient_group = parser.add_argument_group("Transients")
    transient_group.add_argument(
        "--transient-mode",
        choices=["off", "reset", "hybrid", "wsola"],
        default="off",
        help=(
            "Transient handling mode: off (none), reset (phase reset), "
            "hybrid (PV steady + WSOLA transients), or wsola (time-domain transient-safe path)."
        ),
    )
    transient_group.add_argument(
        "--transient-sensitivity",
        type=str,
        default=0.5,
        help="Transient detector sensitivity in [0,1] (higher catches more onsets). Accepts scalar or control file (.csv/.json).",
    )
    transient_group.add_argument(
        "--transient-protect-ms",
        type=str,
        default=30.0,
        help="Transient protection width in milliseconds (default: 30). Accepts scalar or control file (.csv/.json).",
    )
    transient_group.add_argument(
        "--transient-crossfade-ms",
        type=str,
        default=10.0,
        help="Crossfade duration for transient/steady stitching (default: 10 ms). Accepts scalar or control file (.csv/.json).",
    )

    stereo_group = parser.add_argument_group("Stereo")
    stereo_group.add_argument(
        "--stereo-mode",
        choices=["independent", "mid_side_lock", "ref_channel_lock"],
        default="independent",
        help=(
            "Channel coherence strategy: independent (legacy), "
            "mid_side_lock (M/S-coupled), ref_channel_lock (phase-lock to reference channel)."
        ),
    )
    stereo_group.add_argument(
        "--ref-channel",
        type=int,
        default=0,
        help="Reference channel index used by --stereo-mode ref_channel_lock (default: 0).",
    )
    stereo_group.add_argument(
        "--coherence-strength",
        type=str,
        default=0.0,
        help="Coherence lock strength in [0,1] (0=off, 1=full lock). Accepts scalar or control file (.csv/.json).",
    )

    pitch_group = time_group
    pitch_mutex = pitch_group.add_mutually_exclusive_group()
    pitch_mutex.add_argument(
        "--pitch-shift-semitones",
        "--target-pitch-shift-semitones",
        "--pitch",
        "--semitones",
        type=str,
        default=None,
        help="Pitch shift in semitones (+12 is one octave up). Accepts scalar or control file (.csv/.json).",
    )
    pitch_mutex.add_argument(
        "--pitch-shift-cents",
        "--cents",
        type=str,
        default=None,
        help="Pitch shift in cents (+1200 is one octave up). Accepts scalar or control file (.csv/.json).",
    )
    pitch_mutex.add_argument(
        "--pitch-shift-ratio",
        "--ratio",
        type=str,
        default=None,
        help=(
            "Pitch ratio (>1 up, <1 down). Accepts decimals (1.5), "
            "integer ratios (3/2), expressions (2^(1/12)), or a control file (.csv/.json)."
        ),
    )
    pitch_mutex.add_argument(
        "--target-f0",
        type=float,
        default=None,
        help="Target fundamental frequency in Hz. Auto-estimates source F0 per file.",
    )
    pitch_group.add_argument(
        "--analysis-channel",
        choices=["first", "mix"],
        default="mix",
        help="Channel strategy for F0 estimation with --target-f0 (default: mix)",
    )
    pitch_group.add_argument(
        "--f0-min",
        type=float,
        default=50.0,
        help="Minimum F0 search bound in Hz (default: 50)",
    )
    pitch_group.add_argument(
        "--f0-max",
        type=float,
        default=1000.0,
        help="Maximum F0 search bound in Hz (default: 1000)",
    )
    pitch_group.add_argument(
        "--pitch-mode",
        choices=["standard", "formant-preserving"],
        default="standard",
        help="Pitch mode: standard shift or formant-preserving correction (default: standard)",
    )
    pitch_group.add_argument(
        "--formant-lifter",
        type=str,
        default=32,
        help="Cepstral lifter cutoff for formant envelope extraction (default: 32). Accepts scalar or control file (.csv/.json).",
    )
    pitch_group.add_argument(
        "--formant-strength",
        type=str,
        default=1.0,
        help="Formant correction blend 0..1 when pitch mode is formant-preserving (default: 1.0). Accepts scalar or control file (.csv/.json).",
    )
    pitch_group.add_argument(
        "--formant-max-gain-db",
        type=str,
        default=12.0,
        help="Max per-bin formant correction gain in dB (default: 12). Accepts scalar or control file (.csv/.json).",
    )
    pitch_group.add_argument(
        "--pitch-map",
        type=Path,
        default=None,
        help=(
            "CSV control map for time-varying stretch/pitch. "
            "Columns: start_sec,end_sec plus optional stretch,pitch_ratio/pitch_cents/pitch_semitones,confidence. "
            "Use '-' to read from stdin."
        ),
    )
    pitch_group.add_argument(
        "--pitch-map-stdin",
        action="store_true",
        help="Read control-map CSV from stdin.",
    )
    pitch_group.add_argument(
        "--control-stdin",
        action="store_true",
        help="Alias for --pitch-map-stdin (canonical control-bus CSV stdin path).",
    )
    pitch_group.add_argument(
        "--route",
        action="append",
        default=[],
        metavar="EXPR",
        help=(
            "Control-bus routing expression for map rows. Repeat flag to chain routes. "
            "Syntax: target=source, target=const(v), target=inv(source), target=pow(source,exp), "
            "target=mul(source,factor), target=add(source,offset), target=affine(source,scale,bias), "
            "target=clip(source,lo,hi). Targets: stretch,pitch_ratio. "
            "Sources: any numeric column present in the control-map CSV."
        ),
    )
    pitch_group.add_argument(
        "--pitch-follow-stdin",
        action="store_true",
        help="Shortcut for --pitch-map-stdin (sidechain pitch-follow workflows).",
    )
    pitch_group.add_argument(
        "--pitch-conf-min",
        type=float,
        default=0.0,
        help="Minimum accepted map confidence (default: 0 disables gating).",
    )
    pitch_group.add_argument(
        "--pitch-lowconf-mode",
        choices=["hold", "unity", "interp"],
        default="hold",
        help="Low-confidence map handling mode (default: hold).",
    )
    pitch_group.add_argument(
        "--pitch-map-smooth-ms",
        type=float,
        default=0.0,
        help="Moving-average smoothing over map pitch ratios in milliseconds.",
    )
    pitch_group.add_argument(
        "--pitch-map-crossfade-ms",
        type=float,
        default=8.0,
        help="Crossfade between processed map segments in milliseconds (default: 8.0).",
    )

    output_group = parser.add_argument_group("Output/Mastering")
    output_group.add_argument(
        "--target-sample-rate",
        type=int,
        default=None,
        help="Output sample rate in Hz (default: keep input rate)",
    )
    output_group.add_argument(
        "--resample-mode",
        choices=["auto", "fft", "linear"],
        default="auto",
        help="Resampling engine (auto=fft if scipy available, else linear)",
    )
    voc_core.add_mastering_args(output_group)
    output_group.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="Write processing manifest JSON with per-file settings and outcomes.",
    )
    output_group.add_argument(
        "--manifest-append",
        action="store_true",
        help="Append entries to an existing --manifest-json file instead of replacing it.",
    )
    output_group.add_argument(
        "--subtype",
        default=None,
        help="Explicit libsndfile output subtype override (e.g., PCM_16, PCM_24, FLOAT)",
    )
    output_group.add_argument(
        "--bit-depth",
        choices=list(BIT_DEPTH_CHOICES),
        default="inherit",
        help="Output bit-depth policy (default: inherit). Ignored when --subtype is set.",
    )
    output_group.add_argument(
        "--dither",
        choices=list(DITHER_CHOICES),
        default="none",
        help="Dither policy before quantized writes (default: none)",
    )
    output_group.add_argument(
        "--dither-seed",
        type=int,
        default=None,
        help="Deterministic RNG seed for dithering (default: random seed)",
    )
    output_group.add_argument(
        "--true-peak-max-dbtp",
        type=float,
        default=None,
        help="Apply output gain trim to enforce max true-peak in dBTP",
    )
    output_group.add_argument(
        "--metadata-policy",
        choices=list(METADATA_POLICY_CHOICES),
        default="none",
        help="Output metadata policy: none, sidecar, or copy (sidecar implementation)",
    )

    debug_group.add_argument(
        "--explain-plan",
        action="store_true",
        help="Print resolved processing plan JSON and exit without rendering audio.",
    )

    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    raw_dynamic_values: dict[str, str] = dict(
        getattr(args, "_dynamic_control_raw_values", {}) or {}
    )
    for attr_name, raw_value in raw_dynamic_values.items():
        if hasattr(args, attr_name):
            setattr(args, attr_name, raw_value)

    if args.stretch is not None:
        args.time_stretch = args.stretch

    interp_mode = voc_core._coerce_control_interp(
        getattr(args, "interp", "linear"), context="--interp"
    )
    args.interp = interp_mode
    args.order = voc_core._parse_int_cli_value(getattr(args, "order", 3), context="--order")
    if int(args.order) < 1:
        parser.error("--order must be >= 1")

    dynamic_refs: dict[str, voc_core.DynamicControlRef] = {}

    for attr, parameter, value_kind, default_value in voc_core._DYNAMIC_NUMERIC_ARG_SPECS:
        raw = getattr(args, attr)
        if voc_core._looks_like_control_signal_reference(raw):
            ref = voc_core.DynamicControlRef(
                parameter=parameter,
                path=Path(str(raw)).expanduser(),
                value_kind=value_kind,
                interpolation=interp_mode,
                order=int(args.order),
            )
            if str(ref.path) == "-":
                parser.error(
                    f"Dynamic control for --{attr.replace('_', '-')} does not support stdin ('-')"
                )
            if ref.path.suffix.lower() not in {".csv", ".json"}:
                parser.error(
                    f"--{attr.replace('_', '-')} control file must use .csv or .json extension: {ref.path}"
                )
            dynamic_refs[parameter] = ref
            raw_dynamic_values[attr] = str(ref.path)
            if value_kind == "int":
                setattr(args, attr, int(round(default_value)))
            else:
                setattr(args, attr, float(default_value))
        else:
            raw_dynamic_values.pop(attr, None)
            try:
                if value_kind == "int":
                    setattr(
                        args,
                        attr,
                        voc_core._parse_int_cli_value(
                            raw,
                            context=f"--{attr.replace('_', '-')}",
                        ),
                    )
                else:
                    setattr(
                        args,
                        attr,
                        voc_core._parse_scalar_cli_value(
                            raw,
                            context=f"--{attr.replace('_', '-')}",
                        ),
                    )
            except ValueError as exc:
                parser.error(str(exc))

    pitch_ratio = getattr(args, "pitch_shift_ratio", None)
    if pitch_ratio is not None:
        if voc_core._looks_like_control_signal_reference(pitch_ratio):
            ref = voc_core.DynamicControlRef(
                parameter="pitch_ratio",
                path=Path(str(pitch_ratio)).expanduser(),
                value_kind="pitch_ratio",
                interpolation=interp_mode,
                order=int(args.order),
            )
            if str(ref.path) == "-":
                parser.error("--pitch-shift-ratio dynamic control does not support stdin ('-')")
            if ref.path.suffix.lower() not in {".csv", ".json"}:
                parser.error(f"--pitch-shift-ratio control file must be .csv or .json: {ref.path}")
            dynamic_refs["pitch_ratio"] = ref
            raw_dynamic_values["pitch_shift_ratio"] = str(ref.path)
            args.pitch_shift_ratio = None
        else:
            raw_dynamic_values.pop("pitch_shift_ratio", None)
            try:
                args.pitch_shift_ratio = voc_core.parse_pitch_ratio_value(
                    pitch_ratio,
                    context="--pitch-shift-ratio",
                )
            except ValueError as exc:
                parser.error(str(exc))

    pitch_semitones = getattr(args, "pitch_shift_semitones", None)
    if pitch_semitones is not None:
        if voc_core._looks_like_control_signal_reference(pitch_semitones):
            ref = voc_core.DynamicControlRef(
                parameter="pitch_ratio",
                path=Path(str(pitch_semitones)).expanduser(),
                value_kind="pitch_semitones",
                interpolation=interp_mode,
                order=int(args.order),
            )
            if str(ref.path) == "-":
                parser.error("--pitch-shift-semitones dynamic control does not support stdin ('-')")
            if ref.path.suffix.lower() not in {".csv", ".json"}:
                parser.error(
                    f"--pitch-shift-semitones control file must be .csv or .json: {ref.path}"
                )
            dynamic_refs["pitch_ratio"] = ref
            raw_dynamic_values["pitch_shift_semitones"] = str(ref.path)
            args.pitch_shift_semitones = None
        else:
            raw_dynamic_values.pop("pitch_shift_semitones", None)
            try:
                args.pitch_shift_semitones = voc_core._parse_scalar_cli_value(
                    pitch_semitones,
                    context="--pitch-shift-semitones",
                )
            except ValueError as exc:
                parser.error(str(exc))

    pitch_cents = getattr(args, "pitch_shift_cents", None)
    if pitch_cents is not None:
        if voc_core._looks_like_control_signal_reference(pitch_cents):
            ref = voc_core.DynamicControlRef(
                parameter="pitch_ratio",
                path=Path(str(pitch_cents)).expanduser(),
                value_kind="pitch_cents",
                interpolation=interp_mode,
                order=int(args.order),
            )
            if str(ref.path) == "-":
                parser.error("--pitch-shift-cents dynamic control does not support stdin ('-')")
            if ref.path.suffix.lower() not in {".csv", ".json"}:
                parser.error(f"--pitch-shift-cents control file must be .csv or .json: {ref.path}")
            dynamic_refs["pitch_ratio"] = ref
            raw_dynamic_values["pitch_shift_cents"] = str(ref.path)
            args.pitch_shift_cents = None
        else:
            raw_dynamic_values.pop("pitch_shift_cents", None)
            try:
                args.pitch_shift_cents = voc_core._parse_scalar_cli_value(
                    pitch_cents,
                    context="--pitch-shift-cents",
                )
            except ValueError as exc:
                parser.error(str(exc))

    args._dynamic_control_refs = dynamic_refs
    args._dynamic_control_raw_values = raw_dynamic_values

    if args.gpu and args.cpu:
        parser.error("Choose only one of --gpu or --cpu.")
    if args.gpu:
        args.device = "cuda"
    if args.cpu:
        args.device = "cpu"

    if args.pitch_follow_stdin:
        args.pitch_map_stdin = True
    if bool(getattr(args, "control_stdin", False)):
        args.pitch_map_stdin = True

    route_exprs = list(getattr(args, "route", []) or [])
    try:
        args._control_routes = parse_control_routes(route_exprs)
    except ValueError as exc:
        parser.error(str(exc))

    if args.n_fft <= 0:
        parser.error("--n-fft must be > 0")
    if args.win_length <= 0:
        parser.error("--win-length must be > 0")
    if args.win_length > args.n_fft:
        parser.error("--win-length must be <= --n-fft")
    if args.hop_size <= 0:
        parser.error("--hop-size must be > 0")
    if args.hop_size > args.win_length:
        parser.error("--hop-size should be <= --win-length")
    if args.time_stretch <= 0:
        parser.error("--time-stretch must be > 0")
    if args.extreme_stretch_threshold <= 1.0:
        parser.error("--extreme-stretch-threshold must be > 1.0")
    if args.max_stage_stretch <= 1.0:
        parser.error("--max-stage-stretch must be > 1.0")
    if args.output is not None and args.output_dir is not None:
        parser.error("--output cannot be combined with --output-dir")
    if args.output is not None and args.stdout:
        parser.error("--output cannot be combined with --stdout")
    if args.target_duration is not None and args.target_duration <= 0:
        parser.error("--target-duration must be > 0")
    if args.pitch_conf_min < 0.0:
        parser.error("--pitch-conf-min must be >= 0")
    if args.pitch_map_smooth_ms < 0.0:
        parser.error("--pitch-map-smooth-ms must be >= 0")
    if args.pitch_map_crossfade_ms < 0.0:
        parser.error("--pitch-map-crossfade-ms must be >= 0")
    if dynamic_refs and (args.pitch_map is not None or args.pitch_map_stdin):
        parser.error(
            "Dynamic per-parameter control files cannot be combined with --pitch-map/--pitch-map-stdin"
        )
    if "time_stretch" in dynamic_refs and args.target_duration is not None:
        parser.error(
            "--target-duration cannot be combined with dynamic --time-stretch control files"
        )
    for ref in dynamic_refs.values():
        if not ref.path.exists():
            parser.error(f"Dynamic control file not found: {ref.path}")
    if args.pitch_map_stdin and args.pitch_map is not None and str(args.pitch_map) != "-":
        parser.error("--pitch-map-stdin cannot be combined with --pitch-map path")
    if args._control_routes and not (args.pitch_map is not None or args.pitch_map_stdin):
        parser.error("--route requires --pitch-map, --pitch-map-stdin, or --control-stdin")
    if args.target_f0 is not None and args.target_f0 <= 0:
        parser.error("--target-f0 must be > 0")
    if args.f0_min <= 0 or args.f0_max <= 0 or args.f0_min >= args.f0_max:
        parser.error("--f0-min and --f0-max must satisfy 0 < f0-min < f0-max")
    if args.target_sample_rate is not None and args.target_sample_rate <= 0:
        parser.error("--target-sample-rate must be > 0")
    if args.transient_threshold <= 0:
        parser.error("--transient-threshold must be > 0")
    if str(args.transient_mode) not in {"off", "reset", "hybrid", "wsola"}:
        parser.error("--transient-mode must be one of: off, reset, hybrid, wsola")
    if not (0.0 <= float(args.transient_sensitivity) <= 1.0):
        parser.error("--transient-sensitivity must be between 0.0 and 1.0")
    if float(args.transient_protect_ms) <= 0.0:
        parser.error("--transient-protect-ms must be > 0")
    if float(args.transient_crossfade_ms) < 0.0:
        parser.error("--transient-crossfade-ms must be >= 0")
    if str(args.stereo_mode) not in {"independent", "mid_side_lock", "ref_channel_lock"}:
        parser.error("--stereo-mode must be one of: independent, mid_side_lock, ref_channel_lock")
    if int(args.ref_channel) < 0:
        parser.error("--ref-channel must be >= 0")
    if not (0.0 <= float(args.coherence_strength) <= 1.0):
        parser.error("--coherence-strength must be between 0.0 and 1.0")
    if str(args.phase_engine) not in voc_core.PHASE_ENGINE_CHOICES:
        parser.error(f"--phase-engine must be one of: {', '.join(voc_core.PHASE_ENGINE_CHOICES)}")
    if not (0.0 <= args.ambient_phase_mix <= 1.0):
        parser.error("--ambient-phase-mix must be between 0.0 and 1.0")
    if not (0.0 <= args.onset_credit_pull <= 1.0):
        parser.error("--onset-credit-pull must be between 0.0 and 1.0")
    if args.onset_credit_max < 0.0:
        parser.error("--onset-credit-max must be >= 0.0")
    if args.formant_lifter < 0:
        parser.error("--formant-lifter must be >= 0")
    if not (0.0 <= args.formant_strength <= 1.0):
        parser.error("--formant-strength must be between 0.0 and 1.0")
    if args.formant_max_gain_db <= 0:
        parser.error("--formant-max-gain-db must be > 0")
    if args.fourier_sync_min_fft < 16:
        parser.error("--fourier-sync-min-fft must be >= 16")
    if args.fourier_sync_max_fft < args.fourier_sync_min_fft:
        parser.error("--fourier-sync-max-fft must be >= --fourier-sync-min-fft")
    if args.fourier_sync_smooth <= 0:
        parser.error("--fourier-sync-smooth must be > 0")
    if args.kaiser_beta < 0:
        parser.error("--kaiser-beta must be >= 0")
    if args.cuda_device < 0:
        parser.error("--cuda-device must be >= 0")
    if args.pitch_map is not None and str(args.pitch_map) != "-" and not args.pitch_map.exists():
        parser.error(f"Control-map file not found: {args.pitch_map}")
    if args.auto_profile_lookahead_seconds <= 0.0:
        parser.error("--auto-profile-lookahead-seconds must be > 0")
    if args.auto_segment_seconds < 0.0:
        parser.error("--auto-segment-seconds must be >= 0")
    if args.resume and args.checkpoint_dir is None:
        parser.error("--resume requires --checkpoint-dir")
    uses_segment_writes = (
        bool(dynamic_refs)
        or bool(args.pitch_map is not None)
        or bool(args.pitch_map_stdin)
        or float(args.auto_segment_seconds) > 0.0
    )
    if args.checkpoint_dir is not None and args.checkpoint_id is None and uses_segment_writes:
        if "-" in set(getattr(args, "inputs", []) or []):
            parser.error(
                "--checkpoint-dir with stdin audio input requires an explicit --checkpoint-id"
            )
        if (
            bool(getattr(args, "pitch_map_stdin", False))
            or str(getattr(args, "pitch_map", "")) == "-"
        ):
            parser.error(
                "--checkpoint-dir with stdin control maps requires an explicit --checkpoint-id"
            )
    if args.manifest_append and args.manifest_json is None:
        parser.error("--manifest-append requires --manifest-json")
    if str(args.quality_profile) not in voc_core.QUALITY_PROFILE_CHOICES:
        parser.error(
            f"--quality-profile must be one of: {', '.join(voc_core.QUALITY_PROFILE_CHOICES)}"
        )
    if str(args.preset) not in PRESET_CHOICES:
        parser.error(f"--preset must be one of: {', '.join(PRESET_CHOICES)}")
    if args.auto_profile and str(args.preset) not in {"none", "default"}:
        parser.error("Use either --auto-profile or --preset (not both together).")
    if args.multires_weights is not None and not args.multires_fusion:
        parser.error("--multires-weights requires --multires-fusion")

    if args.multires_fusion:
        try:
            ffts = voc_core.parse_int_list(args.multires_ffts, context="--multires-ffts")
        except ValueError as exc:
            parser.error(str(exc))
        if not ffts:
            parser.error("--multires-ffts must contain at least one size")
        if any(int(v) < 16 for v in ffts):
            parser.error("--multires-ffts entries must be >= 16")
        args._multires_ffts = [int(v) for v in ffts]

        if args.multires_weights is None:
            args._multires_weights = [1.0 for _ in args._multires_ffts]
        else:
            try:
                weights = voc_core.parse_numeric_list(
                    args.multires_weights,
                    context="--multires-weights",
                )
            except ValueError as exc:
                parser.error(str(exc))
            if len(weights) != len(args._multires_ffts):
                parser.error("--multires-weights count must equal --multires-ffts count")
            if any(float(w) < 0.0 for w in weights):
                parser.error("--multires-weights entries must be non-negative")
            if not any(float(w) > 0.0 for w in weights):
                parser.error("--multires-weights must contain at least one positive value")
            args._multires_weights = [float(w) for w in weights]
    else:
        args._multires_ffts = [int(args.n_fft)]
        args._multires_weights = [1.0]

    if str(args.transient_mode) == "reset":
        args.transient_preserve = True

    voc_core.validate_transform_available(args.transform, parser)
    voc_core.validate_mastering_args(args, parser)
    validate_output_policy_args(args, parser)
