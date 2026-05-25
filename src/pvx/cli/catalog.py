#!/usr/bin/env python3

"""Static command catalog for the unified pvx CLI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    entrypoint: str
    summary: str
    aliases: tuple[str, ...] = ()


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="voc",
        entrypoint="pvx.core.voc:main",
        summary="General-purpose phase-vocoder time/pitch processing",
        aliases=("pvxvoc", "vocoder", "timepitch"),
    ),
    ToolSpec(
        name="freeze",
        entrypoint="pvx.cli.pvxfreeze:main",
        summary="Freeze a spectral frame into a sustained texture",
        aliases=("pvxfreeze",),
    ),
    ToolSpec(
        name="harmonize",
        entrypoint="pvx.cli.pvxharmonize:main",
        summary="Generate harmony voices from one source",
        aliases=("pvxharmonize", "harm"),
    ),
    ToolSpec(
        name="conform",
        entrypoint="pvx.cli.pvxconform:main",
        summary="Apply CSV segment map for time/pitch conformity",
        aliases=("pvxconform",),
    ),
    ToolSpec(
        name="morph",
        entrypoint="pvx.cli.pvxmorph:main",
        summary="Morph two sources in the STFT domain",
        aliases=("pvxmorph",),
    ),
    ToolSpec(
        name="warp",
        entrypoint="pvx.cli.pvxwarp:main",
        summary="Apply variable stretch map from CSV",
        aliases=("pvxwarp",),
    ),
    ToolSpec(
        name="formant",
        entrypoint="pvx.cli.pvxformant:main",
        summary="Formant shift/preserve processing",
        aliases=("pvxformant",),
    ),
    ToolSpec(
        name="transient",
        entrypoint="pvx.cli.pvxtransient:main",
        summary="Transient-aware time/pitch processing",
        aliases=("pvxtransient",),
    ),
    ToolSpec(
        name="unison",
        entrypoint="pvx.cli.pvxunison:main",
        summary="Unison thickening and width enhancement",
        aliases=("pvxunison",),
    ),
    ToolSpec(
        name="denoise",
        entrypoint="pvx.cli.pvxdenoise:main",
        summary="Spectral denoise",
        aliases=("pvxdenoise",),
    ),
    ToolSpec(
        name="deverb",
        entrypoint="pvx.cli.pvxdeverb:main",
        summary="Dereverb spectral tail reduction",
        aliases=("pvxdeverb", "dereverb"),
    ),
    ToolSpec(
        name="retune",
        entrypoint="pvx.cli.pvxretune:main",
        summary="Monophonic pitch retune to scale/root",
        aliases=("pvxretune",),
    ),
    ToolSpec(
        name="layer",
        entrypoint="pvx.cli.pvxlayer:main",
        summary="Split/process harmonic and percussive layers",
        aliases=("pvxlayer",),
    ),
    ToolSpec(
        name="pitch-track",
        entrypoint="pvx.cli.hps_pitch_track:main",
        summary="Track f0 and emit control-map CSV",
        aliases=("hps-pitch-track", "hps", "track"),
    ),
    ToolSpec(
        name="analysis",
        entrypoint="pvx.cli.pvxanalysis:main",
        summary="Create/inspect reusable PVXAN analysis artifacts",
        aliases=("pvxanalysis",),
    ),
    ToolSpec(
        name="response",
        entrypoint="pvx.cli.pvxresponse:main",
        summary="Create/inspect reusable PVXRF response artifacts",
        aliases=("pvxresponse",),
    ),
    ToolSpec(
        name="envelope",
        entrypoint="pvx.cli.pvxenvelope:main",
        summary="Generate control-rate envelope maps (CSV/JSON)",
        aliases=("pvxenvelope", "lfo"),
    ),
    ToolSpec(
        name="reshape",
        entrypoint="pvx.cli.pvxreshape:main",
        summary="Reshape control-rate maps for pvx routing",
        aliases=("pvxreshape",),
    ),
    ToolSpec(
        name="filter",
        entrypoint="pvx.cli.pvxfilter:main",
        summary="Response-driven spectral filtering (PVC-inspired)",
        aliases=("pvxfilter",),
    ),
    ToolSpec(
        name="tvfilter",
        entrypoint="pvx.cli.pvxtvfilter:main",
        summary="Time-varying response filter with control maps",
        aliases=("pvxtvfilter",),
    ),
    ToolSpec(
        name="noisefilter",
        entrypoint="pvx.cli.pvxnoisefilter:main",
        summary="Response-referenced noise filtering",
        aliases=("pvxnoisefilter",),
    ),
    ToolSpec(
        name="bandamp",
        entrypoint="pvx.cli.pvxbandamp:main",
        summary="Response-peak band amplification",
        aliases=("pvxbandamp",),
    ),
    ToolSpec(
        name="spec-compander",
        entrypoint="pvx.cli.pvxspeccompander:main",
        summary="Response-referenced spectral compander",
        aliases=("pvxspeccompander", "speccompander"),
    ),
    ToolSpec(
        name="ring",
        entrypoint="pvx.cli.pvxring:main",
        summary="Ring modulation operator",
        aliases=("pvxring",),
    ),
    ToolSpec(
        name="ringfilter",
        entrypoint="pvx.cli.pvxringfilter:main",
        summary="Ring modulation plus resonator filtering",
        aliases=("pvxringfilter",),
    ),
    ToolSpec(
        name="ringtvfilter",
        entrypoint="pvx.cli.pvxringtvfilter:main",
        summary="Time-varying ring modulation plus resonator filtering",
        aliases=("pvxringtvfilter",),
    ),
    ToolSpec(
        name="chordmapper",
        entrypoint="pvx.cli.pvxchordmapper:main",
        summary="Chord-aware spectral mapping",
        aliases=("pvxchordmapper",),
    ),
    ToolSpec(
        name="inharmonator",
        entrypoint="pvx.cli.pvxinharmonator:main",
        summary="Inharmonic spectral warping",
        aliases=("pvxinharmonator",),
    ),
    ToolSpec(
        name="trajectory-reverb",
        entrypoint="pvx.cli.pvxtrajectoryreverb:main",
        summary="Mono-to-multichannel trajectory convolution reverb",
        aliases=("pvxtrajectoryreverb", "trajreverb", "spatial-reverb"),
    ),
    ToolSpec(
        name="noise",
        entrypoint="pvx.cli.pvxnoise:main",
        summary="Add synthetic or background noise at a controlled SNR",
        aliases=("pvxnoise", "addnoise"),
    ),
    ToolSpec(
        name="rir",
        entrypoint="pvx.cli.pvxrir:main",
        summary="Simulate room acoustics via impulse response convolution",
        aliases=("pvxrir", "reverb-sim", "room-sim"),
    ),
    ToolSpec(
        name="codec",
        entrypoint="pvx.cli.pvxcodec:main",
        summary="Simulate lossy codec artifacts (MP3, telephone, VoIP)",
        aliases=("pvxcodec", "codec-sim"),
    ),
    ToolSpec(
        name="specaugment",
        entrypoint="pvx.cli.pvxspecaugment:main",
        summary="Apply SpecAugment frequency and time masking (Park et al. 2019)",
        aliases=("pvxspecaugment", "spec-augment"),
    ),
    ToolSpec(
        name="gain",
        entrypoint="pvx.cli.pvxgain:main",
        summary="Random gain perturbation or loudness normalization",
        aliases=("pvxgain",),
    ),
)


EXAMPLE_COMMANDS: dict[str, tuple[str, str]] = {
    "doctor": ("Environment diagnostics", "pvx doctor"),
    "quickstart": (
        "Minimal launch/demo sequence",
        "pvx quickstart input.wav --output output.wav",
    ),
    "safe": (
        "Quality-first conservative voc wrapper",
        "pvx safe input.wav --material mix --output output_safe.wav",
    ),
    "transforms": (
        "Transform availability and recommendation guide",
        "pvx transforms",
    ),
    "smoke": (
        "Synthetic end-to-end smoke render",
        "pvx smoke --output smoke_out.wav",
    ),
    "augment": (
        "Deterministic dataset augmentation for AI research",
        "pvx augment data/*.wav --output-dir aug_out --variants-per-input 4 --intent asr_robust --seed 1337",
    ),
    "augment-manifest": (
        "Validate/merge augmentation manifests",
        "pvx augment-manifest validate aug_out/augment_manifest.jsonl",
    ),
    "basic": ("Basic stretch", "pvx voc input.wav --stretch 1.20 --output output.wav"),
    "speech": (
        "Slow speech for review",
        "pvx voc speech.wav --preset vocal_studio --stretch 1.30 --output speech_slow.wav",
    ),
    "vocal": (
        "Vocal pitch/formant correction",
        "pvx voc vocal.wav --preset vocal_studio --pitch -2 --output vocal_fixed.wav",
    ),
    "retune": (
        "Scale retune",
        "pvx retune vocal.wav --root C --scale major --strength 0.85 --output vocal_retuned.wav",
    ),
    "freeze": (
        "Freeze a spectral moment",
        "pvx freeze hit.wav --freeze-time 0.2 --duration 10 --output hit_freeze.wav",
    ),
    "ambient": (
        "Extreme ambient stretch",
        "pvx voc one_shot.wav --preset extreme_ambient --target-duration 600 --output one_shot_ambient.wav",
    ),
    "drums": (
        "Transient-safe drums",
        "pvx voc drums.wav --preset drums_safe --stretch 1.25 --output drums_safe.wav",
    ),
    "morph": (
        "Source morph",
        "pvx morph source_a.wav source_b.wav --alpha controls/alpha_curve.csv --interp linear --output morph_traj.wav",
    ),
    "map": (
        "Time/pitch map conform",
        "pvx conform source.wav --map map_conform.csv --output source_conformed.wav",
    ),
    "microtonal": (
        "Microtonal pitch ratio",
        "pvx voc input.wav --stretch 1.0 --ratio 3/2 --output input_perfect_fifth.wav",
    ),
    "pipe": (
        "Short one-line pipe",
        "pvx voc input.wav --stretch 1.2 --stdout | pvx deverb - --strength 0.3 --output output.wav",
    ),
    "pipeline": (
        "Pitch-follow pipeline",
        "pvx pitch-track guide.wav --emit pitch_to_stretch --output - | pvx voc target.wav --control-stdin --output followed.wav",
    ),
    "follow": (
        "Single-command sidechain follow",
        "pvx follow guide.wav target.wav --output followed.wav --emit pitch_to_stretch --pitch-conf-min 0.75",
    ),
    "follow-feature": (
        "Feature-driven follow (MFCC + MPEG-7 spectral flux)",
        "pvx follow guide.wav target.wav --feature-set all --mfcc-count 13 --emit pitch_map --stretch 1.0 --route pitch_ratio=affine(mfcc_01,0.002,1.0) --route pitch_ratio=clip(pitch_ratio,0.5,2.0) --route stretch=affine(mpeg7_spectral_flux,0.05,1.0) --route stretch=clip(stretch,0.85,1.6) --output followed_feature.wav",
    ),
    "follow-formant": (
        "Feature-driven follow (formant and onset)",
        "pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 --route pitch_ratio=affine(formant_f1_hz,0.0016,0.2) --route pitch_ratio=clip(pitch_ratio,0.7,1.5) --route stretch=affine(onset_norm,-0.35,1.2) --route stretch=clip(stretch,0.8,1.3) --output followed_formant_onset.wav",
    ),
    "follow-noise-aware": (
        "Feature-driven follow (noise-aware hiss/hum control)",
        "pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 --route stretch=affine(hiss_ratio,-0.6,1.2) --route stretch=clip(stretch,0.8,1.2) --route pitch_ratio=affine(hum_60_ratio,-0.4,1.15) --route pitch_ratio=clip(pitch_ratio,0.9,1.2) --output followed_noise_aware.wav",
    ),
    "analysis": (
        "Create reusable analysis artifact",
        "pvx analysis create input.wav --output input.pvxan.npz --n-fft 4096 --hop-size 256",
    ),
    "response": (
        "Derive reusable response artifact",
        "pvx response create input.pvxan.npz --output input.pvxrf.npz --method median --normalize peak",
    ),
    "envelope": (
        "Generate a stretch envelope/LFO control map",
        "pvx lfo --wave triangle --duration 8 --frequency-hz 0.5 --center 1.0 --amplitude 0.2 --key stretch --output stretch_lfo.csv",
    ),
    "reshape": (
        "Reshape and resample a control map",
        "pvx reshape stretch_env.csv --key stretch --operation resample --rate 50 --interp polynomial --order 5 --output stretch_env_dense.csv",
    ),
    "filter": (
        "Response-driven static filter",
        "pvx filter input.wav --response input.pvxrf.npz --response-mix 1.0 --output filtered.wav",
    ),
    "tvfilter": (
        "Time-varying response filter",
        "pvx tvfilter input.wav --response input.pvxrf.npz --tv-map mix_map.csv --tv-interp linear --output tvfiltered.wav",
    ),
    "ringfilter": (
        "Ring + resonator filter",
        "pvx ringfilter input.wav --frequency-hz 55 --resonance-hz 1200 --resonance-q 9 --output ringfilter.wav",
    ),
    "chordmapper": (
        "Chord-aware harmonic mapping",
        "pvx chordmapper input.wav --root-hz 220 --chord minor --strength 0.75 --output chordmapped.wav",
    ),
    "inharmonator": (
        "Inharmonic spectral warping",
        "pvx inharmonator input.wav --inharmonic-f0-hz 220 --inharmonicity 0.0002 --inharmonic-mix 1.0 --output inharm.wav",
    ),
    "trajectory-reverb": (
        "Mono through multichannel room IR with A->B movement",
        "pvx trajectory-reverb source.wav --ir room_4ch.wav --coord-system cartesian --start -1,0,1 --end 1,0,1 --output flythrough.wav",
    ),
    "chain": (
        "Managed multi-stage chain",
        'pvx chain input.wav --pipeline "voc --stretch 1.2 | formant --mode preserve" --output output_chain.wav',
    ),
    "stream": (
        "Chunked stream wrapper over pvx voc",
        "pvx stream input.wav --output output_stream.wav --chunk-seconds 0.2 --time-stretch 2.0 --preset extreme_ambient",
    ),
    "stretch-budget": (
        "Estimate max safe stretch from a file and disk budget",
        "pvx stretch-budget input.wav --disk-budget 20GB --bit-depth 16 --requested-stretch 1000000",
    ),
}


FOLLOW_EXAMPLE_COMMANDS: dict[str, tuple[str, str]] = {
    "basic": (
        "Pitch-to-stretch sidechain",
        "pvx follow guide.wav target.wav --emit pitch_to_stretch --pitch-conf-min 0.75 --output followed.wav",
    ),
    "pitch": (
        "Pitch-map follow with fixed stretch",
        "pvx follow guide.wav target.wav --emit pitch_map --stretch 1.0 --output followed_pitch.wav",
    ),
    "mfcc_flux": (
        "MFCC + MPEG-7 flux dual control",
        "pvx follow guide.wav target.wav --feature-set all --mfcc-count 13 --emit pitch_map --stretch 1.0 --route pitch_ratio=affine(mfcc_01,0.002,1.0) --route pitch_ratio=clip(pitch_ratio,0.5,2.0) --route stretch=affine(mpeg7_spectral_flux,0.05,1.0) --route stretch=clip(stretch,0.85,1.6) --output followed_mfcc_flux.wav",
    ),
    "formant_onset": (
        "Formant + onset dual control",
        "pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 --route pitch_ratio=affine(formant_f1_hz,0.0016,0.2) --route pitch_ratio=clip(pitch_ratio,0.7,1.5) --route stretch=affine(onset_norm,-0.35,1.2) --route stretch=clip(stretch,0.8,1.3) --output followed_formant_onset.wav",
    ),
    "noise_aware": (
        "Noise-aware hiss/hum routing",
        "pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 --route stretch=affine(hiss_ratio,-0.6,1.2) --route stretch=clip(stretch,0.8,1.2) --route pitch_ratio=affine(hum_60_ratio,-0.4,1.15) --route pitch_ratio=clip(pitch_ratio,0.9,1.2) --output followed_noise_aware.wav",
    ),
}


FOLLOW_EXAMPLE_CHOICES: tuple[str, ...] = ("all", *tuple(FOLLOW_EXAMPLE_COMMANDS.keys()))

_AUDIO_EXTENSIONS: set[str] = {
    ".wav",
    ".flac",
    ".aiff",
    ".aif",
    ".ogg",
    ".oga",
    ".caf",
    ".mp3",
    ".m4a",
    ".aac",
    ".wma",
}

_CHAIN_TOOL_ALLOWLIST: set[str] = {
    "voc",
    "freeze",
    "harmonize",
    "conform",
    "warp",
    "formant",
    "transient",
    "unison",
    "denoise",
    "deverb",
    "retune",
    "layer",
    "filter",
    "tvfilter",
    "noisefilter",
    "bandamp",
    "spec-compander",
    "ring",
    "ringfilter",
    "ringtvfilter",
    "chordmapper",
    "inharmonator",
    "trajectory-reverb",
}
_CHAIN_STAGE_FORBIDDEN_FLAGS: set[str] = {
    "-o",
    "--out",
    "--output",
    "--output-dir",
    "--stdout",
}

_LUCKY_SUPPORTED_TOOLS: set[str] = set(_CHAIN_TOOL_ALLOWLIST) | {"morph"}
_LUCKY_PRESETS: tuple[str, ...] = (
    "default",
    "vocal_studio",
    "drums_safe",
    "extreme_ambient",
    "stereo_coherent",
)
_LUCKY_WINDOWS: tuple[str, ...] = ("hann", "hamming", "blackmanharris", "kaiser", "tukey")


def build_tool_index(specs: tuple[ToolSpec, ...]) -> dict[str, ToolSpec]:
    out: dict[str, ToolSpec] = {}
    for spec in specs:
        out[spec.name] = spec
        for alias in spec.aliases:
            out[alias] = spec
    return out
