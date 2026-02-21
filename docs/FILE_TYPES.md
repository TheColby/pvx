# Supported File Types

`pvx` uses `python-soundfile` (`soundfile`) backed by `libsndfile` for audio input/output (I/O).  
This means some format support is runtime/platform dependent.

## I/O Behavior by Category

| Category | Used by | File types |
| --- | --- | --- |
| Audio input (file path) | `pvxvoc`, `pvxfreeze`, `pvxharmonize`, `pvxmorph`, `pvxretune`, `pvxdenoise`, `pvxdeverb`, and other audio CLIs | Any audio container supported by the active `soundfile/libsndfile` build (see full table below) |
| Audio input (stdin, `-`) | CLIs that accept `-` as input | Any container `soundfile` can decode from byte stream |
| Audio output (file path) | Audio CLIs with `--output-format` or inferred output extension | Any format `soundfile` can write in the active build |
| Audio output (stdout, `--stdout`) | Stream/pipeline mode in `pvxvoc` and common CLI helpers | Explicitly supported: `wav`, `flac`, `aiff`/`aif`, `ogg`/`oga`, `caf` |
| Control maps | `pvxvoc --pitch-map`, `pvxwarp --map`, `pvxconform --map` | `csv` |
| Time-varying per-parameter control signals | `pvxvoc --stretch stretch.csv`, `pvxvoc --pitch-shift-ratio pitch.json`, `pvxvoc --n-fft nfft.csv`, `pvx stream ... --stretch stretch.csv` | `csv`, `json` |
| Pitch tracking map output | `HPS-pitch-track.py` | `csv` |
| Render manifest | `pvxvoc --manifest-json` | `json` |
| Documentation output | docs generators | `html`, `pdf` |

## Full Audio Container List (Current Build)

The following formats are reported by the current local `soundfile/libsndfile` runtime:

| Extension | Format |
| --- | --- |
| `aiff` | AIFF (Apple/SGI) |
| `au` | AU (Sun/NeXT) |
| `avr` | AVR (Audio Visual Research) |
| `caf` | CAF (Apple Core Audio File) |
| `flac` | FLAC (Free Lossless Audio Codec) |
| `htk` | HTK (HMM Tool Kit) |
| `ircam` | SF (Berkeley/IRCAM/CARL) |
| `mat4` | MAT4 (GNU Octave 2.0 / Matlab 4.2) |
| `mat5` | MAT5 (GNU Octave 2.1 / Matlab 5.0) |
| `mp3` | MPEG-1/2 Audio |
| `mpc2k` | MPC (Akai MPC 2k) |
| `nist` | WAV (NIST Sphere) |
| `ogg` | OGG (OGG Container format) |
| `paf` | PAF (Ensoniq PARIS) |
| `pvf` | PVF (Portable Voice Format) |
| `raw` | RAW (header-less) |
| `rf64` | RF64 (RIFF 64) |
| `sd2` | SD2 (Sound Designer II) |
| `sds` | SDS (Midi Sample Dump Standard) |
| `svx` | IFF (Amiga IFF/SVX8/SV16) |
| `voc` | VOC (Creative Labs) |
| `w64` | W64 (SoundFoundry WAVE 64) |
| `wav` | WAV (Microsoft) |
| `wavex` | WAVEX (Microsoft) |
| `wve` | WVE (Psion Series 3) |
| `xi` | XI (FastTracker 2) |

## Verify on Your Machine

Use this command to print the exact formats available in your environment:

```bash
python3 - <<'PY'
import soundfile as sf
for ext, name in sorted(sf.available_formats().items()):
    print(f"{ext.lower():<8} {name}")
PY
```
