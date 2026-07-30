#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORMULA_PATH="${ROOT}/Formula/pvx.rb"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <version|tag> [formula-path]" >&2
  exit 2
fi

RAW_VERSION="$1"
if [[ $# -eq 2 ]]; then
  FORMULA_PATH="$2"
fi

VERSION="${RAW_VERSION#v}"
TAG="v${VERSION}"
URL="https://github.com/TheColby/pvx/archive/refs/tags/${TAG}.tar.gz"

TMP="$(mktemp -t pvx-homebrew-formula.XXXXXX.tar.gz)"
cleanup() {
  rm -f "$TMP"
}
trap cleanup EXIT

curl -L "$URL" -o "$TMP"
SHA="$(shasum -a 256 "$TMP" | awk '{print $1}')"

python3 - "$FORMULA_PATH" "$URL" "$SHA" "$VERSION" <<'PY'
from pathlib import Path
import re
import sys

formula_path = Path(sys.argv[1])
url = sys.argv[2]
sha = sys.argv[3]
version = sys.argv[4]

text = formula_path.read_text(encoding="utf-8")
replacements = (
    (r'(?m)^  url "[^"]+"$', f'  url "{url}"'),
    (r'(?m)^  sha256 "[0-9a-f]{64}"$', f'  sha256 "{sha}"'),
    (r'(?m)^  version "[^"]+"$', f'  version "{version}"'),
)
for pattern, replacement in replacements:
    text, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise SystemExit(f"could not update formula field matching {pattern!r}")
formula_path.write_text(text, encoding="utf-8")
PY

echo "Updated ${FORMULA_PATH} for ${TAG}"
echo "  url: ${URL}"
echo "  sha256: ${SHA}"
