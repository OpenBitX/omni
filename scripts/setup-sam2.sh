#!/usr/bin/env bash
# Sets up server-v2/.venv with SAM2-tiny + FastAPI deps.
# Idempotent: re-run whenever requirements drift.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
V2_DIR="$ROOT/server-v2"
VENV="$V2_DIR/.venv"
CHECKPOINTS_DIR="$V2_DIR/checkpoints"
CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
CHECKPOINT_FILE="$CHECKPOINTS_DIR/sam2.1_hiera_tiny.pt"

CUDA=0
for arg in "$@"; do
  case "$arg" in
    --cuda) CUDA=1 ;;
    -h|--help)
      echo "Usage: $0 [--cuda]"
      echo "  --cuda   Install the CUDA torch wheel (Linux). Default is CPU; macOS MPS is auto-detected at runtime."
      exit 0
      ;;
  esac
done

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "✗ python3 not found. Install Python 3.10+ first." >&2
  exit 1
fi

# Require Python 3.10+ because SAM2 uses match statements and newer typing.
PY_MAJOR_MINOR=$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
case "$PY_MAJOR_MINOR" in
  3.10|3.11|3.12|3.13) ;;
  *)
    echo "✗ Python $PY_MAJOR_MINOR found — SAM2 needs 3.10+. Set PYTHON=/path/to/python3.12 and re-run." >&2
    exit 1
    ;;
esac

echo "→ using $PY ($PY_MAJOR_MINOR)"

if [ ! -d "$VENV" ]; then
  echo "→ creating venv at $VENV"
  "$PY" -m venv "$VENV"
else
  echo "→ venv exists at $VENV"
fi

PIP="$VENV/bin/pip"
PYV="$VENV/bin/python"

"$PIP" install --upgrade pip wheel setuptools >/dev/null

if [ "$CUDA" = "1" ]; then
  echo "→ installing torch (CUDA)"
  "$PIP" install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
  echo "→ installing torch (CPU / macOS MPS)"
  "$PIP" install --upgrade torch torchvision
fi

echo "→ installing FastAPI deps from requirements.txt"
"$PIP" install --upgrade -r "$V2_DIR/requirements.txt"

echo "→ installing SAM2 from GitHub"
"$PIP" install --upgrade "git+https://github.com/facebookresearch/sam2.git"

mkdir -p "$CHECKPOINTS_DIR"
if [ -f "$CHECKPOINT_FILE" ]; then
  echo "→ checkpoint already present: $CHECKPOINT_FILE"
else
  echo "→ downloading SAM2.1-hiera-tiny checkpoint (~150 MB)"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --progress-bar -o "$CHECKPOINT_FILE" "$CHECKPOINT_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$CHECKPOINT_FILE" "$CHECKPOINT_URL"
  else
    echo "✗ neither curl nor wget found — download manually from $CHECKPOINT_URL into $CHECKPOINTS_DIR/" >&2
    exit 1
  fi
fi

"$PYV" - <<'PY'
import importlib, sys
for mod in ("torch", "sam2", "fastapi", "uvicorn", "cv2"):
    try:
        importlib.import_module(mod)
        print(f"  ok {mod}")
    except Exception as e:
        print(f"  !! {mod}: {e}", file=sys.stderr)
        sys.exit(1)
print("✓ server-v2 environment ready — run `npm run server:v2`")
PY
