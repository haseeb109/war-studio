#!/bin/bash
# One-time setup on macOS (Apple Silicon). Safe to run again.
set -e
cd "$(dirname "$0")"

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is missing. Install it from https://brew.sh then run this again."
  exit 1
fi

echo "==> Installing ffmpeg, node (for HyperFrames), uv"
brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg
brew list node   >/dev/null 2>&1 || brew install node
command -v uv >/dev/null 2>&1 || brew install uv

echo "==> Python packages (first time downloads PyTorch, a few minutes)"
uv sync --frozen

echo "==> Downloading the CLIP model once (~600 MB)"
uv run --frozen python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k'); print('CLIP ready')"

echo "==> Warming up HyperFrames"
npx -y hyperframes --version >/dev/null 2>&1 && echo "HyperFrames ready" || echo "HyperFrames not available (PNG overlays will be used)"

echo "==> Built-in sound effects"
uv run --frozen python -c "from warstudio.sfx import ensure_defaults; ensure_defaults(); print('SFX ready')"

chmod +x WarStudio.command
echo
echo "Done. Double-click WarStudio.command to start."
