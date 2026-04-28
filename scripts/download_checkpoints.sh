#!/usr/bin/env bash
set -euo pipefail

REPO="umass-lass/ReVo"
DEST=".checkpoints"

hf download "$REPO" h264/h264_rgb.pth       --local-dir "$DEST"
hf download "$REPO" h264/h264_depth.pth     --local-dir "$DEST"
hf download "$REPO" h265/h265_rgb.pth       --local-dir "$DEST"
hf download "$REPO" h265/h265_depth.pth     --local-dir "$DEST"
hf download "$REPO" dcvcrt/dcvcrt_rgb.pth   --local-dir "$DEST"
hf download "$REPO" dcvcrt/dcvcrt_depth.pth --local-dir "$DEST"

echo "All checkpoints saved to $DEST/"
