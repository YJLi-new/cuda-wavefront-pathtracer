#!/usr/bin/env bash
# bench_synthetic.sh - run the built-in synthetic scene (ground + 7x7 bumpy spheres, ~2.0 M triangles)
# with the fastest configuration found in the ablation study, at 512x512 and 1280x720, 256 spp.
#   BATCH=256 scripts/bench_synthetic.sh    # samples per pixel in flight per wavefront pass (default 64)
# BATCH=64 needs ~7 GB of GPU memory at 720p; BATCH=256 needs ~29 GB and is faster on large GPUs.
# Prints one JSON line per run (ray counts, ms per frame, Mrays/s, memory) and writes out/*.png / *.pfm.
set -euo pipefail
cd "$(dirname "$0")/.."
BATCH=${BATCH:-64}
mkdir -p out
for res in "512 512" "1280 720"; do
  set -- $res
  ./build/pt --synthetic 7 --width "$1" --height "$2" --spp 256 --bounces 4 \
             --variant bvh2 --tri woop --fuse 1 --batch "$BATCH" --reps 2 \
             --out "out/synthetic_${1}x${2}"
done
