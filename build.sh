#!/usr/bin/env bash
# build.sh - compile the path tracer. Needs the CUDA toolkit (nvcc), not a GPU.
#   ARCH=sm_90 ./build.sh          # default: Hopper (H100 / H800). Use e.g. sm_80, sm_86, sm_89 for other GPUs.
#   VARIANTS=1 ./build.sh          # also build the register-cap variants used in the ablation study
# Outputs: build/pt (recommended: __launch_bounds__(128,12), <= 40 registers/thread on sm_90)
#          build/pt_nocap, build/pt_mb16 (only with VARIANTS=1)
set -euo pipefail
cd "$(dirname "$0")"
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
ARCH=${ARCH:-sm_90}
FLAGS="-O3 -std=c++17 -arch=${ARCH} --use_fast_math -lineinfo --ptxas-options=-v"
mkdir -p build
echo "nvcc: ${NVCC}  arch: ${ARCH}"
"${NVCC}" ${FLAGS} -DMINB=12 -o build/pt src/main.cu > build/pt.log 2>&1 || { cat build/pt.log; exit 1; }
if [ "${VARIANTS:-0}" = "1" ]; then
  "${NVCC}" ${FLAGS} -o build/pt_nocap src/main.cu > build/pt_nocap.log 2>&1 || { cat build/pt_nocap.log; exit 1; }
  "${NVCC}" ${FLAGS} -DMINB=16 -o build/pt_mb16 src/main.cu > build/pt_mb16.log 2>&1 || { cat build/pt_mb16.log; exit 1; }
fi
# register usage of the fused trace+shade kernel (BVH2, Woop triangles)
awk '/Compiling entry function .*k_trace_shadeILi2ELi1ELb0E/{f=1} f && /Used [0-9]+ registers/{match($0, /Used [0-9]+ registers/); print "k_trace_shade<bvh2,woop>: " substr($0, RSTART, RLENGTH); exit}' build/pt.log
ls -l build/pt*
