#!/usr/bin/env bash
# build.sh - thin wrapper around CMake. Needs the CUDA toolkit (nvcc) and CMake >= 3.22, not a GPU.
#   ARCH=sm_90 ./build.sh          # default: Hopper (H100 / H800). Use e.g. sm_80, sm_86, sm_89 for other GPUs.
#   VARIANTS=1 ./build.sh          # also build the register-cap variants used in the ablation study
#   ASSIMP=OFF ./build.sh          # without Assimp/stb (no network access needed; built-in OBJ/PLY loaders only)
#   JOBS=8 ./build.sh              # parallel compile jobs (default: all cores)
# Outputs: build/pt (recommended: __launch_bounds__(128,12), <= 40 registers/thread on sm_90)
#          build/pt_nocap, build/pt_mb16 (only with VARIANTS=1)
# Without CMake it falls back to one direct nvcc call per binary.
set -euo pipefail
cd "$(dirname "$0")"
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
ARCH=${ARCH:-sm_90}
mkdir -p build
echo "nvcc: ${NVCC}  arch: ${ARCH}"
if command -v cmake > /dev/null; then
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="${NVCC}" -DCMAKE_CUDA_ARCHITECTURES="${ARCH#sm_}" \
        -DCWP_VARIANTS=$([ "${VARIANTS:-0}" = "1" ] && echo ON || echo OFF) -DCWP_WITH_ASSIMP="${ASSIMP:-ON}" > build/cmake.log 2>&1 || { cat build/cmake.log; exit 1; }
  J=${JOBS:-$(nproc 2> /dev/null || echo 4)}
  cmake --build build -j "$J" --target pt > build/pt.log 2>&1 || { cat build/pt.log; exit 1; }
  cmake --build build -j "$J" > build/all.log 2>&1 || { cat build/all.log; exit 1; }
else
  SRC="src/app/pt.cpp src/render/wavefront.cu src/accel/hploc.cu $(ls src/scene/*.cpp src/io/*.cpp src/accel/*.cpp | tr '\n' ' ')"
  FLAGS="-O3 -std=c++17 -arch=${ARCH} --use_fast_math -lineinfo --ptxas-options=-v -Isrc"
  "${NVCC}" ${FLAGS} -DMINB=12 -o build/pt ${SRC} > build/pt.log 2>&1 || { cat build/pt.log; exit 1; }
  if [ "${VARIANTS:-0}" = "1" ]; then
    "${NVCC}" ${FLAGS} -o build/pt_nocap ${SRC} > build/pt_nocap.log 2>&1 || { cat build/pt_nocap.log; exit 1; }
    "${NVCC}" ${FLAGS} -DMINB=16 -o build/pt_mb16 ${SRC} > build/pt_mb16.log 2>&1 || { cat build/pt_mb16.log; exit 1; }
  fi
fi
# register usage of the fused trace+shade kernel (BVH2, Woop triangles) in build/pt
awk '/Compiling entry function .*k_trace_shadeILi2ELi1ELb0E/{f=1} f && /Used [0-9]+ registers/{match($0, /Used [0-9]+ registers/); print "k_trace_shade<bvh2,woop>: " substr($0, RSTART, RLENGTH); exit}' build/pt.log
ls -l build/pt*
