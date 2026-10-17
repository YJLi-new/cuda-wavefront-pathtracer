// app/display.cu - viewer display transform on the GPU: accumulated radiance sums -> 8-bit RGBA for the screen.
// Mode 0 reproduces the batch renderer's PNG transform (io/image.cpp tonemapSRGB8: clip to [0,1], sRGB OETF,
// x * 255 + 0.5 truncated) with IEEE division and no FMA contraction, after scaling by 2^exposure (exposure 0 = scale
// 1, exact). Modes 1 and 2 are display-only tone curves: Reinhard x / (1 + x) and the ACES filmic fit of Narkowicz (2015).
// Compiled without --use_fast_math.
#include "app/display.h"
#include <cuda_runtime.h>

__device__ __forceinline__ unsigned char encodeSRGB8(float x) {
    x = fminf(fmaxf(x, 0.f), 1.f);
    x = x <= 0.0031308f ? __fmul_rn(12.92f, x) : __fsub_rn(__fmul_rn(1.055f, powf(x, 1.f / 2.4f)), 0.055f);
    return (unsigned char)__fadd_rn(__fmul_rn(x, 255.f), 0.5f);
}

__device__ __forceinline__ float toneCurve(float x, int mode) {
    if (mode == 1) return x / (1.f + x);
    if (mode == 2) { x *= 0.6f; return (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f); }
    return x;
}

__global__ void k_display(const float4* __restrict__ acc, uchar4* __restrict__ out, int n, float spp, float scale, int mode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float4 a = acc[i];
    float r = a.x / spp, g = a.y / spp, b = a.z / spp;   // as the batch renderer: sum / spp
    if (scale != 1.f) { r = __fmul_rn(r, scale); g = __fmul_rn(g, scale); b = __fmul_rn(b, scale); }
    if (mode != 0) { r = toneCurve(fmaxf(r, 0.f), mode); g = toneCurve(fmaxf(g, 0.f), mode); b = toneCurve(fmaxf(b, 0.f), mode); }
    out[i] = make_uchar4(encodeSRGB8(r), encodeSRGB8(g), encodeSRGB8(b), 255);
}

void launchDisplay(const float4* accum, uchar4* out, int n, int spp, float exposureEV, int mode) {
    float scale = exposureEV == 0.f ? 1.f : exp2f(exposureEV);
    k_display<<<(n + 255) / 256, 256>>>(accum, out, n, (float)(spp > 0 ? spp : 1), scale, mode);
}
