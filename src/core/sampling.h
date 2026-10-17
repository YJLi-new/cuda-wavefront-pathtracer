// core/sampling.h - orthonormal bases and the warps used by the path tracer. Written from scratch.
#pragma once
#include "core/vec.h"
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CWP_TWO_PI 6.2831853f
#define CWP_INV_PI 0.31830989f

// orthonormal basis (t, b, n) from a unit normal; Duff et al., "Building an Orthonormal Basis, Revisited", JCGT 2017
HD void onb(float3 n, float3& t, float3& b) {
    float s = n.z >= 0.f ? 1.f : -1.f; float a = -1.f / (s + n.z); float c = n.x * n.y * a;
    t = mk3(1.f + s * n.x * n.x * a, s * c, -s * n.x); b = mk3(c, s + n.y * n.y * a, -n.y);
}

// cosine-weighted direction on the hemisphere around n (pdf = cos / pi)
HD float3 sampleCosineHemisphere(float3 n, float u0, float u1) {
    float r = sqrtf(u0), ph = CWP_TWO_PI * u1; float3 t, b; onb(n, t, b);
    return normalize(t * (r * cosf(ph)) + b * (r * sinf(ph)) + n * sqrtf(fmaxf(0.f, 1.f - u0)));
}

// uniform direction inside the cone of half-angle acos(cosMax) around the unit axis
HD float3 sampleUniformCone(float3 axis, float cosMax, float u0, float u1) {
    float cz = 1.f - u0 * (1.f - cosMax); float sz = sqrtf(fmaxf(0.f, 1.f - cz * cz)); float ph = CWP_TWO_PI * u1;
    float3 t, b; onb(axis, t, b);
    return normalize(axis * cz + t * (sz * cosf(ph)) + b * (sz * sinf(ph)));
}

// piecewise-linear lookup into an inverse-CDF table of n+1 entries, u in [0, 1)
HD float sampleTable(const float* tab, int n, float u) {
    float f = u * n; int k = (int)f; k = k < n - 1 ? k : n - 1; float t = f - k; return tab[k] * (1.f - t) + tab[k + 1] * t;
}

// host: inverse CDF of the 1D Blackman-Harris window of the given total width (in pixels), tabulated at n+1 points.
// Sampling a pixel offset from it gives filter importance sampling (the filter weight cancels the pdf).
inline void buildBlackmanHarrisTable(float* tab, int n, double width) {
    const int NS = 4096; double cdf[NS + 1]; cdf[0] = 0.0;
    for (int i = 0; i < NS; i++) { double t = (i + 0.5) / NS; double f = 0.35875 - 0.48829 * cos(2 * M_PI * t) + 0.14128 * cos(4 * M_PI * t) - 0.01168 * cos(6 * M_PI * t); cdf[i + 1] = cdf[i] + fmax(f, 0.0); }
    for (int i = 0; i <= NS; i++) cdf[i] /= cdf[NS];
    for (int k = 0; k <= n; k++) { double u = (double)k / n; int j = 0; while (j < NS - 1 && cdf[j + 1] < u) j++; double t = (cdf[j + 1] > cdf[j]) ? (u - cdf[j]) / (cdf[j + 1] - cdf[j]) : 0.0; tab[k] = (float)(((j + t) / NS - 0.5) * width); }
}
