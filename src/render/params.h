// render/params.h - per-frame launch parameters (copied to __constant__ memory) shared by host and device
#pragma once
#include "scene/camera.h"

static constexpr int kFilterTable = 64;    // entries of the pixel-filter inverse CDF (kFilterTable + 1 points)
static constexpr int kSortBins = 32768;    // bins of the experimental bounce-ray sort (--sort)

struct Params {
    Cam cam;
    float3 sunTo, sunE, sky, bmin, binv;   // sun direction / irradiance, sky radiance; scene box origin and 16/extent (sort keys)
    float sunCos;                          // cos of the sun's angular radius
    int maxB; unsigned seed;               // max bounce depth, RNG seed
    float eps, near;                       // ray origin offset (world units), camera near clip
    int hasN, filterBH;                    // shading normals present, Blackman-Harris pixel filter (else 1 px box)
    float ftab[kFilterTable + 1];
};

#include "scene/material.h"

// light-sampling strategies for emitters and the environment map (the sun is always light-sampled)
enum LightMode { kLightMIS = 0, kLightNEE = 1, kLightBSDF = 2 };

// parameters of the full material integrator (render/wavefront.cu, k_trace_shade_full); device pointers
struct FullParams {
    const Material* mats;
    const float* triLightPdf;      // per triangle (BVH leaf order): selection probability / area, 0 if not emissive
    const int* emTri; const float* emCdf; int nEm;   // emissive triangles (leaf order) and their power CDF (nEm + 1)
    const float4* envTex; const float* envMarg; const float* envCond; int envW, envH; float envRot;
    float pSun, pTri, pEnv;        // light-type selection probabilities for next-event estimation
    int lightMode, rr, rrDepth, hasEnv, gdEnergy;
};
