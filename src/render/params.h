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
