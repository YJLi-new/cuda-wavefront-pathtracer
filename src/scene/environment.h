// scene/environment.h - equirectangular environment map with a piecewise-constant importance-sampling
// distribution (marginal CDF over rows, conditional CDF per row, texel weight = luminance * sin(theta)).
// Orientation (Blender/Cycles convention, verified against a Cycles render of the same HDRI): row 0 = +z (up); the centre
// column faces +x and u increases toward -y, i.e. u = 0.5 - atan2(y, x) / (2 pi); v = acos(z) / pi.
#pragma once
#include "core/vec.h"
#include <string>
#include <vector>

struct EnvMapHost {
    int w = 0, h = 0;
    std::vector<float4> texels;   // radiance * strength, row-major, row 0 = top
    std::vector<float> marg;      // h + 1 entries
    std::vector<float> cond;      // h rows of (w + 1) entries
};
bool buildEnvMap(const std::string& path, float strength, EnvMapHost& E);
