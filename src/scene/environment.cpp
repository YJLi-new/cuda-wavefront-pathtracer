// scene/environment.cpp (written from scratch)
#include "scene/environment.h"
#include "io/hdr_image.h"
#include <cmath>
#include <cstdio>

bool buildEnvMap(const std::string& path, float strength, EnvMapHost& E) {
    std::vector<float> rgb;
    if (!loadHDR(path, E.w, E.h, rgb)) return false;
    const int w = E.w, h = E.h;
    E.texels.resize((size_t)w * h); E.marg.assign(h + 1, 0.f); E.cond.assign((size_t)h * (w + 1), 0.f);
    std::vector<double> rowSum(h, 0.0), rowCdf(w + 1);
    double mean[3] = { 0, 0, 0 };
    for (int y = 0; y < h; y++) {
        double st = sin(3.14159265358979323846 * (y + 0.5) / h);
        rowCdf[0] = 0.0;
        for (int x = 0; x < w; x++) {
            const float* c = &rgb[((size_t)y * w + x) * 3];
            float3 L = mk3(c[0], c[1], c[2]) * strength;
            E.texels[(size_t)y * w + x] = make_float4(L.x, L.y, L.z, 0.f);
            for (int k = 0; k < 3; k++) mean[k] += c[k];
            double lum = 0.2126 * L.x + 0.7152 * L.y + 0.0722 * L.z;
            rowCdf[x + 1] = rowCdf[x] + (lum > 0 ? lum : 0.0) * st;
        }
        rowSum[y] = rowCdf[w];
        float* cr = &E.cond[(size_t)y * (w + 1)];
        for (int x = 0; x <= w; x++) cr[x] = rowSum[y] > 0 ? (float)(rowCdf[x] / rowSum[y]) : (float)x / w;
        cr[w] = 1.f;
    }
    double tot = 0; std::vector<double> mc(h + 1, 0.0);
    for (int y = 0; y < h; y++) { mc[y + 1] = mc[y] + rowSum[y]; }
    tot = mc[h];
    for (int y = 0; y <= h; y++) E.marg[y] = tot > 0 ? (float)(mc[y] / tot) : (float)y / h;
    E.marg[h] = 1.f;
    size_t n = (size_t)w * h;
    fprintf(stderr, "env: %s %dx%d, mean (%.5f %.5f %.5f) x strength %.3f\n", path.c_str(), w, h, mean[0] / n, mean[1] / n, mean[2] / n, strength);
    return true;
}
