// io/scene_files.cpp (written from scratch)
#include "io/scene_files.h"
#include "io/json.h"
#include <cstdio>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

bool loadLightingFile(const std::string& path, Lighting& L, Mesh& m) {
    std::string js = readTextFile(path); if (js.empty()) { fprintf(stderr, "lighting file %s unreadable\n", path.c_str()); return false; }
    std::vector<double> v = jsonNumbers(js, "sun_dir_to_light_world_zup"); if (v.size() == 3) L.sunTo = mk3((float)v[0], (float)v[1], (float)v[2]);
    double x; if (jsonNumber(js, "sun_irradiance_W_m2", x)) L.sunE = mk3((float)x, (float)x, (float)x);
    if (jsonNumber(js, "sun_angle_rad", x)) L.sunAngleDeg = (float)(x * 180.0 / M_PI);
    v = jsonNumbers(js, "sky_radiance_rgb_linear"); if (v.size() == 3) L.sky = mk3((float)v[0], (float)v[1], (float)v[2]);
    v = jsonNumbers(js, "albedo_rgb_linear_by_material_index");
    if (v.size() >= 3) { m.kd.clear(); for (size_t j = 0; j + 2 < v.size(); j += 3) m.kd.push_back(mk3((float)v[j], (float)v[j + 1], (float)v[j + 2])); }
    int maxMat = 0; for (int t : m.fm) if (t > maxMat) maxMat = t;
    while ((int)m.kd.size() <= maxMat) m.kd.push_back(mk3(0.8f, 0.8f, 0.8f));
    return true;
}

bool loadCameraFile(const std::string& path, const std::string& key, int frame, int w, int h, float& fx, float& fy, float& hfov, float c2w[16]) {
    std::string js = readTextFile(path); if (js.empty()) { fprintf(stderr, "camera file %s unreadable\n", path.c_str()); return false; }
    double v;
    if (fx == 0 && jsonNumber(js, "fx", v)) {
        double fw = 0, fh = 0; jsonNumber(js, "width", fw); jsonNumber(js, "height", fh);
        if ((int)fw == w && (int)fh == h) { fx = (float)v; jsonNumber(js, "fy", v); fy = (float)v; }
        else if (hfov == 0 && jsonNumber(js, "hfov_deg", v)) hfov = (float)v;
    }
    if (!jsonMatrix(js, key, frame, c2w)) { fprintf(stderr, "no matrix %s in camera file\n", key.c_str()); return false; }
    return true;
}
