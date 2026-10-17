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
    JsonValue J; std::string err;
    if (!parseJson(js, J, &err)) { fprintf(stderr, "warn: lighting file %s is not strict JSON (%s); env keys ignored\n", path.c_str(), err.c_str()); return true; }
    L.envMap = J.string("env_map", L.envMap); L.envStrength = (float)J.number("env_strength", L.envStrength); L.envRotDeg = (float)J.number("env_rotation_deg", L.envRotDeg);
    return true;
}

static float3 vec3Or(const JsonValue& o, const char* key, float3 def) { std::vector<double> v = o.numbers(key); return v.size() == 3 ? mk3((float)v[0], (float)v[1], (float)v[2]) : def; }

bool loadMaterialsFile(const std::string& path, Mesh& mesh) {
    std::string js = readTextFile(path); if (js.empty()) { fprintf(stderr, "materials file %s unreadable\n", path.c_str()); return false; }
    JsonValue J; std::string err;
    if (!parseJson(js, J, &err)) { fprintf(stderr, "materials file %s: %s\n", path.c_str(), err.c_str()); return false; }
    const JsonValue* arr = J.get("materials"); if (!arr || arr->type != JsonValue::Array) { fprintf(stderr, "materials file %s: no \"materials\" array\n", path.c_str()); return false; }
    std::vector<Material>& mats = mesh.mat; mats.clear();
    std::vector<bool> set; int pos = 0, byName = 0;
    for (const JsonValue& o : arr->arr) {
        Material m = defaultMaterial();
        m.baseColor = vec3Or(o, "base_color", m.baseColor);
        m.metallic = (float)o.number("metallic", m.metallic);
        m.roughness = (float)o.number("roughness", m.roughness);
        m.ior = (float)o.number("ior", m.ior);
        m.specularWeight = (float)o.number("specular_weight", m.specularWeight);
        m.specularColor = vec3Or(o, "specular_color", m.specularColor);
        m.transmission = (float)o.number("transmission", m.transmission);
        m.emission = vec3Or(o, "emission_color", mk3(1.f, 1.f, 1.f)) * (float)o.number("emission_strength", 0.0);
        int id = pos++;
        std::string nm = o.string("name", "");
        for (size_t j = 0; !nm.empty() && j < mesh.matName.size(); j++) if (mesh.matName[j] == nm) { id = (int)j; byName++; break; }
        if ((int)mats.size() <= id) { mats.resize(id + 1, lambertMaterial(mk3(0.8f, 0.8f, 0.8f))); set.resize(id + 1, false); }
        mats[id] = m; set[id] = true;
    }
    // ids not given by the file keep their Lambertian kd
    for (size_t j = 0; j < mats.size(); j++) if (!set[j]) mats[j] = lambertMaterial(j < mesh.kd.size() ? mesh.kd[j] : mk3(0.8f, 0.8f, 0.8f));
    fprintf(stderr, "materials: %d entries from %s (%d matched by name)\n", pos, path.c_str(), byName);
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

void loadCameraLens(const std::string& path, float& apertureR, float& focusD) {
    std::string js = readTextFile(path); double v;
    if (apertureR < 0.f && jsonNumber(js, "aperture_radius", v)) apertureR = (float)v;
    if (focusD < 0.f && jsonNumber(js, "focus_distance", v)) focusD = (float)v;
}
