// scene/instances.cpp - scene files with instances and animated transforms (written from scratch)
#include "scene/instances.h"
#include "io/json.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

static bool readXform(const std::vector<double>& v, Xform& X) {
    if (v.size() != 12 && v.size() != 16) return false;
    for (int i = 0; i < 12; i++) X.m[i] = (float)v[i];
    return true;
}
static float3 vec3Of(const JsonValue& o, const char* key, float3 def, bool allowScalar = false) {
    const JsonValue* v = o.get(key); if (!v) return def;
    if (allowScalar && v->type == JsonValue::Number) return mk3((float)v->num, (float)v->num, (float)v->num);
    std::vector<double> a = o.numbers(key); return a.size() == 3 ? mk3((float)a[0], (float)a[1], (float)a[2]) : def;
}

bool loadSceneFile(const std::string& path, SceneDesc& S) {
    std::string js = readTextFile(path); if (js.empty()) { fprintf(stderr, "scene file %s unreadable\n", path.c_str()); return false; }
    JsonValue J; std::string err;
    if (!parseJson(js, J, &err)) { fprintf(stderr, "scene file %s: %s\n", path.c_str(), err.c_str()); return false; }
    std::string dir = path.find_last_of('/') == std::string::npos ? std::string() : path.substr(0, path.find_last_of('/') + 1);
    auto rel = [&](const std::string& p) { return p.empty() || p[0] == '/' ? p : dir + p; };
    S.frames = (int)J.number("frames", 1);
    const JsonValue* ms = J.get("meshes"); const JsonValue* is = J.get("instances");
    if (!ms || ms->type != JsonValue::Array || !is || is->type != JsonValue::Array) { fprintf(stderr, "scene file %s: needs \"meshes\" and \"instances\" arrays\n", path.c_str()); return false; }
    for (const JsonValue& m : ms->arr) {
        MeshDesc d; d.file = rel(m.string("file", "")); d.materials = rel(m.string("materials", "")); d.loader = m.string("loader", "auto"); d.yup = (int)m.number("yup_to_zup", -1);
        if (d.file.empty()) { fprintf(stderr, "scene file: mesh without \"file\"\n"); return false; }
        S.meshes.push_back(d);
    }
    for (const JsonValue& o : is->arr) {
        InstanceDesc I; I.mesh = (int)o.number("mesh", 0);
        if (I.mesh < 0 || I.mesh >= (int)S.meshes.size()) { fprintf(stderr, "scene file: instance mesh %d out of range\n", I.mesh); return false; }
        if (o.get("transform")) { if (!readXform(o.numbers("transform"), I.matrix)) { fprintf(stderr, "scene file: transform needs 12 or 16 numbers\n"); return false; } I.hasMatrix = true; }
        if (const JsonValue* tf = o.get("transforms")) {
            for (const JsonValue& t : tf->arr) { std::vector<double> v; for (const JsonValue& x : t.arr) v.push_back(x.num); Xform X; if (!readXform(v, X)) { fprintf(stderr, "scene file: transforms entries need 12 or 16 numbers\n"); return false; } I.perFrame.push_back(X); }
        }
        I.translate = vec3Of(o, "translate", I.translate); I.translateVel = vec3Of(o, "translate_per_frame", I.translateVel);
        I.axis = vec3Of(o, "axis", I.axis); I.angleDeg = (float)o.number("angle_deg", 0.0); I.angleVel = (float)o.number("angle_deg_per_frame", 0.0);
        I.scale = vec3Of(o, "scale", I.scale, true); I.scaleVel = vec3Of(o, "scale_per_frame", I.scaleVel, true);
        S.instances.push_back(I);
    }
    return true;
}

Xform instanceXform(const InstanceDesc& I, int f) {
    if (!I.perFrame.empty()) return I.perFrame[f < (int)I.perFrame.size() ? f : I.perFrame.size() - 1];
    if (I.hasMatrix) return I.matrix;
    double t[3] = { I.translate.x + (double)f * I.translateVel.x, I.translate.y + (double)f * I.translateVel.y, I.translate.z + (double)f * I.translateVel.z };
    double s[3] = { I.scale.x + (double)f * I.scaleVel.x, I.scale.y + (double)f * I.scaleVel.y, I.scale.z + (double)f * I.scaleVel.z };
    double ang = (I.angleDeg + (double)f * I.angleVel) * 3.14159265358979323846 / 180.0;
    double ax = I.axis.x, ay = I.axis.y, az = I.axis.z, l = sqrt(ax * ax + ay * ay + az * az);
    if (l > 0) { ax /= l; ay /= l; az /= l; } else { ax = 0; ay = 0; az = 1; }
    double c = cos(ang), sn = sin(ang), C = 1 - c;
    double r[3][3] = { { c + ax * ax * C, ax * ay * C - az * sn, ax * az * C + ay * sn },
                       { ay * ax * C + az * sn, c + ay * ay * C, ay * az * C - ax * sn },
                       { az * ax * C - ay * sn, az * ay * C + ax * sn, c + az * az * C } };
    Xform X;
    for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) X.m[4 * i + j] = (float)(r[i][j] * s[j]); X.m[4 * i + 3] = (float)t[i]; }
    return X;
}

void invertXform(const Xform& M, Xform& inv, float& detSign) {
    const float* m = M.m;
    double a[3][3] = { { m[0], m[1], m[2] }, { m[4], m[5], m[6] }, { m[8], m[9], m[10] } };
    double det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    detSign = det < 0 ? -1.f : 1.f;
    double id = det != 0 ? 1.0 / det : 0.0, b[3][3];
    b[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * id; b[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * id; b[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * id;
    b[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * id; b[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * id; b[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * id;
    b[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * id; b[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * id; b[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * id;
    for (int i = 0; i < 3; i++) {
        double t = -(b[i][0] * m[3] + b[i][1] * m[7] + b[i][2] * m[11]);
        for (int j = 0; j < 3; j++) inv.m[4 * i + j] = (float)b[i][j];
        inv.m[4 * i + 3] = (float)t;
    }
}

void normalizeMaterials(Mesh& m) {
    int maxMat = 0; for (int t : m.fm) if (t > maxMat) maxMat = t;
    size_t nm = std::max(std::max(m.kd.size(), m.mat.size()), (size_t)maxMat + 1);
    std::vector<Material> mats;
    for (size_t j = 0; j < nm; j++) mats.push_back(j < m.mat.size() ? m.mat[j] : lambertMaterial(j < m.kd.size() ? m.kd[j] : mk3(0.8f, 0.8f, 0.8f)));
    m.mat = mats; m.kd.resize(nm); for (size_t j = 0; j < nm; j++) m.kd[j] = mats[j].baseColor;
    m.matName.resize(nm); m.mtex.resize(nm);
}

int appendMaterials(Mesh& dst, const Mesh& srcIn) {
    Mesh src = srcIn; src.v.clear(); src.f.clear(); src.n.clear(); src.uv.clear(); src.tan.clear(); src.fm = srcIn.fm;
    normalizeMaterials(src);
    int off = (int)dst.mat.size(), toff = (int)dst.tex.size();
    for (size_t j = 0; j < src.mat.size(); j++) {
        dst.mat.push_back(src.mat[j]); dst.kd.push_back(src.mat[j].baseColor); dst.matName.push_back(src.matName[j]);
        MaterialTextures t = src.mtex[j];
        int* ids[5] = { &t.baseColor, &t.emission, &t.normal, &t.metallic, &t.roughness };
        for (int* p : ids) if (*p >= 0) *p += toff;
        dst.mtex.push_back(t);
    }
    for (const TextureImage& T : src.tex) dst.tex.push_back(T);
    return off;
}

void appendGeometry(Mesh& dst, const Mesh& src, int matOffset, const Xform* X) {
    const size_t v0 = dst.v.size(), nv = src.v.size();
    Xform inv; float ds = 1.f; if (X) invertXform(*X, inv, ds);
    const bool wantN = !dst.n.empty() || src.hasN(), wantUV = !dst.uv.empty() || src.hasUV(), wantT = !dst.tan.empty() || src.hasTan();
    if (wantN) dst.n.resize(v0, mk3(0, 0, 0));
    if (wantUV) dst.uv.resize(v0, make_float2(0, 0));
    if (wantT) dst.tan.resize(v0, make_float4(0, 0, 0, 1));
    for (size_t i = 0; i < nv; i++) {
        float3 p = src.v[i]; dst.v.push_back(X ? xfPoint(*X, p) : p);
        if (wantN) { float3 n = src.hasN() ? src.n[i] : mk3(0, 0, 0); dst.n.push_back(X ? xfNormal(inv, n) : n); }
        if (wantUV) dst.uv.push_back(src.hasUV() ? src.uv[i] : make_float2(0, 0));
        if (wantT) { float4 t = src.hasTan() ? src.tan[i] : make_float4(0, 0, 0, 1); if (X) { float3 tv = xfVector(*X, xyz(t)); t = make_float4(tv.x, tv.y, tv.z, t.w * ds); } dst.tan.push_back(t); }
    }
    for (size_t t = 0; t < src.f.size(); t++) {
        int3 f = src.f[t]; dst.f.push_back(make_int3(f.x + (int)v0, f.y + (int)v0, f.z + (int)v0));
        dst.fm.push_back((src.fm.empty() ? 0 : src.fm[t]) + matOffset);
    }
}
