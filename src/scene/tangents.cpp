// scene/tangents.cpp (written from scratch)
#include "scene/tangents.h"
#include "core/sampling.h"
#include <cmath>

void computeTangents(const std::vector<float3>& p, const std::vector<float3>& n, const std::vector<float2>& uv,
                     const std::vector<int3>& f, std::vector<float4>& tan) {
    const size_t nv = p.size();
    std::vector<float3> t1(nv, mk3(0, 0, 0)), t2(nv, mk3(0, 0, 0)), fn(n.size() == nv ? 0 : nv, mk3(0, 0, 0));
    for (const int3& t : f) {
        int id[3] = { t.x, t.y, t.z };
        float3 e1 = p[id[1]] - p[id[0]], e2 = p[id[2]] - p[id[0]];
        float du1 = uv[id[1]].x - uv[id[0]].x, dv1 = uv[id[1]].y - uv[id[0]].y, du2 = uv[id[2]].x - uv[id[0]].x, dv2 = uv[id[2]].y - uv[id[0]].y;
        float det = du1 * dv2 - du2 * dv1;
        if (!fn.empty()) { float3 c = cross(e1, e2); for (int k = 0; k < 3; k++) fn[id[k]] += c; }
        if (fabsf(det) < 1e-20f) continue;
        float r = 1.f / det;
        float3 s = (e1 * dv2 - e2 * dv1) * r, u = (e2 * du1 - e1 * du2) * r;
        for (int k = 0; k < 3; k++) { t1[id[k]] += s; t2[id[k]] += u; }
    }
    tan.resize(nv);
    for (size_t i = 0; i < nv; i++) {
        float3 nn = normalize(fn.empty() ? n[i] : fn[i]);
        float3 t = t1[i] - nn * dot(nn, t1[i]);
        if (dot(t, t) < 1e-24f) { float3 b; onb(nn, t, b); }   // no usable uv gradient: any tangent
        t = normalize(t);
        float w = dot(cross(nn, t), t2[i]) < 0.f ? -1.f : 1.f;
        tan[i] = make_float4(t.x, t.y, t.z, w);
    }
}
