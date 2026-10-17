// accel/tri_pack.cpp - triangle packing (written from scratch)
#include "accel/tri_pack.h"
#include <cmath>
#include <cstring>

void packTriangles(const Mesh& m, const std::vector<int>& order, std::vector<float4>& tris, std::vector<float4>& nrm) {
    size_t n = order.size(); tris.resize(3 * n); if (m.hasN()) nrm.resize(3 * n);
    for (size_t k = 0; k < n; k++) {
        int t = order[k]; int3 f = m.f[t];
        float3 v0 = m.v[f.x], e1 = m.v[f.y] - v0, e2 = m.v[f.z] - v0;
        int mat = m.fm.empty() ? 0 : m.fm[t];
        float ti, mi; memcpy(&ti, &t, 4); memcpy(&mi, &mat, 4);
        tris[3 * k + 0] = mk4(v0, ti); tris[3 * k + 1] = mk4(e1, mi); tris[3 * k + 2] = mk4(e2, 0.f);
        if (m.hasN()) { nrm[3 * k + 0] = mk4(m.n[f.x], 0); nrm[3 * k + 1] = mk4(m.n[f.y], 0); nrm[3 * k + 2] = mk4(m.n[f.z], 0); }
    }
}

void packWoop(const Mesh& m, const std::vector<int>& order, std::vector<float4>& w) {
    size_t n = order.size(); w.resize(3 * n);
    for (size_t k = 0; k < n; k++) {
        int3 f = m.f[order[k]]; float3 v0 = m.v[f.x], e1 = m.v[f.y] - v0, e2 = m.v[f.z] - v0, nn = cross(e1, e2);
        // A = [e1 e2 nn] columns; inverse via adjugate
        double a[3][3] = { { e1.x, e2.x, nn.x }, { e1.y, e2.y, nn.y }, { e1.z, e2.z, nn.z } };
        double det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
        double inv[3][3];
        if (fabs(det) < 1e-300) { for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) inv[i][j] = 0; }
        else {
            double id = 1.0 / det;
            inv[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * id; inv[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * id; inv[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * id;
            inv[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * id; inv[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * id; inv[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * id;
            inv[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * id; inv[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * id; inv[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * id;
        }
        for (int r = 0; r < 3; r++) {
            double t = -(inv[r][0] * v0.x + inv[r][1] * v0.y + inv[r][2] * v0.z);
            w[3 * k + r] = make_float4((float)inv[r][0], (float)inv[r][1], (float)inv[r][2], (float)t);
        }
    }
}

void packAttributes(const Mesh& m, const std::vector<int>& order, std::vector<float4>& uvs, std::vector<float4>& tans) {
    size_t n = order.size(); uvs.resize(2 * n); if (m.hasTan()) tans.resize(3 * n);
    for (size_t k = 0; k < n; k++) {
        int3 f = m.f[order[k]];
        float2 a = m.uv[f.x], b = m.uv[f.y], c = m.uv[f.z];
        uvs[2 * k] = make_float4(a.x, a.y, b.x, b.y); uvs[2 * k + 1] = make_float4(c.x, c.y, 0.f, 0.f);
        if (m.hasTan()) { tans[3 * k] = m.tan[f.x]; tans[3 * k + 1] = m.tan[f.y]; tans[3 * k + 2] = m.tan[f.z]; }
    }
}
