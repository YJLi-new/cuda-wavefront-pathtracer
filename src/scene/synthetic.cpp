// scene/synthetic.cpp - procedural test scene (written from scratch)
#include "scene/synthetic.h"
#include "scene/camera.h"
#include <cmath>

void makeSynthetic(Mesh& m, int gridN, int nu, int nv, int gs) {
    m.kd.clear();
    m.kd.push_back(mk3(0.5f, 0.5f, 0.5f));          // 0 ground
    m.kd.push_back(mk3(0.8f, 0.3f, 0.2f));          // 1..4 sphere colors
    m.kd.push_back(mk3(0.2f, 0.6f, 0.3f));
    m.kd.push_back(mk3(0.25f, 0.35f, 0.8f));
    m.kd.push_back(mk3(0.8f, 0.8f, 0.8f));
    const float half = 6.0f;
    // ground
    int base = (int)m.v.size();
    for (int j = 0; j <= gs; j++) for (int i = 0; i <= gs; i++) {
        float x = -half + 2 * half * i / gs, y = -half + 2 * half * j / gs;
        float z = 0.05f * sinf(1.3f * x) * cosf(1.7f * y);
        m.v.push_back(mk3(x, y, z));
    }
    for (int j = 0; j < gs; j++) for (int i = 0; i < gs; i++) {
        int a = base + j * (gs + 1) + i, b = a + 1, c = a + gs + 1, d = c + 1;
        m.f.push_back(make_int3(a, b, d)); m.fm.push_back(0);
        m.f.push_back(make_int3(a, d, c)); m.fm.push_back(0);
    }
    // spheres
    float pitch = 2 * half / (gridN + 1);
    float r = pitch * 0.33f;
    for (int sj = 0; sj < gridN; sj++) for (int si = 0; si < gridN; si++) {
        int k = sj * gridN + si;
        float3 c = mk3(-half + pitch * (si + 1), -half + pitch * (sj + 1), r * (1.1f + 0.3f * ((k * 7) % 5) / 4.0f));
        float rr = r * (0.7f + 0.3f * ((k * 3) % 4) / 3.0f);
        int vb = (int)m.v.size();
        for (int j = 0; j <= nv; j++) for (int i = 0; i < nu; i++) {
            float th = 3.14159265f * j / nv, ph = 2 * 3.14159265f * i / nu;
            float bump = 1.0f + 0.12f * sinf(6 * th + k) * cosf(5 * ph + 0.5f * k);
            float3 d = mk3(sinf(th) * cosf(ph), sinf(th) * sinf(ph), cosf(th));
            m.v.push_back(c + d * (rr * bump));
        }
        int mat = 1 + (k % 4);
        for (int j = 0; j < nv; j++) for (int i = 0; i < nu; i++) {
            int i1 = (i + 1) % nu;
            int a = vb + j * nu + i, b = vb + j * nu + i1, cc = vb + (j + 1) * nu + i, d = vb + (j + 1) * nu + i1;
            if (j > 0) { m.f.push_back(make_int3(a, cc, b)); m.fm.push_back(mat); }
            if (j < nv - 1) { m.f.push_back(make_int3(b, cc, d)); m.fm.push_back(mat); }
        }
    }
}

void syntheticCamera(float c2w[16]) { lookAtC2W(mk3(0.f, -9.5f, 6.5f), mk3(0.f, 0.f, 0.4f), mk3(0, 0, 1), c2w); }
