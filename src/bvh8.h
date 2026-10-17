// bvh8.h - compressed 8-wide BVH (CWBVH-style: per-node origin + per-axis power-of-two scale, 8-bit quantized child
// boxes, octant-based child slots, imask + popcount child addressing, per-node contiguous leaf triangles). 80 B nodes.
// Also: Woop unit-triangle transform packing. Written from scratch.
#pragma once
#include "bvh.h"
#include <cmath>

struct BVH8Flat {
    std::vector<float4> nodes;   // 5 float4 per node
    std::vector<int> order;      // triangle order (index into mesh triangles)
    const BVHBuilder* B = nullptr;
    int maxLeafTris = 0;

    static int expo(float ext, float lo, float hi) {
        if (!(ext > 0.f)) return -100;
        int e = (int)ceilf(log2f(ext / 255.f));
        if (e < -100) e = -100;
        while (lo + 255.f * ldexpf(1.f, e) < hi) e++;
        return e;
    }
    static unsigned pack4(const int* q, int base) { return (unsigned)q[base] | ((unsigned)q[base + 1] << 8) | ((unsigned)q[base + 2] << 16) | ((unsigned)q[base + 3] << 24); }

    void build(BVHBuilder& Bb) {
        B = &Bb; nodes.clear(); order.clear(); order.reserve(Bb.idx.size());
        int root = 0;
        if (Bb.nodes[0].leaf()) { BuildNode r; r.box = Bb.nodes[0].box; r.left = 0; r.right = -1; Bb.nodes.push_back(r); root = (int)Bb.nodes.size() - 1; }
        nodes.resize(5);
        emit(root, 0);
    }
    void emit(int bn, int out) {
        const std::vector<BuildNode>& N = B->nodes;
        std::vector<int> ch; if (N[bn].left >= 0) ch.push_back(N[bn].left); if (N[bn].right >= 0) ch.push_back(N[bn].right);
        while ((int)ch.size() < 8) {
            int best = -1; float bestA = -1;
            for (size_t j = 0; j < ch.size(); j++) if (!N[ch[j]].leaf()) { float a = N[ch[j]].box.area(); if (a > bestA) { bestA = a; best = (int)j; } }
            if (best < 0) break;
            int nb = ch[best]; ch[best] = N[nb].left; ch.push_back(N[nb].right);
        }
        // slot assignment by octant of child center relative to node center (bit set = negative side)
        AABB box = N[bn].box; float3 c = (box.lo + box.hi) * 0.5f;
        int slotChild[8]; for (int s = 0; s < 8; s++) slotChild[s] = -1;
        for (size_t j = 0; j < ch.size(); j++) {
            const AABB& cb = N[ch[j]].box; float3 cc = (cb.lo + cb.hi) * 0.5f;
            int want = (cc.x < c.x ? 1 : 0) | (cc.y < c.y ? 2 : 0) | (cc.z < c.z ? 4 : 0);
            int pick = -1, bestD = 99;
            for (int s = 0; s < 8; s++) if (slotChild[s] < 0) { int d = __builtin_popcount(s ^ want); if (d < bestD) { bestD = d; pick = s; } }
            slotChild[pick] = ch[j];
        }
        // quantization frame
        float3 ext = box.extent();
        int ex = expo(ext.x, box.lo.x, box.hi.x), ey = expo(ext.y, box.lo.y, box.hi.y), ez = expo(ext.z, box.lo.z, box.hi.z);
        float sx = ldexpf(1.f, ex), sy = ldexpf(1.f, ey), sz = ldexpf(1.f, ez);
        int qlox[8], qloy[8], qloz[8], qhix[8], qhiy[8], qhiz[8], meta[8];
        int triBase = (int)order.size(); unsigned imask = 0; int nInt = 0;
        for (int s = 0; s < 8; s++) {
            qlox[s] = qloy[s] = qloz[s] = qhix[s] = qhiy[s] = qhiz[s] = 0; meta[s] = 0;
            int cn = slotChild[s]; if (cn < 0) continue;
            const AABB& cb = N[cn].box;
            auto q = [](float v, float o, float sc, bool up) { float f = (v - o) / sc; int i = up ? (int)ceilf(f) : (int)floorf(f); return i < 0 ? 0 : i > 255 ? 255 : i; };
            qlox[s] = q(cb.lo.x, box.lo.x, sx, false); qhix[s] = q(cb.hi.x, box.lo.x, sx, true);
            qloy[s] = q(cb.lo.y, box.lo.y, sy, false); qhiy[s] = q(cb.hi.y, box.lo.y, sy, true);
            qloz[s] = q(cb.lo.z, box.lo.z, sz, false); qhiz[s] = q(cb.hi.z, box.lo.z, sz, true);
            if (N[cn].leaf()) {
                int off = (int)order.size() - triBase; int cnt = N[cn].count;
                for (int t = 0; t < cnt; t++) order.push_back(B->idx[N[cn].first + t]);
                meta[s] = 0x80 | ((cnt - 1) << 5) | off; // cnt 1..4, off 0..31
                if (cnt > maxLeafTris) maxLeafTris = cnt;
            } else { meta[s] = 0x40; imask |= 1u << s; nInt++; }
        }
        int childBase = (int)nodes.size() / 5;
        nodes.resize(nodes.size() + 5 * (size_t)nInt);
        float4* n = &nodes[out * 5];
        unsigned ew = (unsigned)(ex + 127) | ((unsigned)(ey + 127) << 8) | ((unsigned)(ez + 127) << 16) | (imask << 24);
        float ewf, cbf, tbf, m0f, m1f; unsigned m0 = pack4(meta, 0), m1 = pack4(meta, 4);
        memcpy(&ewf, &ew, 4); memcpy(&cbf, &childBase, 4); memcpy(&tbf, &triBase, 4); memcpy(&m0f, &m0, 4); memcpy(&m1f, &m1, 4);
        n[0] = make_float4(box.lo.x, box.lo.y, box.lo.z, ewf);
        n[1] = make_float4(cbf, tbf, m0f, m1f);
        unsigned u[12] = { pack4(qlox, 0), pack4(qlox, 4), pack4(qloy, 0), pack4(qloy, 4), pack4(qloz, 0), pack4(qloz, 4), pack4(qhix, 0), pack4(qhix, 4), pack4(qhiy, 0), pack4(qhiy, 4), pack4(qhiz, 0), pack4(qhiz, 4) };
        float uf[12]; memcpy(uf, u, sizeof uf);
        n[2] = make_float4(uf[0], uf[1], uf[2], uf[3]); n[3] = make_float4(uf[4], uf[5], uf[6], uf[7]); n[4] = make_float4(uf[8], uf[9], uf[10], uf[11]);
        int ord = 0;
        for (int s = 0; s < 8; s++) if (slotChild[s] >= 0 && !N[slotChild[s]].leaf()) { emit(slotChild[s], childBase + ord); ord++; }
    }
};

// Woop unit-triangle transform: rows of M = inverse([e1 e2 n]) with translation, stored (row0,row1,row2) as 3 float4.
static void packWoop(const Mesh& m, const std::vector<int>& order, std::vector<float4>& w) {
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
