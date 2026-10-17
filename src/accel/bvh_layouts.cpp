// accel/bvh_layouts.cpp - BVH2 / BVH4 / compressed BVH8 GPU layouts (written from scratch)
#include "accel/bvh_layouts.h"
#include <cmath>
#include <cstring>

int BVH2Flat::emit(const BVHBuilder& B, int bn) {
    const BuildNode& nd = B.nodes[bn];
    int id = (int)nodes.size() / 4; nodes.resize(nodes.size() + 4);
    int ch[2] = { nd.left, nd.right };
    int c[2], k[2]; AABB bx[2];
    for (int j = 0; j < 2; j++) {
        if (ch[j] < 0) { c[j] = -1; k[j] = 0; bx[j].lo = bx[j].hi = mk3(EMPTY_BOX, EMPTY_BOX, EMPTY_BOX); continue; }
        const BuildNode& cn = B.nodes[ch[j]]; bx[j] = cn.box;
        if (cn.leaf()) { c[j] = cn.first; k[j] = cn.count; } else { k[j] = 0; c[j] = emit(B, ch[j]); }
    }
    float4* n = &nodes[id * 4];
    n[0] = make_float4(bx[0].lo.x, bx[0].hi.x, bx[0].lo.y, bx[0].hi.y);
    n[1] = make_float4(bx[1].lo.x, bx[1].hi.x, bx[1].lo.y, bx[1].hi.y);
    n[2] = make_float4(bx[0].lo.z, bx[0].hi.z, bx[1].lo.z, bx[1].hi.z);
    float m0, m1, m2, m3; memcpy(&m0, &c[0], 4); memcpy(&m1, &c[1], 4); memcpy(&m2, &k[0], 4); memcpy(&m3, &k[1], 4);
    n[3] = make_float4(m0, m1, m2, m3);
    return id;
}
void BVH2Flat::build(BVHBuilder& B) {
    nodes.clear();
    if (B.nodes[0].leaf()) { // root leaf: wrap in an internal node with one empty child
        BuildNode root; root.box = B.nodes[0].box; root.left = 0; root.right = -1;
        B.nodes.push_back(root); emit(B, (int)B.nodes.size() - 1);
    } else emit(B, 0);
}

int BVH4Flat::emit(const BVHBuilder& B, int bn) {
    int id = (int)nodes.size() / 8; nodes.resize(nodes.size() + 8);
    std::vector<int> ch; ch.push_back(B.nodes[bn].left); ch.push_back(B.nodes[bn].right);
    while ((int)ch.size() < 4) {
        int best = -1; float bestA = -1;
        for (size_t j = 0; j < ch.size(); j++) if (ch[j] >= 0 && !B.nodes[ch[j]].leaf()) { float a = B.nodes[ch[j]].box.area(); if (a > bestA) { bestA = a; best = (int)j; } }
        if (best < 0) break;
        int nb = ch[best]; ch[best] = B.nodes[nb].left; ch.push_back(B.nodes[nb].right);
    }
    int c[4], k[4]; AABB bx[4];
    for (int j = 0; j < 4; j++) {
        if (j >= (int)ch.size() || ch[j] < 0) { c[j] = -1; k[j] = 0; bx[j].lo = bx[j].hi = mk3(EMPTY_BOX, EMPTY_BOX, EMPTY_BOX); continue; }
        const BuildNode& cn = B.nodes[ch[j]]; bx[j] = cn.box;
        if (cn.leaf()) { c[j] = cn.first; k[j] = cn.count; } else { k[j] = 0; c[j] = emit(B, ch[j]); }
    }
    float4* n = &nodes[id * 8];
    n[0] = make_float4(bx[0].lo.x, bx[1].lo.x, bx[2].lo.x, bx[3].lo.x);
    n[1] = make_float4(bx[0].hi.x, bx[1].hi.x, bx[2].hi.x, bx[3].hi.x);
    n[2] = make_float4(bx[0].lo.y, bx[1].lo.y, bx[2].lo.y, bx[3].lo.y);
    n[3] = make_float4(bx[0].hi.y, bx[1].hi.y, bx[2].hi.y, bx[3].hi.y);
    n[4] = make_float4(bx[0].lo.z, bx[1].lo.z, bx[2].lo.z, bx[3].lo.z);
    n[5] = make_float4(bx[0].hi.z, bx[1].hi.z, bx[2].hi.z, bx[3].hi.z);
    float cf[4], kf[4]; for (int j = 0; j < 4; j++) { memcpy(&cf[j], &c[j], 4); memcpy(&kf[j], &k[j], 4); }
    n[6] = make_float4(cf[0], cf[1], cf[2], cf[3]);
    n[7] = make_float4(kf[0], kf[1], kf[2], kf[3]);
    return id;
}
void BVH4Flat::build(BVHBuilder& B) {
    nodes.clear();
    if (B.nodes[0].leaf()) { BuildNode root; root.box = B.nodes[0].box; root.left = 0; root.right = -1; B.nodes.push_back(root); emit(B, (int)B.nodes.size() - 1); }
    else emit(B, 0);
}

int BVH8Flat::expo(float ext, float lo, float hi) {
    if (!(ext > 0.f)) return -100;
    int e = (int)ceilf(log2f(ext / 255.f));
    if (e < -100) e = -100;
    while (lo + 255.f * ldexpf(1.f, e) < hi) e++;
    return e;
}
unsigned BVH8Flat::pack4(const int* q, int base) { return (unsigned)q[base] | ((unsigned)q[base + 1] << 8) | ((unsigned)q[base + 2] << 16) | ((unsigned)q[base + 3] << 24); }

void BVH8Flat::build(BVHBuilder& Bb) {
    B = &Bb; nodes.clear(); order.clear(); order.reserve(Bb.idx.size());
    int root = 0;
    if (Bb.nodes[0].leaf()) { BuildNode r; r.box = Bb.nodes[0].box; r.left = 0; r.right = -1; Bb.nodes.push_back(r); root = (int)Bb.nodes.size() - 1; }
    nodes.resize(5);
    emit(root, 0);
}
void BVH8Flat::emit(int bn, int out) {
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
