// bvh.h - CPU binned-SAH BVH2 builder + GPU layouts (BVH2 "children boxes in parent", 64 B nodes;
//         BVH4 collapsed from the BVH2, 128 B nodes). Written from scratch.
#pragma once
#include "vec.h"
#include "mesh.h"
#include <vector>
#include <algorithm>
#include <cfloat>
#include <cstdint>

struct AABB {
    float3 lo, hi;
    AABB() : lo(mk3(FLT_MAX, FLT_MAX, FLT_MAX)), hi(mk3(-FLT_MAX, -FLT_MAX, -FLT_MAX)) {}
    void grow(float3 p) { lo = vmin(lo, p); hi = vmax(hi, p); }
    void grow(const AABB& b) { lo = vmin(lo, b.lo); hi = vmax(hi, b.hi); }
    float area() const { if (hi.x < lo.x) return 0.f; float3 d = hi - lo; return 2.f * (d.x * d.y + d.y * d.z + d.z * d.x); }
    float3 extent() const { return hi - lo; }
};

struct BuildNode { AABB box; int left = -1, right = -1, first = 0, count = 0; bool leaf() const { return count > 0; } };

struct BVHBuilder {
    std::vector<AABB> tb; std::vector<float3> cent; std::vector<int> idx; std::vector<BuildNode> nodes;
    int leafMax = 4; int maxDepth = 0;
    static const int NB = 16;

    void build(const Mesh& m) {
        size_t n = m.f.size(); tb.resize(n); cent.resize(n); idx.resize(n);
        for (size_t i = 0; i < n; i++) {
            AABB b; b.grow(m.v[m.f[i].x]); b.grow(m.v[m.f[i].y]); b.grow(m.v[m.f[i].z]);
            tb[i] = b; cent[i] = (b.lo + b.hi) * 0.5f; idx[i] = (int)i;
        }
        nodes.reserve(2 * n / leafMax + 16);
        rec(0, (int)n, 0);
    }
    int rec(int begin, int end, int depth) {
        int id = (int)nodes.size(); nodes.push_back(BuildNode());
        if (depth > maxDepth) maxDepth = depth;
        AABB box, cb; int n = end - begin;
        for (int i = begin; i < end; i++) { box.grow(tb[idx[i]]); cb.grow(cent[idx[i]]); }
        nodes[id].box = box;
        if (n <= leafMax || depth >= 62) { nodes[id].first = begin; nodes[id].count = n; return id; }
        // binned SAH
        float bestCost = FLT_MAX; int bestAxis = -1, bestBin = -1;
        float3 ext = cb.extent();
        for (int axis = 0; axis < 3; axis++) {
            float lo = axis == 0 ? cb.lo.x : axis == 1 ? cb.lo.y : cb.lo.z;
            float e = axis == 0 ? ext.x : axis == 1 ? ext.y : ext.z;
            if (e <= 1e-12f) continue;
            float scale = NB / e;
            AABB bb[NB]; int bc[NB] = {0};
            for (int i = begin; i < end; i++) {
                float c = axis == 0 ? cent[idx[i]].x : axis == 1 ? cent[idx[i]].y : cent[idx[i]].z;
                int b = (int)((c - lo) * scale); b = b < 0 ? 0 : b > NB - 1 ? NB - 1 : b;
                bb[b].grow(tb[idx[i]]); bc[b]++;
            }
            float rA[NB]; int rC[NB]; AABB acc; int cnt = 0;
            for (int b = NB - 1; b > 0; b--) { acc.grow(bb[b]); cnt += bc[b]; rA[b] = acc.area(); rC[b] = cnt; }
            AABB lacc; int lcnt = 0;
            for (int b = 0; b < NB - 1; b++) {
                lacc.grow(bb[b]); lcnt += bc[b];
                if (lcnt == 0 || rC[b + 1] == 0) continue;
                float cost = lacc.area() * lcnt + rA[b + 1] * rC[b + 1];
                if (cost < bestCost) { bestCost = cost; bestAxis = axis; bestBin = b; }
            }
        }
        int mid;
        float leafCost = box.area() * n;
        if (bestAxis < 0 || (n <= leafMax && bestCost + box.area() >= leafCost)) {
            if (bestAxis < 0) {
                if (n <= leafMax) { nodes[id].first = begin; nodes[id].count = n; return id; }
                // all centroids coincide: median split on arbitrary order
                mid = (begin + end) / 2;
            } else { nodes[id].first = begin; nodes[id].count = n; return id; }
        } else {
            float lo = bestAxis == 0 ? cb.lo.x : bestAxis == 1 ? cb.lo.y : cb.lo.z;
            float e = bestAxis == 0 ? ext.x : bestAxis == 1 ? ext.y : ext.z;
            float scale = NB / e;
            int* p = std::partition(idx.data() + begin, idx.data() + end, [&](int t) {
                float c = bestAxis == 0 ? cent[t].x : bestAxis == 1 ? cent[t].y : cent[t].z;
                int b = (int)((c - lo) * scale); b = b < 0 ? 0 : b > NB - 1 ? NB - 1 : b;
                return b <= bestBin; });
            mid = (int)(p - idx.data());
            if (mid == begin || mid == end) mid = (begin + end) / 2;
        }
        int l = rec(begin, mid, depth + 1);
        int r = rec(mid, end, depth + 1);
        nodes[id].left = l; nodes[id].right = r;
        return id;
    }
};

// GPU triangle layout: 3 float4 per triangle in BVH order: (v0, origTri), (e1, mat), (e2, 0)
static void packTriangles(const Mesh& m, const std::vector<int>& order, std::vector<float4>& tris, std::vector<float4>& nrm) {
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

static const float EMPTY_BOX = 3e37f;

// BVH2 GPU node (64 B): n0xy=(c0.lo.x,c0.hi.x,c0.lo.y,c0.hi.y) n1xy=(same for c1) nz=(c0.lo.z,c0.hi.z,c1.lo.z,c1.hi.z)
// meta=(c0, c1, cnt0, cnt1): cnt>0 -> leaf, c = first triangle; cnt==0 -> c = internal node index (c=-1 empty)
struct BVH2Flat {
    std::vector<float4> nodes; // 4 float4 per node
    int emit(const BVHBuilder& B, int bn) {
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
    void build(BVHBuilder& B) {
        nodes.clear();
        if (B.nodes[0].leaf()) { // root leaf: wrap in an internal node with one empty child
            BuildNode root; root.box = B.nodes[0].box; root.left = 0; root.right = -1;
            B.nodes.push_back(root); emit(B, (int)B.nodes.size() - 1);
        } else emit(B, 0);
    }
};

// BVH4 GPU node (128 B): bminx,bmaxx,bminy,bmaxy,bminz,bmaxz (float4 each, lane = child), child int4, cnt int4
struct BVH4Flat {
    std::vector<float4> nodes; // 8 float4 per node
    int emit(const BVHBuilder& B, int bn) {
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
    void build(BVHBuilder& B) {
        nodes.clear();
        if (B.nodes[0].leaf()) { BuildNode root; root.box = B.nodes[0].box; root.left = 0; root.right = -1; B.nodes.push_back(root); emit(B, (int)B.nodes.size() - 1); }
        else emit(B, 0);
    }
};
