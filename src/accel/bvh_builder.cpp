// accel/bvh_builder.cpp - binned-SAH BVH2 builder (written from scratch)
#include "accel/bvh_builder.h"
#include <algorithm>

void BVHBuilder::build(const Mesh& m) {
    size_t n = m.f.size(); tb.resize(n); cent.resize(n); idx.resize(n);
    for (size_t i = 0; i < n; i++) {
        AABB b; b.grow(m.v[m.f[i].x]); b.grow(m.v[m.f[i].y]); b.grow(m.v[m.f[i].z]);
        tb[i] = b; cent[i] = (b.lo + b.hi) * 0.5f; idx[i] = (int)i;
    }
    nodes.reserve(2 * n / leafMax + 16);
    rec(0, (int)n, 0);
}
int BVHBuilder::rec(int begin, int end, int depth) {
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
