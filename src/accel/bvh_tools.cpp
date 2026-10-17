// accel/bvh_tools.cpp - H-PLOC tree conversion, SAH cost, CWBVH collapse (Ylitie et al. 2017), BLAS concatenation.
// Written from scratch from the papers.
#include "accel/bvh_tools.h"
#include <cmath>
#include <cstring>
#include <algorithm>

static inline int asInt(float f) { int i; memcpy(&i, &f, 4); return i; }
static inline float asFloat(int i) { float f; memcpy(&f, &i, 4); return f; }
static inline unsigned asUint(float f) { unsigned u; memcpy(&u, &f, 4); return u; }

void treeFromHploc(const std::vector<float4>& hn, int n, int root, int leafMax, BVHBuilder& B) {
    const int tot = 2 * n - 1;
    std::vector<int> lc(tot), rc(tot), cnt(tot); std::vector<double> cost(tot); std::vector<char> asLeaf(tot); std::vector<AABB> box(tot);
    for (int h = 0; h < tot; h++) {   // children ids are always smaller than their parent's: increasing id = post-order
        AABB b; b.lo = xyz(hn[2 * h]); b.hi = xyz(hn[2 * h + 1]); box[h] = b;
        lc[h] = asInt(hn[2 * h].w); rc[h] = asInt(hn[2 * h + 1].w);
        double A = b.area();
        if (h < n) { cnt[h] = 1; cost[h] = A; asLeaf[h] = 1; continue; }
        cnt[h] = cnt[lc[h]] + cnt[rc[h]];
        double ci = A + cost[lc[h]] + cost[rc[h]], cl = cnt[h] <= leafMax ? A * cnt[h] : 1e300;
        asLeaf[h] = cl <= ci; cost[h] = std::min(cl, ci);
    }
    B.nodes.clear(); B.idx.clear(); B.idx.reserve(n); B.nodes.reserve(2 * (size_t)n); B.maxDepth = 0; B.leafMax = leafMax;
    struct E { int h, parentSlot, depth; };
    std::vector<E> st; st.push_back({ root, -1, 0 }); std::vector<int> gs;
    while (!st.empty()) {
        E e = st.back(); st.pop_back();
        int id = (int)B.nodes.size(); BuildNode bn; bn.box = box[e.h]; B.nodes.push_back(bn);
        if (e.parentSlot >= 0) { if (e.parentSlot & 1) B.nodes[e.parentSlot >> 1].right = id; else B.nodes[e.parentSlot >> 1].left = id; }
        if (e.depth > B.maxDepth) B.maxDepth = e.depth;
        if (asLeaf[e.h]) {
            int first = (int)B.idx.size(); gs.clear(); gs.push_back(e.h);
            while (!gs.empty()) { int h = gs.back(); gs.pop_back(); if (h < n) B.idx.push_back(lc[h]); else { gs.push_back(rc[h]); gs.push_back(lc[h]); } }
            B.nodes[id].first = first; B.nodes[id].count = (int)B.idx.size() - first;
        } else { st.push_back({ rc[e.h], id * 2 + 1, e.depth + 1 }); st.push_back({ lc[e.h], id * 2, e.depth + 1 }); }
    }
}

double sahCost(const BVHBuilder& B, double cNode, double cPrim) {
    double s = 0, a0 = B.nodes[0].box.area(); if (!(a0 > 0)) return 0;
    for (const BuildNode& n : B.nodes) s += n.leaf() ? n.box.area() * n.count * cPrim : n.box.area() * cNode;
    return s / a0;
}

// ---------------- CWBVH ----------------
namespace {
struct CW {
    std::vector<AABB> box; std::vector<int> L, R, first, cnt;   // binary working tree (leaves: L = R = -1)
    std::vector<float> C; std::vector<unsigned char> D, K8;     // DP tables (i = 1..7) and decisions
    const std::vector<int>* idx = nullptr;
    enum { LEAF = 0, NODE = 1, PREV = 2, DIST = 10 };
    void collect(int n, int i, std::vector<int>& out) const {
        while (true) {
            if (i == 1) { out.push_back(n); return; }
            unsigned char d = D[n * 8 + i];
            if (d == PREV) { i--; continue; }
            int k = d - DIST; collect(L[n], k, out); collect(R[n], i - k, out); return;
        }
    }
    void prims(int n, std::vector<int>& out) const {
        std::vector<int> st(1, n);
        while (!st.empty()) { int h = st.back(); st.pop_back(); if (L[h] < 0) { for (int t = 0; t < cnt[h]; t++) out.push_back((*idx)[first[h] + t]); } else { st.push_back(R[h]); st.push_back(L[h]); } }
    }
};
int expo8(float ext, float lo, float hi) {
    if (!(ext > 0.f)) return -100;
    int e = (int)ceilf(log2f(ext / 255.f)); if (e < -100) e = -100;
    while (lo + 255.f * ldexpf(1.f, e) < hi) e++;
    return e;
}
unsigned pack4(const int* q, int b) { return (unsigned)q[b] | ((unsigned)q[b + 1] << 8) | ((unsigned)q[b + 2] << 16) | ((unsigned)q[b + 3] << 24); }
}

void CWBVH::build(const BVHBuilder& B) {
    CW w; w.idx = &B.idx;
    size_t N0 = B.nodes.size();
    w.box.resize(N0); w.L.resize(N0); w.R.resize(N0); w.first.resize(N0); w.cnt.resize(N0);
    for (size_t i = 0; i < N0; i++) { const BuildNode& b = B.nodes[i]; w.box[i] = b.box; w.L[i] = b.leaf() ? -1 : b.left; w.R[i] = b.leaf() ? -1 : b.right; w.first[i] = b.first; w.cnt[i] = b.leaf() ? b.count : 0; }
    // leaves larger than pMax (rare: builder depth limits) are split into binary subtrees with the same box
    for (size_t i = 0; i < w.box.size(); i++) {
        if (w.L[i] >= 0 || w.cnt[i] <= pMax) continue;
        int h = w.cnt[i] / 2; int a = (int)w.box.size(), b = a + 1;
        for (int j = 0; j < 2; j++) { w.box.push_back(w.box[i]); w.L.push_back(-1); w.R.push_back(-1); }
        w.first.push_back(w.first[i]); w.cnt.push_back(h); w.first.push_back(w.first[i] + h); w.cnt.push_back(w.cnt[i] - h);
        w.L[i] = a; w.R[i] = b; w.cnt[i] = 0;
    }
    const size_t N = w.box.size();
    w.C.assign(N * 8, 0.f); w.D.assign(N * 8, 0); w.K8.assign(N, 1);
    std::vector<int> p(N, 0);
    const float INF = 3e38f;
    for (size_t ii = N; ii-- > 0;) {   // children have larger ids than their parent (pre-order allocation): decreasing id = post-order
        int n = (int)ii; float A = w.box[n].area();
        float* c = &w.C[n * 8]; unsigned char* d = &w.D[n * 8];
        if (w.L[n] < 0) { p[n] = w.cnt[n]; float cl = A * p[n] * cPrim; for (int i = 1; i < 8; i++) { c[i] = cl; d[i] = i == 1 ? (unsigned char)CW::LEAF : (unsigned char)CW::PREV; } continue; }
        int l = w.L[n], r = w.R[n]; p[n] = p[l] + p[r];
        const float* cl_ = &w.C[l * 8]; const float* cr_ = &w.C[r * 8];
        float dist[9]; int dk[9];
        for (int j = 2; j <= 8; j++) { dist[j] = INF; dk[j] = 1; for (int k = 1; k < j; k++) { float v = cl_[k] + cr_[j - k]; if (v < dist[j]) { dist[j] = v; dk[j] = k; } } }
        float cLeaf = p[n] <= pMax ? A * p[n] * cPrim : INF, cInt = dist[8] + A * cNode;
        if (cLeaf <= cInt) { c[1] = cLeaf; d[1] = CW::LEAF; } else { c[1] = cInt; d[1] = CW::NODE; }
        w.K8[n] = (unsigned char)dk[8];
        for (int i = 2; i < 8; i++) { if (dist[i] < c[i - 1]) { c[i] = dist[i]; d[i] = (unsigned char)(CW::DIST + dk[i]); } else { c[i] = c[i - 1]; d[i] = CW::PREV; } }
    }
    nodes.clear(); order.clear(); order.reserve(B.idx.size()); maxLeafTris = 0;
    const int root = 0; const float Ar = w.box[root].area();
    bool rootLeaf = w.L[root] < 0 || w.D[root * 8 + 1] == CW::LEAF;
    if (rootLeaf) sah = Ar > 0 ? (Ar * p[root] * cPrim + Ar * cNode) / Ar : 0;
    else { float d8 = INF; int l = w.L[root], r = w.R[root]; for (int k = 1; k < 8; k++) d8 = std::min(d8, w.C[l * 8 + k] + w.C[r * 8 + 8 - k]); sah = Ar > 0 ? (d8 + Ar * cNode) / Ar : 0; }
    // emit wide nodes (pre-order, children of a node stored contiguously)
    struct Job { int bn, out; bool asLeaf; };
    std::vector<Job> jobs; nodes.resize(5); jobs.push_back({ root, 0, rootLeaf });
    std::vector<int> ch, pr;
    while (!jobs.empty()) {
        Job jb = jobs.back(); jobs.pop_back();
        const int n = jb.bn;
        ch.clear();
        if (jb.asLeaf) ch.push_back(n); else { w.collect(w.L[n], w.K8[n], ch); w.collect(w.R[n], 8 - w.K8[n], ch); }
        const AABB box = w.box[n]; float3 pc = (box.lo + box.hi) * 0.5f;
        // greedy slot assignment: slot s prefers the child farthest along the octant direction of s (bit set = negative axis)
        int slotChild[8]; for (int s = 0; s < 8; s++) slotChild[s] = -1;
        float cost[8][8]; bool used[8] = { false };
        for (size_t c = 0; c < ch.size(); c++) {
            float3 cc = (w.box[ch[c]].lo + w.box[ch[c]].hi) * 0.5f - pc;
            for (int s = 0; s < 8; s++) cost[c][s] = -(cc.x * (s & 1 ? -1.f : 1.f) + cc.y * (s & 2 ? -1.f : 1.f) + cc.z * (s & 4 ? -1.f : 1.f));
        }
        for (size_t it = 0; it < ch.size(); it++) {
            float bc = INF; int bi = -1, bs = -1;
            for (size_t c = 0; c < ch.size(); c++) if (!used[c]) for (int s = 0; s < 8; s++) if (slotChild[s] < 0 && cost[c][s] < bc) { bc = cost[c][s]; bi = (int)c; bs = s; }
            if (bi < 0) for (size_t c = 0; c < ch.size(); c++) if (!used[c]) for (int s = 0; s < 8; s++) if (slotChild[s] < 0 && bi < 0) { bi = (int)c; bs = s; }
            used[bi] = true; slotChild[bs] = ch[bi];
        }
        float3 ext = box.extent();
        int ex = expo8(ext.x, box.lo.x, box.hi.x), ey = expo8(ext.y, box.lo.y, box.hi.y), ez = expo8(ext.z, box.lo.z, box.hi.z);
        float sx = ldexpf(1.f, ex), sy = ldexpf(1.f, ey), sz = ldexpf(1.f, ez);
        int qlox[8], qloy[8], qloz[8], qhix[8], qhiy[8], qhiz[8], meta[8];
        int triBase = (int)order.size(); unsigned imask = 0; int nInt = 0;
        bool childLeaf[8];
        for (int s = 0; s < 8; s++) {
            qlox[s] = qloy[s] = qloz[s] = qhix[s] = qhiy[s] = qhiz[s] = 0; meta[s] = 0; childLeaf[s] = false;
            int cn = slotChild[s]; if (cn < 0) continue;
            const AABB& cb = w.box[cn];
            auto q = [](float v, float o, float sc, bool up) { float f = (v - o) / sc; int i = up ? (int)ceilf(f) : (int)floorf(f); return i < 0 ? 0 : i > 255 ? 255 : i; };
            qlox[s] = q(cb.lo.x, box.lo.x, sx, false); qhix[s] = q(cb.hi.x, box.lo.x, sx, true);
            qloy[s] = q(cb.lo.y, box.lo.y, sy, false); qhiy[s] = q(cb.hi.y, box.lo.y, sy, true);
            qloz[s] = q(cb.lo.z, box.lo.z, sz, false); qhiz[s] = q(cb.hi.z, box.lo.z, sz, true);
            bool lf = jb.asLeaf || w.L[cn] < 0 || w.D[cn * 8 + 1] == CW::LEAF;
            if (lf) {
                pr.clear(); w.prims(cn, pr);
                int off = (int)order.size() - triBase; int cnt = (int)pr.size();
                for (int t : pr) order.push_back(t);
                meta[s] = 0x80 | ((cnt - 1) << 5) | off;
                if (cnt > maxLeafTris) maxLeafTris = cnt;
                childLeaf[s] = true;
            } else { meta[s] = 0x40; imask |= 1u << s; nInt++; }
        }
        int childBase = (int)nodes.size() / 5;
        nodes.resize(nodes.size() + 5 * (size_t)nInt);
        float4* o = &nodes[(size_t)jb.out * 5];
        unsigned ew = (unsigned)(ex + 127) | ((unsigned)(ey + 127) << 8) | ((unsigned)(ez + 127) << 16) | (imask << 24);
        o[0] = make_float4(box.lo.x, box.lo.y, box.lo.z, asFloat((int)ew));
        o[1] = make_float4(asFloat(childBase), asFloat(triBase), asFloat((int)pack4(meta, 0)), asFloat((int)pack4(meta, 4)));
        unsigned u[12] = { pack4(qlox, 0), pack4(qlox, 4), pack4(qloy, 0), pack4(qloy, 4), pack4(qloz, 0), pack4(qloz, 4), pack4(qhix, 0), pack4(qhix, 4), pack4(qhiy, 0), pack4(qhiy, 4), pack4(qhiz, 0), pack4(qhiz, 4) };
        for (int j = 0; j < 3; j++) o[2 + j] = make_float4(asFloat((int)u[4 * j]), asFloat((int)u[4 * j + 1]), asFloat((int)u[4 * j + 2]), asFloat((int)u[4 * j + 3]));
        int ord = 0;
        for (int s = 0; s < 8; s++) if (slotChild[s] >= 0 && !childLeaf[s]) { jobs.push_back({ slotChild[s], childBase + ord, false }); ord++; }
    }
}

int appendBVH2(std::vector<float4>& all, const std::vector<float4>& blas, int triBase) {
    int base = (int)all.size() / 4;
    for (size_t i = 0; i < blas.size(); i += 4) {
        float4 m = blas[i + 3]; int c0 = asInt(m.x), c1 = asInt(m.y), k0 = asInt(m.z), k1 = asInt(m.w);
        if (k0 > 0) c0 += triBase; else if (c0 >= 0) c0 += base;
        if (k1 > 0) c1 += triBase; else if (c1 >= 0) c1 += base;
        all.push_back(blas[i]); all.push_back(blas[i + 1]); all.push_back(blas[i + 2]); all.push_back(make_float4(asFloat(c0), asFloat(c1), m.z, m.w));
    }
    return base;
}
int appendBVH8(std::vector<float4>& all, const std::vector<float4>& blas, int triBase) {
    int base = (int)all.size() / 5;
    for (size_t i = 0; i < blas.size(); i += 5) {
        float4 f1 = blas[i + 1];
        all.push_back(blas[i]);
        all.push_back(make_float4(asFloat(asInt(f1.x) + base), asFloat(asInt(f1.y) + triBase), f1.z, f1.w));
        all.push_back(blas[i + 2]); all.push_back(blas[i + 3]); all.push_back(blas[i + 4]);
    }
    return base;
}
