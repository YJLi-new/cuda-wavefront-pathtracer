// accel/traverse_inst.cuh - two-level (TLAS/BLAS) traversal and step-wise traversal for persistent threads.
// Same node layouts and triangle tests as accel/traverse.cuh.
#pragma once
#include "accel/traverse.cuh"

// ---------------- two-level traversal: TLAS over instances -> BLAS of the instance's mesh ----------------
// The TLAS uses the BVH2 layout with leaves holding instance ids. A ray enters a BLAS in object space (o' = inv o,
// d' = inv d, d' not normalised, so hit distances t are the same in both spaces). BLAS node / triangle indices are
// absolute (all BLASes share one node array and one triangle array); hitInst = instance of the closest hit.
template <int V, bool ANY, int TRI>
__device__ __forceinline__ bool travInst(const float4* __restrict__ tlas, int troot, const DevInstance* __restrict__ inst, const float4* __restrict__ nodes, const float4* __restrict__ tris,
                                         float3 o, float3 d, float tmin, float& hitT, int& hitTri, float& hu, float& hv, int& hitInst) {
    int stack[64]; int sp = 0; int node = troot;
    const float ooeps = 1e-20f;
    float idx = 1.f / (fabsf(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
    float idy = 1.f / (fabsf(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
    float idz = 1.f / (fabsf(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
    float oodx = o.x * idx, oody = o.y * idy, oodz = o.z * idz;
#define ENTER_INST(J) { \
        const float4 r_[3] = { inst[J].inv[0], inst[J].inv[1], inst[J].inv[2] }; int before_ = hitTri; \
        bool h_ = trav<V, ANY, TRI>(nodes, tris, xfRowsP(r_, o), xfRowsV(r_, d), tmin, hitT, hitTri, hu, hv, inst[J].root); \
        if (hitTri != before_) hitInst = J; if (ANY && h_) return true; }
    while (true) {
        const float4 n0 = tlas[node * 4 + 0], n1 = tlas[node * 4 + 1], nz = tlas[node * 4 + 2], mt = tlas[node * 4 + 3];
        int c0 = __float_as_int(mt.x), c1 = __float_as_int(mt.y), k0 = __float_as_int(mt.z), k1 = __float_as_int(mt.w);
        float c0lox = n0.x * idx - oodx, c0hix = n0.y * idx - oodx, c0loy = n0.z * idy - oody, c0hiy = n0.w * idy - oody, c0loz = nz.x * idz - oodz, c0hiz = nz.y * idz - oodz;
        float c1lox = n1.x * idx - oodx, c1hix = n1.y * idx - oodx, c1loy = n1.z * idy - oody, c1hiy = n1.w * idy - oody, c1loz = nz.z * idz - oodz, c1hiz = nz.w * idz - oodz;
        float c0min = fmaxf(fmaxf(fminf(c0lox, c0hix), fminf(c0loy, c0hiy)), fmaxf(fminf(c0loz, c0hiz), tmin));
        float c0max = fminf(fminf(fmaxf(c0lox, c0hix), fmaxf(c0loy, c0hiy)), fminf(fmaxf(c0loz, c0hiz), hitT));
        float c1min = fmaxf(fmaxf(fminf(c1lox, c1hix), fminf(c1loy, c1hiy)), fmaxf(fminf(c1loz, c1hiz), tmin));
        float c1max = fminf(fminf(fmaxf(c1lox, c1hix), fmaxf(c1loy, c1hiy)), fminf(fmaxf(c1loz, c1hiz), hitT));
        bool h0 = c0min <= c0max, h1 = c1min <= c1max;
        if (h0 && k0 > 0) { for (int t = c0; t < c0 + k0; t++) ENTER_INST(t) h0 = false; }
        if (h1 && k1 > 0) { for (int t = c1; t < c1 + k1; t++) ENTER_INST(t) h1 = false; }
        if (!h0 && !h1) { if (sp == 0) break; node = stack[--sp]; }
        else if (h0 && h1) { if (c1min < c0min) { int t = c0; c0 = c1; c1 = t; } stack[sp++] = c1; node = c0; }
        else node = h0 ? c0 : c1;
    }
#undef ENTER_INST
    return hitTri >= 0;
}

// ---------------- step-wise traversal (persistent threads with dynamic ray fetching, Aila & Laine 2009) ----------------
// The traversal state lives in the lane and one step = one node visit (TLAS node, BVH2 node with its leaf triangles,
// or compressed BVH8 node with its leaf triangles), so a lane that finishes its ray can fetch a new one while the rest
// of its warp keeps traversing. V = 2 or 8.
template <int V, bool ANY, int TRI, bool INST>
struct StepTrav {
    float3 o, d, wo, wd; float tmin, hitT, hu, hv; int hitTri, hitInst;
    float idx, idy, idz, oodx, oody, oodz; unsigned oct;
    int node, sp; int stack[V == 2 ? 64 : 1];
    unsigned cb, im, hm; int sp8; uint2 st8[V == 8 ? 40 : 1];
    int level, tnode, tsp, cur; int tstack[INST ? 64 : 1];

    __device__ __forceinline__ void setRay(float3 o_, float3 d_) {
        o = o_; d = d_; const float ooeps = 1e-20f;
        idx = 1.f / (fabsf(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
        idy = 1.f / (fabsf(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
        idz = 1.f / (fabsf(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
        oodx = o.x * idx; oody = o.y * idy; oodz = o.z * idz;
        oct = (d.x > 0.f ? 1u : 0u) | (d.y > 0.f ? 2u : 0u) | (d.z > 0.f ? 4u : 0u);
    }
    __device__ __forceinline__ void beginBlas(int root) { node = root; sp = 0; cb = 0; im = 0; hm = 0; sp8 = 0; }
    __device__ __forceinline__ void init(float3 o_, float3 d_, float tmin_, float tmax_, int troot) {
        wo = o_; wd = d_; tmin = tmin_; hitT = tmax_; hitTri = -1; hitInst = -1; hu = 0.f; hv = 0.f; cur = -1;
        setRay(o_, d_);
        if (INST) { level = 0; tnode = troot; tsp = 0; } else beginBlas(0);
    }
    // 0: continue, 1: BLAS exhausted, 2: any-hit found
    __device__ __forceinline__ int step2(const float4* __restrict__ nodes, const float4* __restrict__ tris) {
        const float4 n0 = nodes[node * 4 + 0], n1 = nodes[node * 4 + 1], nz = nodes[node * 4 + 2], mt = nodes[node * 4 + 3];
        int c0 = __float_as_int(mt.x), c1 = __float_as_int(mt.y), k0 = __float_as_int(mt.z), k1 = __float_as_int(mt.w);
        float c0lox = n0.x * idx - oodx, c0hix = n0.y * idx - oodx, c0loy = n0.z * idy - oody, c0hiy = n0.w * idy - oody, c0loz = nz.x * idz - oodz, c0hiz = nz.y * idz - oodz;
        float c1lox = n1.x * idx - oodx, c1hix = n1.y * idx - oodx, c1loy = n1.z * idy - oody, c1hiy = n1.w * idy - oody, c1loz = nz.z * idz - oodz, c1hiz = nz.w * idz - oodz;
        float c0min = fmaxf(fmaxf(fminf(c0lox, c0hix), fminf(c0loy, c0hiy)), fmaxf(fminf(c0loz, c0hiz), tmin));
        float c0max = fminf(fminf(fmaxf(c0lox, c0hix), fmaxf(c0loy, c0hiy)), fminf(fmaxf(c0loz, c0hiz), hitT));
        float c1min = fmaxf(fmaxf(fminf(c1lox, c1hix), fminf(c1loy, c1hiy)), fmaxf(fminf(c1loz, c1hiz), tmin));
        float c1max = fminf(fminf(fmaxf(c1lox, c1hix), fmaxf(c1loy, c1hiy)), fminf(fmaxf(c1loz, c1hiz), hitT));
        bool h0 = c0min <= c0max, h1 = c1min <= c1max;
        if (h0 && k0 > 0) { if (isectLeaf<ANY, TRI>(tris, c0, k0, o, d, tmin, hitT, hitTri, hu, hv) && ANY) return 2; h0 = false; }
        if (h1 && k1 > 0) { if (isectLeaf<ANY, TRI>(tris, c1, k1, o, d, tmin, hitT, hitTri, hu, hv) && ANY) return 2; h1 = false; }
        if (!h0 && !h1) { if (sp == 0) return 1; node = stack[--sp]; }
        else if (h0 && h1) { if (c1min < c0min) { int t = c0; c0 = c1; c1 = t; } stack[sp++] = c1; node = c0; }
        else node = h0 ? c0 : c1;
        return 0;
    }
    __device__ __forceinline__ int step8(const float4* __restrict__ nodes, const float4* __restrict__ tris) {
#define SB8(u, s) (((u) >> (8 * ((s) & 3))) & 255u)
        const float4* nb = nodes + node * 5;
        float4 f0 = nb[0], f1 = nb[1], f2 = nb[2], f3 = nb[3], f4 = nb[4];
        unsigned ew = __float_as_uint(f0.w);
        float sx = __uint_as_float((ew & 255u) << 23) * idx, sy = __uint_as_float(((ew >> 8) & 255u) << 23) * idy, sz = __uint_as_float(((ew >> 16) & 255u) << 23) * idz;
        float ox = (f0.x - o.x) * idx, oy = (f0.y - o.y) * idy, oz = (f0.z - o.z) * idz;
        unsigned imask = ew >> 24, m0123 = __float_as_uint(f1.z), m4567 = __float_as_uint(f1.w);
        int childBase = __float_as_int(f1.x), triBase = __float_as_int(f1.y);
        unsigned qlox0 = __float_as_uint(f2.x), qlox1 = __float_as_uint(f2.y), qloy0 = __float_as_uint(f2.z), qloy1 = __float_as_uint(f2.w);
        unsigned qloz0 = __float_as_uint(f3.x), qloz1 = __float_as_uint(f3.y), qhix0 = __float_as_uint(f3.z), qhix1 = __float_as_uint(f3.w);
        unsigned qhiy0 = __float_as_uint(f4.x), qhiy1 = __float_as_uint(f4.y), qhiz0 = __float_as_uint(f4.z), qhiz1 = __float_as_uint(f4.w);
        unsigned hits = 0;
#pragma unroll
        for (int s = 0; s < 8; s++) {
            unsigned m = s < 4 ? SB8(m0123, s) : SB8(m4567, s);
            if (m == 0) continue;
            float x0 = ox + (float)(s < 4 ? SB8(qlox0, s) : SB8(qlox1, s)) * sx, x1 = ox + (float)(s < 4 ? SB8(qhix0, s) : SB8(qhix1, s)) * sx;
            float y0 = oy + (float)(s < 4 ? SB8(qloy0, s) : SB8(qloy1, s)) * sy, y1 = oy + (float)(s < 4 ? SB8(qhiy0, s) : SB8(qhiy1, s)) * sy;
            float z0 = oz + (float)(s < 4 ? SB8(qloz0, s) : SB8(qloz1, s)) * sz, z1 = oz + (float)(s < 4 ? SB8(qhiz0, s) : SB8(qhiz1, s)) * sz;
            float tn = fmaxf(fmaxf(fminf(x0, x1), fminf(y0, y1)), fmaxf(fminf(z0, z1), tmin));
            float tf = fminf(fminf(fmaxf(x0, x1), fmaxf(y0, y1)), fminf(fmaxf(z0, z1), hitT));
            if (tn <= tf) {
                if (m & 0x80u) { if (isectLeaf<ANY, TRI>(tris, triBase + (int)(m & 31u), (int)((m >> 5) & 3u) + 1, o, d, tmin, hitT, hitTri, hu, hv) && ANY) return 2; }
                else hits |= 1u << (s ^ oct);
            }
        }
#undef SB8
        if (hits) { if (hm) st8[sp8++] = make_uint2(cb, (im << 8) | hm); cb = (unsigned)childBase; im = imask; hm = hits; }
        while (hm == 0) { if (sp8 == 0) return 1; uint2 e = st8[--sp8]; cb = e.x; im = e.y >> 8; hm = e.y & 255u; }
        unsigned k = __ffs(hm) - 1; hm &= hm - 1; unsigned slot = k ^ oct;
        node = (int)cb + __popc(im & ((1u << slot) - 1u));
        return 0;
    }
    // TLAS node visit (world ray); leaf children are queued as instance markers -(id + 2) and entered on the next step
    __device__ __forceinline__ bool stepTlas(const float4* __restrict__ tlas, const DevInstance* __restrict__ inst) {
        if (tnode < -1) {
            cur = -tnode - 2;
            const float4 r_[3] = { inst[cur].inv[0], inst[cur].inv[1], inst[cur].inv[2] };
            setRay(xfRowsP(r_, wo), xfRowsV(r_, wd)); beginBlas(inst[cur].root); level = 1;
            return false;
        }
        const float4 n0 = tlas[tnode * 4 + 0], n1 = tlas[tnode * 4 + 1], nz = tlas[tnode * 4 + 2], mt = tlas[tnode * 4 + 3];
        int c0 = __float_as_int(mt.x), c1 = __float_as_int(mt.y), k0 = __float_as_int(mt.z), k1 = __float_as_int(mt.w);
        float c0lox = n0.x * idx - oodx, c0hix = n0.y * idx - oodx, c0loy = n0.z * idy - oody, c0hiy = n0.w * idy - oody, c0loz = nz.x * idz - oodz, c0hiz = nz.y * idz - oodz;
        float c1lox = n1.x * idx - oodx, c1hix = n1.y * idx - oodx, c1loy = n1.z * idy - oody, c1hiy = n1.w * idy - oody, c1loz = nz.z * idz - oodz, c1hiz = nz.w * idz - oodz;
        float c0min = fmaxf(fmaxf(fminf(c0lox, c0hix), fminf(c0loy, c0hiy)), fmaxf(fminf(c0loz, c0hiz), tmin));
        float c0max = fminf(fminf(fmaxf(c0lox, c0hix), fmaxf(c0loy, c0hiy)), fminf(fmaxf(c0loz, c0hiz), hitT));
        float c1min = fmaxf(fmaxf(fminf(c1lox, c1hix), fminf(c1loy, c1hiy)), fmaxf(fminf(c1loz, c1hiz), tmin));
        float c1max = fminf(fminf(fmaxf(c1lox, c1hix), fmaxf(c1loy, c1hiy)), fminf(fmaxf(c1loz, c1hiz), hitT));
        bool h0 = c0min <= c0max, h1 = c1min <= c1max;
        int a0 = k0 > 0 ? -(c0 + 2) : c0, a1 = k1 > 0 ? -(c1 + 2) : c1;   // TLAS leaves hold one instance (H-PLOC)
        if (h0 && h1) { if (c1min < c0min) { int t = a0; a0 = a1; a1 = t; } tstack[tsp++] = a1; tnode = a0; }
        else if (h0) tnode = a0;
        else if (h1) tnode = a1;
        else { if (tsp == 0) return true; tnode = tstack[--tsp]; }
        return false;
    }
    // one step; true when the ray is finished
    __device__ __forceinline__ bool step(const float4* __restrict__ tlas, const DevInstance* __restrict__ inst, const float4* __restrict__ nodes, const float4* __restrict__ tris) {
        if (INST && level == 0) return stepTlas(tlas, inst);
        int before = hitTri;
        int r = V == 2 ? step2(nodes, tris) : step8(nodes, tris);
        if (INST && hitTri != before) hitInst = cur;
        if (r == 2) return true;
        if (r == 1) {
            if (!INST) return true;
            level = 0; setRay(wo, wd);
            if (tsp == 0) return true;
            tnode = tstack[--tsp];
        }
        return false;
    }
};
