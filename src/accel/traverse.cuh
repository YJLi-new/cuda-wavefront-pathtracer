// accel/traverse.cuh - device-side ray traversal on CUDA cores (no RT cores required).
//   isectLeaf<ANY, TRI>: leaf triangle tests, TRI 0 = Moller-Trumbore on (v0, e1, e2), TRI 1 = Woop unit-triangle rows
//   trav2 / trav4 / trav8: BVH2 (64 B nodes), BVH4 (128 B), compressed BVH8 (80 B); ANY = any-hit (shadow rays)
// Written from scratch following Aila & Laine 2009 (BVH2 layout, nearer-child-first stack traversal) and
// Ylitie et al. 2017 (compressed wide BVH traversal with octant-ordered child slots).
#pragma once
#include "core/vec.h"

// ---------------- triangle tests ----------------
template <bool ANY, int TRI>
__device__ __forceinline__ bool isectLeaf(const float4* __restrict__ tris, int first, int cnt, float3 o, float3 d, float tmin, float& hitT, int& hitTri, float& hu, float& hv) {
    for (int t = first; t < first + cnt; t++) {
        if (TRI == 0) { // Moller-Trumbore, (v0, e1, e2)
            float4 a = tris[3 * t], b = tris[3 * t + 1], c = tris[3 * t + 2];
            float3 v0 = xyz(a), e1 = xyz(b), e2 = xyz(c);
            float3 pv = cross(d, e2); float det = dot(e1, pv);
            if (fabsf(det) < 1e-12f) continue;
            float inv = 1.f / det; float3 tv = o - v0; float u = dot(tv, pv) * inv;
            if (u < 0.f || u > 1.f) continue;
            float3 qv = cross(tv, e1); float v = dot(d, qv) * inv;
            if (v < 0.f || u + v > 1.f) continue;
            float tt = dot(e2, qv) * inv;
            if (tt > tmin && tt < hitT) { hitT = tt; hitTri = t; hu = u; hv = v; if (ANY) return true; }
        } else { // Woop unit-triangle transform rows
            float4 r2 = tris[3 * t + 2];
            float Oz = r2.w + o.x * r2.x + o.y * r2.y + o.z * r2.z, Dz = d.x * r2.x + d.y * r2.y + d.z * r2.z;
            float tt = -Oz / Dz;
            if (tt > tmin && tt < hitT) {
                float4 r0 = tris[3 * t];
                float Ox = r0.w + o.x * r0.x + o.y * r0.y + o.z * r0.z, Dx = d.x * r0.x + d.y * r0.y + d.z * r0.z;
                float u = Ox + tt * Dx;
                if (u >= 0.f && u <= 1.f) {
                    float4 r1 = tris[3 * t + 1];
                    float Oy = r1.w + o.x * r1.x + o.y * r1.y + o.z * r1.z, Dy = d.x * r1.x + d.y * r1.y + d.z * r1.z;
                    float v = Oy + tt * Dy;
                    if (v >= 0.f && u + v <= 1.f) { hitT = tt; hitTri = t; hu = u; hv = v; if (ANY) return true; }
                }
            }
        }
    }
    return false;
}

// ---------------- BVH2 traversal ----------------
template <bool ANY, int TRI>
__device__ __forceinline__ bool trav2(const float4* __restrict__ nodes, const float4* __restrict__ tris, float3 o, float3 d, float tmin, float& hitT, int& hitTri, float& hu, float& hv) {
    int stack[64]; int sp = 0; int node = 0;
    const float ooeps = 1e-20f;
    float idx = 1.f / (fabsf(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
    float idy = 1.f / (fabsf(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
    float idz = 1.f / (fabsf(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
    float oodx = o.x * idx, oody = o.y * idy, oodz = o.z * idz;
    while (true) {
        const float4 n0 = nodes[node * 4 + 0], n1 = nodes[node * 4 + 1], nz = nodes[node * 4 + 2], mt = nodes[node * 4 + 3];
        int c0 = __float_as_int(mt.x), c1 = __float_as_int(mt.y), k0 = __float_as_int(mt.z), k1 = __float_as_int(mt.w);
        float c0lox = n0.x * idx - oodx, c0hix = n0.y * idx - oodx, c0loy = n0.z * idy - oody, c0hiy = n0.w * idy - oody, c0loz = nz.x * idz - oodz, c0hiz = nz.y * idz - oodz;
        float c1lox = n1.x * idx - oodx, c1hix = n1.y * idx - oodx, c1loy = n1.z * idy - oody, c1hiy = n1.w * idy - oody, c1loz = nz.z * idz - oodz, c1hiz = nz.w * idz - oodz;
        float c0min = fmaxf(fmaxf(fminf(c0lox, c0hix), fminf(c0loy, c0hiy)), fmaxf(fminf(c0loz, c0hiz), tmin));
        float c0max = fminf(fminf(fmaxf(c0lox, c0hix), fmaxf(c0loy, c0hiy)), fminf(fmaxf(c0loz, c0hiz), hitT));
        float c1min = fmaxf(fmaxf(fminf(c1lox, c1hix), fminf(c1loy, c1hiy)), fmaxf(fminf(c1loz, c1hiz), tmin));
        float c1max = fminf(fminf(fmaxf(c1lox, c1hix), fmaxf(c1loy, c1hiy)), fminf(fmaxf(c1loz, c1hiz), hitT));
        bool h0 = c0min <= c0max, h1 = c1min <= c1max;
        if (h0 && k0 > 0) { if (isectLeaf<ANY, TRI>(tris, c0, k0, o, d, tmin, hitT, hitTri, hu, hv) && ANY) return true; h0 = false; }
        if (h1 && k1 > 0) { if (isectLeaf<ANY, TRI>(tris, c1, k1, o, d, tmin, hitT, hitTri, hu, hv) && ANY) return true; h1 = false; }
        if (!h0 && !h1) { if (sp == 0) break; node = stack[--sp]; }
        else if (h0 && h1) { if (c1min < c0min) { int t = c0; c0 = c1; c1 = t; } stack[sp++] = c1; node = c0; }
        else node = h0 ? c0 : c1;
    }
    return hitTri >= 0;
}

// ---------------- BVH4 traversal ----------------
#define CSWAP(a, b) { if (ht##b < ht##a) { float tf = ht##a; ht##a = ht##b; ht##b = tf; int ti = hc##a; hc##a = hc##b; hc##b = ti; } }
template <bool ANY, int TRI>
__device__ __forceinline__ bool trav4(const float4* __restrict__ nodes, const float4* __restrict__ tris, float3 o, float3 d, float tmin, float& hitT, int& hitTri, float& hu, float& hv) {
    int stack[64]; int sp = 0; int node = 0;
    const float ooeps = 1e-20f;
    float idx = 1.f / (fabsf(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
    float idy = 1.f / (fabsf(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
    float idz = 1.f / (fabsf(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
    float oodx = o.x * idx, oody = o.y * idy, oodz = o.z * idz;
    const float INF = 3.4e38f;
    while (true) {
        const float4* nb = nodes + node * 8;
        float4 bx0 = nb[0], bx1 = nb[1], by0 = nb[2], by1 = nb[3], bz0 = nb[4], bz1 = nb[5], cf = nb[6], kf = nb[7];
        float ht0, ht1, ht2, ht3; int hc0, hc1, hc2, hc3;
#define SLAB(J, LX, HX, LY, HY, LZ, HZ, CC, KK) { \
            float x0 = LX * idx - oodx, x1 = HX * idx - oodx, y0 = LY * idy - oody, y1 = HY * idy - oody, z0 = LZ * idz - oodz, z1 = HZ * idz - oodz; \
            float tn = fmaxf(fmaxf(fminf(x0, x1), fminf(y0, y1)), fmaxf(fminf(z0, z1), tmin)); \
            float tf = fminf(fminf(fmaxf(x0, x1), fmaxf(y0, y1)), fminf(fmaxf(z0, z1), hitT)); \
            int c = __float_as_int(CC), k = __float_as_int(KK); bool hit = tn <= tf; \
            ht##J = INF; hc##J = c; \
            if (hit) { if (k > 0) { if (isectLeaf<ANY, TRI>(tris, c, k, o, d, tmin, hitT, hitTri, hu, hv) && ANY) return true; } else ht##J = tn; } }
        SLAB(0, bx0.x, bx1.x, by0.x, by1.x, bz0.x, bz1.x, cf.x, kf.x)
        SLAB(1, bx0.y, bx1.y, by0.y, by1.y, bz0.y, bz1.y, cf.y, kf.y)
        SLAB(2, bx0.z, bx1.z, by0.z, by1.z, bz0.z, bz1.z, cf.z, kf.z)
        SLAB(3, bx0.w, bx1.w, by0.w, by1.w, bz0.w, bz1.w, cf.w, kf.w)
#undef SLAB
        CSWAP(0, 1) CSWAP(2, 3) CSWAP(0, 2) CSWAP(1, 3) CSWAP(1, 2)
        if (ht0 == INF) { if (sp == 0) break; node = stack[--sp]; continue; }
        if (ht3 < INF) stack[sp++] = hc3;
        if (ht2 < INF) stack[sp++] = hc2;
        if (ht1 < INF) stack[sp++] = hc1;
        node = hc0;
    }
    return hitTri >= 0;
}
#undef CSWAP

// ---------------- BVH8 compressed traversal (CWBVH-style) ----------------
#define B8(u, s) (((u) >> (8 * ((s) & 3))) & 255u)
template <bool ANY, int TRI>
__device__ __forceinline__ bool trav8(const float4* __restrict__ nodes, const float4* __restrict__ tris, float3 o, float3 d, float tmin, float& hitT, int& hitTri, float& hu, float& hv) {
    uint2 stack[40]; int sp = 0;
    const float ooeps = 1e-20f;
    float idx = 1.f / (fabsf(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
    float idy = 1.f / (fabsf(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
    float idz = 1.f / (fabsf(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
    unsigned p = (d.x > 0.f ? 1u : 0u) | (d.y > 0.f ? 2u : 0u) | (d.z > 0.f ? 4u : 0u);
    unsigned cb = 0, im = 0, hm = 0; int node = 0;
    while (true) {
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
            unsigned m = s < 4 ? B8(m0123, s) : B8(m4567, s);
            if (m == 0) continue;
            float x0 = ox + (float)(s < 4 ? B8(qlox0, s) : B8(qlox1, s)) * sx, x1 = ox + (float)(s < 4 ? B8(qhix0, s) : B8(qhix1, s)) * sx;
            float y0 = oy + (float)(s < 4 ? B8(qloy0, s) : B8(qloy1, s)) * sy, y1 = oy + (float)(s < 4 ? B8(qhiy0, s) : B8(qhiy1, s)) * sy;
            float z0 = oz + (float)(s < 4 ? B8(qloz0, s) : B8(qloz1, s)) * sz, z1 = oz + (float)(s < 4 ? B8(qhiz0, s) : B8(qhiz1, s)) * sz;
            float tn = fmaxf(fmaxf(fminf(x0, x1), fminf(y0, y1)), fmaxf(fminf(z0, z1), tmin));
            float tf = fminf(fminf(fmaxf(x0, x1), fmaxf(y0, y1)), fminf(fmaxf(z0, z1), hitT));
            if (tn <= tf) {
                if (m & 0x80u) { if (isectLeaf<ANY, TRI>(tris, triBase + (int)(m & 31u), (int)((m >> 5) & 3u) + 1, o, d, tmin, hitT, hitTri, hu, hv) && ANY) return true; }
                else hits |= 1u << (s ^ p);
            }
        }
        if (hits) { if (hm) stack[sp++] = make_uint2(cb, (im << 8) | hm); cb = (unsigned)childBase; im = imask; hm = hits; }
        while (hm == 0) { if (sp == 0) return hitTri >= 0; uint2 e = stack[--sp]; cb = e.x; im = e.y >> 8; hm = e.y & 255u; }
        unsigned k = __ffs(hm) - 1; hm &= hm - 1; unsigned slot = k ^ p;
        node = (int)cb + __popc(im & ((1u << slot) - 1u));
    }
}
#undef B8

template <int V, bool ANY, int TRI>
__device__ __forceinline__ bool trav(const float4* __restrict__ nodes, const float4* __restrict__ tris, float3 o, float3 d, float tmin, float& hitT, int& hitTri, float& hu, float& hv) {
    if (V == 2) return trav2<ANY, TRI>(nodes, tris, o, d, tmin, hitT, hitTri, hu, hv);
    if (V == 4) return trav4<ANY, TRI>(nodes, tris, o, d, tmin, hitT, hitTri, hu, hv);
    return trav8<ANY, TRI>(nodes, tris, o, d, tmin, hitT, hitTri, hu, hv);
}
