// main.cu - wavefront CUDA path tracer with software BVH traversal on CUDA cores (no RT cores required).
// Kernels: raygen / trace(closest) / shade / trace(shadow, any-hit),
// queue compaction via atomic append. Options: BVH2 (Aila-Laine style 64 B nodes) / BVH4 (128 B) / BVH8 compressed
// (80 B, quantized child boxes, octant-ordered); Moller-Trumbore or Woop triangles; host-driven loop (one sync per
// bounce) or device-driven loop (persistent-style kernels reading queue counts from device memory, no host syncs);
// optional counting-sort binning of bounce rays by (direction octant, Morton of origin); Lambert + sun NEE + sky.
#include "vec.h"
#include "mesh.h"
#include "bvh.h"
#include "bvh8.h"
#include "png.h"
#include <chrono>
#include <string>
#include <cstring>
#include <cstdint>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#ifndef MINB
#define LB __launch_bounds__(128)
#else
#define LB __launch_bounds__(128, MINB)
#endif

struct Cam { float3 pos, ax, ay, az; float fx, fy, px, py; int w, h; };
#define FT 64
#define NBINS 32768
struct Params { Cam cam; float3 sunTo, sunE, sky, bmin, binv; float sunCos; int maxB; unsigned seed; float eps, near; int hasN, filterBH; float ftab[FT + 1]; };
__constant__ Params P;

__device__ __forceinline__ unsigned hash32(unsigned x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
__device__ __forceinline__ float rnd(unsigned pix, unsigned smp, unsigned depth, unsigned dim) {
    unsigned h = hash32(pix * 0x9E3779B1U ^ hash32(smp * 0x85EBCA77U ^ hash32((depth * 8u + dim) * 0xC2B2AE3DU ^ P.seed)));
    return (h >> 8) * (1.0f / 16777216.0f);
}
__device__ __forceinline__ float filterSample(float u) { float f = u * FT; int k = (int)f; k = k < FT - 1 ? k : FT - 1; float t = f - k; return P.ftab[k] * (1.f - t) + P.ftab[k + 1] * t; }
__device__ __forceinline__ void basis(float3 n, float3& t, float3& b) {
    float s = n.z >= 0.f ? 1.f : -1.f; float a = -1.f / (s + n.z); float c = n.x * n.y * a;
    t = mk3(1.f + s * n.x * n.x * a, s * c, -s * n.x); b = mk3(c, s + n.y * n.y * a, -n.y);
}
__device__ __forceinline__ int warpFetch(int* fetch) { int b = 0; if ((threadIdx.x & 31) == 0) b = atomicAdd(fetch, 32); return __shfl_sync(0xffffffffu, b, 0); }
// persistent-style loop over [0, *nPtr): each warp grabs 32-element chunks
#define WARP_LOOP(nPtr, fetch, k) for (int n_ = *(nPtr), b_ = warpFetch(fetch); b_ < n_; b_ = warpFetch(fetch)) if (int k = b_ + (threadIdx.x & 31); k < n_)

// ---------------- ray generation ----------------
__global__ void k_raygen(float4* pO, float4* pD, float4* pT, int* pDepth, int* q, int smpBase, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; int N = P.cam.w * P.cam.h; if (i >= N * K) return;
    int pix = i % N, smp = smpBase + i / N;
    int x = pix % P.cam.w, y = pix / P.cam.w;
    float u0 = rnd(pix, smp, 0, 0), u1 = rnd(pix, smp, 0, 1), jx, jy;
    if (P.filterBH) { jx = filterSample(u0); jy = filterSample(u1); } else { jx = u0 - 0.5f; jy = u1 - 0.5f; }
    float dx = (x + 0.5f + jx - P.cam.px) / P.cam.fx, dy = -(y + 0.5f + jy - P.cam.py) / P.cam.fy;
    float3 d = normalize(P.cam.ax * dx + P.cam.ay * dy - P.cam.az);
    pO[i] = mk4(P.cam.pos, P.near); pD[i] = mk4(d, 1e30f); pT[i] = make_float4(1.f, 1.f, 1.f, __int_as_float(pix)); pDepth[i] = 0; q[i] = i;
}

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

// ---------------- trace kernels ----------------
template <int V, int TRI>
__device__ __forceinline__ void traceOne(const float4* __restrict__ nodes, const float4* __restrict__ tris, int i, const float4* __restrict__ pO, const float4* __restrict__ pD, float4* hits) {
    float4 O = pO[i], D = pD[i]; float hitT = D.w; int tri = -1; float u = 0, v = 0;
    trav<V, false, TRI>(nodes, tris, xyz(O), xyz(D), O.w, hitT, tri, u, v);
    hits[i] = make_float4(hitT, u, v, __int_as_float(tri));
}
template <int V, int TRI>
__device__ __forceinline__ void shadowOne(const float4* __restrict__ nodes, const float4* __restrict__ tris, int k, const float4* __restrict__ sO, const float4* __restrict__ sD, const float4* __restrict__ sC, float4* accum) {
    float4 O = sO[k], D = sD[k]; float hitT = D.w; int tri = -1; float u, v;
    bool occ = trav<V, true, TRI>(nodes, tris, xyz(O), xyz(D), O.w, hitT, tri, u, v);
    if (!occ) { float4 c = sC[k]; int pix = __float_as_int(c.w); atomicAdd(&accum[pix].x, c.x); atomicAdd(&accum[pix].y, c.y); atomicAdd(&accum[pix].z, c.z); }
}
template <int V, int TRI>
__global__ void LB k_trace(const float4* __restrict__ nodes, const float4* __restrict__ tris, const int* __restrict__ q, int n, const float4* __restrict__ pO, const float4* __restrict__ pD, float4* hits) {
    int k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= n) return;
    traceOne<V, TRI>(nodes, tris, q[k], pO, pD, hits);
}
template <int V, int TRI>
__global__ void LB k_shadow(const float4* __restrict__ nodes, const float4* __restrict__ tris, int n, const float4* __restrict__ sO, const float4* __restrict__ sD, const float4* __restrict__ sC, float4* accum) {
    int k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= n) return;
    shadowOne<V, TRI>(nodes, tris, k, sO, sD, sC, accum);
}
// device-driven versions: queue count read from device memory, warp-chunk fetch
template <int V, int TRI>
__global__ void LB k_trace_dev(const float4* __restrict__ nodes, const float4* __restrict__ tris, const int* __restrict__ q, const int* nPtr, int* fetch, const float4* __restrict__ pO, const float4* __restrict__ pD, float4* hits) {
    WARP_LOOP(nPtr, fetch, k) traceOne<V, TRI>(nodes, tris, q[k], pO, pD, hits);
}
template <int V, int TRI>
__global__ void LB k_shadow_dev(const float4* __restrict__ nodes, const float4* __restrict__ tris, const int* nPtr, int* fetch, const float4* __restrict__ sO, const float4* __restrict__ sD, const float4* __restrict__ sC, float4* accum) {
    WARP_LOOP(nPtr, fetch, k) shadowOne<V, TRI>(nodes, tris, k, sO, sD, sC, accum);
}

// ---------------- shading ----------------
template <int V, int TRI, bool FS>
__device__ __forceinline__ void shadeOne(int i, float4 h, const float4* __restrict__ tris, const float4* __restrict__ nrm, const float3* __restrict__ albedo,
                                         float4* pO, float4* pD, float4* pT, int* pDepth, float4* accum, int* qOut, int* qOutCnt,
                                         float4* sO, float4* sD, float4* sC, int* sCnt, unsigned* keys, int smp, const float4* __restrict__ nodes, const float4* __restrict__ trav_) {
    int tri = __float_as_int(h.w);
    float4 T4 = pT[i]; float3 T = xyz(T4); int pix = __float_as_int(T4.w);
    if (tri < 0) { atomicAdd(&accum[pix].x, T.x * P.sky.x); atomicAdd(&accum[pix].y, T.y * P.sky.y); atomicAdd(&accum[pix].z, T.z * P.sky.z); return; }
    int depth = pDepth[i];
    if (depth > P.maxB) return;
    float4 O = pO[i], D = pD[i]; float3 o = xyz(O), d = xyz(D);
    float3 p = o + d * h.x;
    float4 b = tris[3 * tri + 1], c = tris[3 * tri + 2];
    float3 e1 = xyz(b), e2 = xyz(c);
    float3 ng = normalize(cross(e1, e2));
    float3 ns = ng;
    if (P.hasN) { float u = h.y, v = h.z; float3 n0 = xyz(nrm[3 * tri]), n1 = xyz(nrm[3 * tri + 1]), n2 = xyz(nrm[3 * tri + 2]); float3 nn = n0 * (1.f - u - v) + n1 * u + n2 * v; float l = length(nn); if (l > 0.f) ns = nn * (1.f / l); if (dot(ns, ng) < 0.f) ns = -ns; }
    if (dot(ng, d) > 0.f) { ng = -ng; ns = -ns; }
    int mat = __float_as_int(b.w); float3 alb = albedo[mat];
    float3 po = p + ng * P.eps;
    {
        float u0 = rnd(pix, smp, depth + 1, 2), u1 = rnd(pix, smp, depth + 1, 3);
        float cz = 1.f - u0 * (1.f - P.sunCos); float sz = sqrtf(fmaxf(0.f, 1.f - cz * cz)); float ph = 6.2831853f * u1;
        float3 t, bb; basis(P.sunTo, t, bb);
        float3 l = normalize(P.sunTo * cz + t * (sz * cosf(ph)) + bb * (sz * sinf(ph)));
        float cl = dot(ns, l);
        if (cl > 0.f && dot(ng, l) > 0.f) {
            float3 contrib = T * alb * P.sunE * (cl * 0.31830989f);
            int s = atomicAdd(sCnt, 1);
            if (FS) { float st = 1e30f; int stri = -1; float su, sv; if (!trav<V, true, TRI>(nodes, trav_, po, l, 0.f, st, stri, su, sv)) { atomicAdd(&accum[pix].x, contrib.x); atomicAdd(&accum[pix].y, contrib.y); atomicAdd(&accum[pix].z, contrib.z); } }
            else { sO[s] = mk4(po, 0.f); sD[s] = mk4(l, 1e30f); sC[s] = mk4(contrib, __int_as_float(pix)); }
        }
    }
    float u0 = rnd(pix, smp, depth + 1, 0), u1 = rnd(pix, smp, depth + 1, 1);
    float r = sqrtf(u0), ph = 6.2831853f * u1; float3 t, bb; basis(ns, t, bb);
    float3 nd = normalize(t * (r * cosf(ph)) + bb * (r * sinf(ph)) + ns * sqrtf(fmaxf(0.f, 1.f - u0)));
    if (dot(nd, ng) <= 0.f) return;
    T = T * alb;
    pO[i] = mk4(po, 0.f); pD[i] = mk4(nd, 1e30f); pT[i] = mk4(T, __int_as_float(pix)); pDepth[i] = depth + 1;
    if (keys) { // binning key: direction octant (3 bits) + Morton of origin (4 bits/axis) in the scene box
        float3 rr = (po - P.bmin) * P.binv; unsigned qx = (unsigned)fminf(fmaxf(rr.x, 0.f), 15.f), qy = (unsigned)fminf(fmaxf(rr.y, 0.f), 15.f), qz = (unsigned)fminf(fmaxf(rr.z, 0.f), 15.f);
        unsigned mo = 0; for (int bI = 0; bI < 4; bI++) mo |= ((qx >> bI) & 1u) << (3 * bI) | ((qy >> bI) & 1u) << (3 * bI + 1) | ((qz >> bI) & 1u) << (3 * bI + 2);
        unsigned oct = (nd.x > 0.f ? 1u : 0u) | (nd.y > 0.f ? 2u : 0u) | (nd.z > 0.f ? 4u : 0u);
        keys[i] = (oct << 12) | mo;
    }
    int s = atomicAdd(qOutCnt, 1); qOut[s] = i;
}
__global__ void k_shade(const int* __restrict__ q, int n, const float4* __restrict__ hits, const float4* __restrict__ tris, const float4* __restrict__ nrm, const float3* __restrict__ albedo,
                        float4* pO, float4* pD, float4* pT, int* pDepth, float4* accum, int* qOut, int* qOutCnt, float4* sO, float4* sD, float4* sC, int* sCnt, unsigned* keys, int smpBase, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= n) return;
    int i = q[k]; shadeOne<2, 0, false>(i, hits[i], tris, nrm, albedo, pO, pD, pT, pDepth, accum, qOut, qOutCnt, sO, sD, sC, sCnt, keys, smpBase + i / N, nullptr, nullptr);
}
// fused closest-hit trace + shade (no hit buffer round trip, one launch less per bounce)
template <int V, int TRI, bool FS>
__global__ void LB k_trace_shade(const float4* __restrict__ nodes, const float4* __restrict__ trav_, const int* __restrict__ q, int n, const float4* __restrict__ tris, const float4* __restrict__ nrm, const float3* __restrict__ albedo,
                                 float4* pO, float4* pD, float4* pT, int* pDepth, float4* accum, int* qOut, int* qOutCnt, float4* sO, float4* sD, float4* sC, int* sCnt, int smpBase, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= n) return;
    int i = q[k];
    float4 O = pO[i], D = pD[i]; float hitT = D.w; int tri = -1; float u = 0, v = 0;
    trav<V, false, TRI>(nodes, trav_, xyz(O), xyz(D), O.w, hitT, tri, u, v);
    shadeOne<V, TRI, FS>(i, make_float4(hitT, u, v, __int_as_float(tri)), tris, nrm, albedo, pO, pD, pT, pDepth, accum, qOut, qOutCnt, sO, sD, sC, sCnt, nullptr, smpBase + i / N, nodes, trav_);
}
__global__ void k_shade_dev(const int* __restrict__ q, const int* nPtr, int* fetch, const float4* __restrict__ hits, const float4* __restrict__ tris, const float4* __restrict__ nrm, const float3* __restrict__ albedo,
                            float4* pO, float4* pD, float4* pT, int* pDepth, float4* accum, int* qOut, int* qOutCnt, float4* sO, float4* sD, float4* sC, int* sCnt, unsigned* keys, int smpBase, int N) {
    WARP_LOOP(nPtr, fetch, k) { int i = q[k]; shadeOne<2, 0, false>(i, hits[i], tris, nrm, albedo, pO, pD, pT, pDepth, accum, qOut, qOutCnt, sO, sD, sC, sCnt, keys, smpBase + i / N, nullptr, nullptr); }
}
// ---------------- counting sort of a queue by key (device-driven) ----------------
__global__ void k_hist(const int* __restrict__ q, const int* nPtr, int* fetch, const unsigned* __restrict__ keys, int* hist) { WARP_LOOP(nPtr, fetch, k) atomicAdd(&hist[keys[q[k]]], 1); }
__global__ void k_scan(int* hist) { // single block, 1024 threads, NBINS bins -> exclusive offsets in place
    __shared__ int sh[1024]; int tid = threadIdx.x; const int per = NBINS / 1024; int base = tid * per; int sum = 0;
    for (int j = 0; j < per; j++) sum += hist[base + j];
    sh[tid] = sum; __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) { int v = tid >= off ? sh[tid - off] : 0; __syncthreads(); sh[tid] += v; __syncthreads(); }
    int run = sh[tid] - sum;
    for (int j = 0; j < per; j++) { int c = hist[base + j]; hist[base + j] = run; run += c; }
}
__global__ void k_scatter(const int* __restrict__ q, const int* nPtr, int* fetch, const unsigned* __restrict__ keys, int* cursor, int* qOut) { WARP_LOOP(nPtr, fetch, k) { int i = q[k]; int pos = atomicAdd(&cursor[keys[i]], 1); qOut[pos] = i; } }
__global__ void k_sumcounts(const int* cnt, int depths, unsigned long long* tot) { if (threadIdx.x == 0) { unsigned long long a = 0, b = 0; for (int d = 0; d < depths; d++) { a += cnt[d * 8]; b += cnt[d * 8 + 1]; } tot[0] += a; tot[1] += b; } }

// ---------------- host ----------------
struct Args {
    std::string mesh, camera, out = "out", variant = "bvh2", c2wKey = "c2w_blender_zup", tri = "mt";
    int fuse = 0, leaf = 4, batch = 64, synthetic = 0, frame = 0, w = 512, h = 512, spp = 256, bounces = 8, persistent = 0, filterBH = 1, smooth = 1, warm = 8, yupToZup = 0, reps = 1, devloop = 0, sort = 0;
    float fx = 0, fy = 0, hfov = 0; unsigned seed = 1000;
    float3 sunTo = mk3(0.3f, -0.5f, 0.8f), sunE = mk3(3.f, 3.f, 3.f), sky = mk3(0.3f, 0.3f, 0.3f), albedo = mk3(-1, 0, 0);
    float sunAngle = 0.526f, epsScale = 1e-4f, near = 0.f;
    std::string c2w, lighting;
};
static float3 parse3(char** a, int& i) { float3 v = mk3((float)atof(a[i + 1]), (float)atof(a[i + 2]), (float)atof(a[i + 3])); i += 3; return v; }
static bool jsonNumber(const std::string& s, const std::string& key, double& out) {
    size_t k = s.find("\"" + key + "\""); if (k == std::string::npos) return false; k = s.find(':', k); if (k == std::string::npos) return false;
    out = atof(s.c_str() + k + 1); return true;
}
static std::vector<double> jsonNumbers(const std::string& s, const std::string& key) {
    std::vector<double> nums; size_t k = s.find("\"" + key + "\""); if (k == std::string::npos) return nums; k = s.find('[', k); if (k == std::string::npos) return nums;
    int depth = 0;
    for (size_t i = k; i < s.size(); i++) {
        char c = s[i];
        if (c == '[') depth++; else if (c == ']') { depth--; if (depth == 0) break; }
        else if ((c >= '0' && c <= '9') || c == '-' || c == '.') { char* e; nums.push_back(strtod(s.c_str() + i, &e)); i = (e - s.c_str()) - 1; }
    }
    return nums;
}
static bool jsonMatrix(const std::string& s, const std::string& key, int frame, float* m16) {
    std::vector<double> nums = jsonNumbers(s, key);
    if (nums.size() < 16) return false;
    size_t off = (size_t)frame * 16; if (nums.size() == 16) off = 0; if (off + 16 > nums.size()) { fprintf(stderr, "frame %d out of range\n", frame); return false; }
    for (int j = 0; j < 16; j++) m16[j] = (float)nums[off + j];
    return true;
}
static std::string readFile(const std::string& p) { FILE* fp = fopen(p.c_str(), "rb"); if (!fp) return ""; std::string s; char buf[65536]; size_t n; while ((n = fread(buf, 1, sizeof buf, fp)) > 0) s.append(buf, n); fclose(fp); return s; }

int main(int argc, char** argv) {
    Args A;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto nf = [&]() { return (float)atof(argv[++i]); }; auto ni = [&]() { return atoi(argv[++i]); };
        if (k == "--mesh") A.mesh = argv[++i]; else if (k == "--camera") A.camera = argv[++i]; else if (k == "--out") A.out = argv[++i];
        else if (k == "--variant") A.variant = argv[++i]; else if (k == "--c2w_key") A.c2wKey = argv[++i]; else if (k == "--c2w") A.c2w = argv[++i]; else if (k == "--tri") A.tri = argv[++i];
        else if (k == "--synthetic") A.synthetic = ni(); else if (k == "--frame") A.frame = ni(); else if (k == "--width") A.w = ni(); else if (k == "--height") A.h = ni();
        else if (k == "--spp") A.spp = ni(); else if (k == "--bounces") A.bounces = ni(); else if (k == "--persistent") A.persistent = ni(); else if (k == "--filter") A.filterBH = std::string(argv[++i]) == "bh";
        else if (k == "--smooth") A.smooth = ni(); else if (k == "--warm") A.warm = ni(); else if (k == "--yup_to_zup") A.yupToZup = ni(); else if (k == "--reps") A.reps = ni(); else if (k == "--batch") A.batch = ni();
        else if (k == "--devloop") A.devloop = ni(); else if (k == "--sort") A.sort = ni(); else if (k == "--fuse") A.fuse = ni(); else if (k == "--leaf") A.leaf = ni();
        else if (k == "--fx") A.fx = nf(); else if (k == "--fy") A.fy = nf(); else if (k == "--hfov") A.hfov = nf(); else if (k == "--seed") A.seed = (unsigned)ni();
        else if (k == "--sun_to") A.sunTo = parse3(argv, i); else if (k == "--sun_rgb") A.sunE = parse3(argv, i); else if (k == "--sky") A.sky = parse3(argv, i); else if (k == "--albedo") A.albedo = parse3(argv, i);
        else if (k == "--sun_angle") A.sunAngle = nf(); else if (k == "--eps") A.epsScale = nf(); else if (k == "--near") A.near = nf(); else if (k == "--lighting") A.lighting = argv[++i];
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (A.sort) A.devloop = 1;
    // ---- scene ----
    Mesh M; auto t0 = std::chrono::steady_clock::now();
    float c2w[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 }; bool haveC2w = false;
    if (A.synthetic) {
        makeSynthetic(M, A.synthetic, 200, 100, 200);
        float3 pos = mk3(0.f, -9.5f, 6.5f), tgt = mk3(0.f, 0.f, 0.4f), up = mk3(0, 0, 1);
        float3 f = normalize(tgt - pos), r = normalize(cross(f, up)), u = cross(r, f);
        float3 az = -f; float cols[3][3] = { { r.x, r.y, r.z }, { u.x, u.y, u.z }, { az.x, az.y, az.z } };
        for (int row = 0; row < 3; row++) for (int col = 0; col < 3; col++) c2w[row * 4 + col] = cols[col][row];
        c2w[3] = pos.x; c2w[7] = pos.y; c2w[11] = pos.z; haveC2w = true;
        if (A.hfov == 0 && A.fx == 0) A.hfov = 50.f;
    } else {
        if (!loadMesh(A.mesh, M)) { fprintf(stderr, "failed to load mesh %s\n", A.mesh.c_str()); return 1; }
        if (A.yupToZup) { for (auto& v : M.v) v = mk3(v.x, -v.z, v.y); for (auto& n : M.n) n = mk3(n.x, -n.z, n.y); }
    }
    if (!A.smooth) M.n.clear();
    if (!A.lighting.empty()) {
        std::string js = readFile(A.lighting); if (js.empty()) { fprintf(stderr, "lighting file %s unreadable\n", A.lighting.c_str()); return 1; }
        std::vector<double> v = jsonNumbers(js, "sun_dir_to_light_world_zup"); if (v.size() == 3) A.sunTo = mk3((float)v[0], (float)v[1], (float)v[2]);
        double x; if (jsonNumber(js, "sun_irradiance_W_m2", x)) A.sunE = mk3((float)x, (float)x, (float)x);
        if (jsonNumber(js, "sun_angle_rad", x)) A.sunAngle = (float)(x * 180.0 / M_PI);
        v = jsonNumbers(js, "sky_radiance_rgb_linear"); if (v.size() == 3) A.sky = mk3((float)v[0], (float)v[1], (float)v[2]);
        v = jsonNumbers(js, "albedo_rgb_linear_by_material_index");
        if (v.size() >= 3) { M.kd.clear(); for (size_t j = 0; j + 2 < v.size(); j += 3) M.kd.push_back(mk3((float)v[j], (float)v[j + 1], (float)v[j + 2])); }
        int maxMat = 0; for (int t : M.fm) if (t > maxMat) maxMat = t;
        while ((int)M.kd.size() <= maxMat) M.kd.push_back(mk3(0.8f, 0.8f, 0.8f));
    }
    if (A.albedo.x >= 0) for (auto& kd : M.kd) kd = A.albedo;
    if (M.kd.empty()) M.kd.push_back(mk3(0.8f, 0.8f, 0.8f));
    double loadS = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    AABB sb; for (auto& v : M.v) sb.grow(v);
    float extent = maxc(sb.extent());
    fprintf(stderr, "mesh: %zu verts %zu tris, normals=%d, mats=%zu, extent=%.3f load %.2fs\n", M.v.size(), M.f.size(), (int)M.hasN(), M.kd.size(), extent, loadS);

    // ---- camera ----
    if (!A.camera.empty()) {
        std::string js = readFile(A.camera); if (js.empty()) { fprintf(stderr, "camera file %s unreadable\n", A.camera.c_str()); return 1; }
        double v;
        if (A.fx == 0 && jsonNumber(js, "fx", v)) {
            double fw = 0, fh = 0; jsonNumber(js, "width", fw); jsonNumber(js, "height", fh);
            if ((int)fw == A.w && (int)fh == A.h) { A.fx = (float)v; jsonNumber(js, "fy", v); A.fy = (float)v; }
            else if (A.hfov == 0 && jsonNumber(js, "hfov_deg", v)) A.hfov = (float)v;
        }
        if (!jsonMatrix(js, A.c2wKey, A.frame, c2w)) { fprintf(stderr, "no matrix %s in camera file\n", A.c2wKey.c_str()); return 1; }
        haveC2w = true;
    }
    if (!A.c2w.empty()) { std::string s = A.c2w; for (auto& ch : s) if (ch == ',') ch = ' '; const char* p = s.c_str(); char* e; for (int j = 0; j < 16; j++) { c2w[j] = (float)strtod(p, &e); p = e; } haveC2w = true; }
    if (!haveC2w) { fprintf(stderr, "no camera\n"); return 1; }
    if (A.fx == 0) { if (A.hfov == 0) A.hfov = 50.f; A.fx = (A.w * 0.5f) / tanf(A.hfov * 0.5f * 3.14159265f / 180.f); }
    if (A.fy == 0) A.fy = A.fx;
    Params H; memset(&H, 0, sizeof H);
    H.cam.pos = mk3(c2w[3], c2w[7], c2w[11]);
    H.cam.ax = mk3(c2w[0], c2w[4], c2w[8]); H.cam.ay = mk3(c2w[1], c2w[5], c2w[9]); H.cam.az = mk3(c2w[2], c2w[6], c2w[10]);
    H.cam.fx = A.fx; H.cam.fy = A.fy; H.cam.px = A.w * 0.5f; H.cam.py = A.h * 0.5f; H.cam.w = A.w; H.cam.h = A.h;
    H.sunTo = normalize(A.sunTo); H.sunE = A.sunE; H.sky = A.sky; H.sunCos = cosf(0.5f * A.sunAngle * 3.14159265f / 180.f);
    H.maxB = A.bounces; H.seed = A.seed; H.eps = A.epsScale * extent; H.near = A.near; H.hasN = M.hasN() ? 1 : 0; H.filterBH = A.filterBH;
    H.bmin = sb.lo; float3 ex3 = sb.extent(); H.binv = mk3(16.f / fmaxf(ex3.x, 1e-6f), 16.f / fmaxf(ex3.y, 1e-6f), 16.f / fmaxf(ex3.z, 1e-6f));
    {
        const int NS = 4096; std::vector<double> cdf(NS + 1, 0.0); const double w = 1.5;
        for (int i = 0; i < NS; i++) { double t = (i + 0.5) / NS; double f = 0.35875 - 0.48829 * cos(2 * M_PI * t) + 0.14128 * cos(4 * M_PI * t) - 0.01168 * cos(6 * M_PI * t); cdf[i + 1] = cdf[i] + fmax(f, 0.0); }
        for (int i = 0; i <= NS; i++) cdf[i] /= cdf[NS];
        for (int k = 0; k <= FT; k++) { double u = (double)k / FT; int j = 0; while (j < NS - 1 && cdf[j + 1] < u) j++; double t = (cdf[j + 1] > cdf[j]) ? (u - cdf[j]) / (cdf[j + 1] - cdf[j]) : 0.0; H.ftab[k] = (float)(((j + t) / NS - 0.5) * w); }
    }
    fprintf(stderr, "cam pos (%.3f %.3f %.3f) fx %.2f %dx%d spp %d bounces %d eps %.2e sunTo (%.3f %.3f %.3f)\n", H.cam.pos.x, H.cam.pos.y, H.cam.pos.z, A.fx, A.w, A.h, A.spp, A.bounces, H.eps, H.sunTo.x, H.sunTo.y, H.sunTo.z);

    // ---- BVH ----
    t0 = std::chrono::steady_clock::now();
    BVHBuilder B; B.leafMax = (A.variant == "bvh8" && A.leaf > 4) ? 4 : A.leaf; B.build(M);
    std::vector<float4> gnodes; int nodeStride; int V = A.variant == "bvh4" ? 4 : A.variant == "bvh8" ? 8 : 2; int TRI = A.tri == "woop" ? 1 : 0;
    std::vector<int> order;
    if (V == 2) { BVH2Flat F; F.build(B); gnodes.swap(F.nodes); nodeStride = 4; order = B.idx; }
    else if (V == 4) { BVH4Flat F; F.build(B); gnodes.swap(F.nodes); nodeStride = 8; order = B.idx; }
    else { BVH8Flat F; F.build(B); gnodes.swap(F.nodes); nodeStride = 5; order.swap(F.order); }
    std::vector<float4> gtris, gnrm, gwoop; packTriangles(M, order, gtris, gnrm);
    if (TRI == 1) packWoop(M, order, gwoop);
    double bvhS = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    size_t nNodes = gnodes.size() / nodeStride;
    fprintf(stderr, "BVH%d: %zu nodes (%.1f MB), depth %d, build %.2fs, tri=%s\n", V, nNodes, gnodes.size() * 16 / 1e6, B.maxDepth, bvhS, A.tri.c_str());

    // ---- GPU ----
    int dev = 0; CK(cudaSetDevice(dev)); cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    size_t memFree0, memTot; CK(cudaMemGetInfo(&memFree0, &memTot));
    float4 *dNodes, *dTris, *dNrm = nullptr, *dWoop = nullptr; float3* dAlb;
    CK(cudaMalloc(&dNodes, gnodes.size() * 16)); CK(cudaMemcpy(dNodes, gnodes.data(), gnodes.size() * 16, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dTris, gtris.size() * 16)); CK(cudaMemcpy(dTris, gtris.data(), gtris.size() * 16, cudaMemcpyHostToDevice));
    if (TRI == 1) { CK(cudaMalloc(&dWoop, gwoop.size() * 16)); CK(cudaMemcpy(dWoop, gwoop.data(), gwoop.size() * 16, cudaMemcpyHostToDevice)); }
    const float4* dTrav = TRI == 1 ? dWoop : dTris;
    if (!gnrm.empty()) { CK(cudaMalloc(&dNrm, gnrm.size() * 16)); CK(cudaMemcpy(dNrm, gnrm.data(), gnrm.size() * 16, cudaMemcpyHostToDevice)); }
    CK(cudaMalloc(&dAlb, M.kd.size() * sizeof(float3))); CK(cudaMemcpy(dAlb, M.kd.data(), M.kd.size() * sizeof(float3), cudaMemcpyHostToDevice));
    CK(cudaMemcpyToSymbol(P, &H, sizeof H));
    int N = A.w * A.h; int K = A.batch; if (K < 1) K = 1; if (A.spp % K) { fprintf(stderr, "spp must be a multiple of batch\n"); return 1; } size_t NK = (size_t)N * K;
    int DEPTHS = A.bounces + 2;
    float4 *pO, *pD, *pT, *hits, *sO, *sD, *sC, *accum; int *pDepth, *q0, *q1, *q2, *cnt, *hist; unsigned* keys = nullptr; unsigned long long* dtot;
    CK(cudaMalloc(&pO, NK * 16)); CK(cudaMalloc(&pD, NK * 16)); CK(cudaMalloc(&pT, NK * 16)); CK(cudaMalloc(&hits, NK * 16));
    CK(cudaMalloc(&sO, NK * 16)); CK(cudaMalloc(&sD, NK * 16)); CK(cudaMalloc(&sC, NK * 16)); CK(cudaMalloc(&accum, N * 16));
    CK(cudaMalloc(&pDepth, NK * 4)); CK(cudaMalloc(&q0, NK * 4)); CK(cudaMalloc(&q1, NK * 4)); CK(cudaMalloc(&q2, NK * 4));
    CK(cudaMalloc(&cnt, (DEPTHS + 1) * 8 * 4)); CK(cudaMalloc(&hist, NBINS * 4)); CK(cudaMalloc(&dtot, 16));
    if (A.sort) CK(cudaMalloc(&keys, NK * 4));
    size_t memFree1; CK(cudaMemGetInfo(&memFree1, &memTot));
    double vramMB = (double)(memFree0 - memFree1) / 1048576.0;
    int* hcnt; CK(cudaMallocHost(&hcnt, 64)); int* hNK; CK(cudaMallocHost(&hNK, 16)); hNK[0] = (int)NK;
    int ptBlocks = 0;
    {
        int bpm = 0;
#define OCC(VV, TT) CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpm, k_trace_dev<VV, TT>, 128, 0))
        if (V == 2) { if (TRI) OCC(2, 1); else OCC(2, 0); } else if (V == 4) { if (TRI) OCC(4, 1); else OCC(4, 0); } else { if (TRI) OCC(8, 1); else OCC(8, 0); }
#undef OCC
        ptBlocks = bpm * prop.multiProcessorCount;
        fprintf(stderr, "trace kernel: %d blocks/SM x %d SMs (occupancy %.0f%%)\n", bpm, prop.multiProcessorCount, 100.0 * bpm * 128 / prop.maxThreadsPerMultiProcessor);
    }
    const int TB = 128;
    // dispatch helpers (V, TRI runtime -> template)
#define DISPATCH(F, G, T, ARGS) do {         if (V == 2) { if (TRI) F<2, 1><<<G, T>>>ARGS; else F<2, 0><<<G, T>>>ARGS; }         else if (V == 4) { if (TRI) F<4, 1><<<G, T>>>ARGS; else F<4, 0><<<G, T>>>ARGS; }         else { if (TRI) F<8, 1><<<G, T>>>ARGS; else F<8, 0><<<G, T>>>ARGS; } } while (0)
#define DISPATCH3(F, FS, G, T, ARGS) do { \
        if (V == 2) { if (TRI) F<2, 1, FS><<<G, T>>>ARGS; else F<2, 0, FS><<<G, T>>>ARGS; } \
        else if (V == 4) { if (TRI) F<4, 1, FS><<<G, T>>>ARGS; else F<4, 0, FS><<<G, T>>>ARGS; } \
        else { if (TRI) F<8, 1, FS><<<G, T>>>ARGS; else F<8, 0, FS><<<G, T>>>ARGS; } } while (0)
    int* qOut_ = nullptr; int smp_ = 0;
    auto launchTrace = [&](int* q, int n) {
        if (n <= 0) return;
        if (A.fuse == 1) { int g = (n + TB - 1) / TB; DISPATCH3(k_trace_shade, false, g, TB, (dNodes, dTrav, q, n, dTris, dNrm, dAlb, pO, pD, pT, pDepth, accum, qOut_, cnt, sO, sD, sC, cnt + 1, smp_, N)); return; }
        if (A.fuse == 2) { int g = (n + TB - 1) / TB; DISPATCH3(k_trace_shade, true, g, TB, (dNodes, dTrav, q, n, dTris, dNrm, dAlb, pO, pD, pT, pDepth, accum, qOut_, cnt, sO, sD, sC, cnt + 1, smp_, N)); return; }
        if (A.persistent) { int g = ptBlocks < (n + 31) / 32 ? ptBlocks : (n + 31) / 32; CK(cudaMemsetAsync(cnt + 2, 0, 4)); DISPATCH(k_trace_dev, g, TB, (dNodes, dTrav, q, cnt + 8 * DEPTHS, cnt + 2, pO, pD, hits)); }
        else { int g = (n + TB - 1) / TB; DISPATCH(k_trace, g, TB, (dNodes, dTrav, q, n, pO, pD, hits)); }
    };
    auto launchShadow = [&](int n) {
        if (n <= 0) return;
        if (A.persistent) { int g = ptBlocks < (n + 31) / 32 ? ptBlocks : (n + 31) / 32; CK(cudaMemsetAsync(cnt + 3, 0, 4)); DISPATCH(k_shadow_dev, g, TB, (dNodes, dTrav, cnt + 8 * DEPTHS + 1, cnt + 3, sO, sD, sC, accum)); }
        else { int g = (n + TB - 1) / TB; DISPATCH(k_shadow, g, TB, (dNodes, dTrav, n, sO, sD, sC, accum)); }
    };
    std::vector<cudaEvent_t> evs; size_t evUsed = 0;
    auto ev = [&]() { if (evUsed == evs.size()) { cudaEvent_t e; CK(cudaEventCreate(&e)); evs.push_back(e); } return evs[evUsed++]; };
    cudaEvent_t f0, f1; CK(cudaEventCreate(&f0)); CK(cudaEventCreate(&f1));

    struct Stats { double rays = 0, shadowRays = 0, traceMs = 0, frameMs = 0; };
    auto render = [&](int spp, Stats& S) {
        CK(cudaMemset(accum, 0, N * 16)); CK(cudaMemset(dtot, 0, 16)); evUsed = 0;
        std::vector<std::pair<cudaEvent_t, cudaEvent_t>> pairs;
        CK(cudaEventRecord(f0));
        for (int s = 0; s < spp; s += K) {
            k_raygen<<<(int)((NK + 255) / 256), 256>>>(pO, pD, pT, pDepth, q0, s, K);
            int* qIn = q0; int* qOut = q1; int* qS = q2; int n = (int)NK;
            if (!A.devloop) {
                for (int depth = 0; depth < DEPTHS && n > 0; depth++) {
                    CK(cudaMemsetAsync(cnt, 0, 16));
                    // persistent kernels read counts from cnt+8*DEPTHS: set them to (n, nShadow)
                    if (A.persistent) { hNK[0] = n; CK(cudaMemcpyAsync(cnt + 8 * DEPTHS, hNK, 4, cudaMemcpyHostToDevice)); }
                    qOut_ = qOut; smp_ = s;
                    cudaEvent_t a = ev(), b = ev(); CK(cudaEventRecord(a)); launchTrace(qIn, n); CK(cudaEventRecord(b)); pairs.push_back({ a, b });
                    S.rays += n;
                    if (A.fuse) {} else
                    k_shade<<<(n + 127) / 128, 128>>>(qIn, n, hits, dTris, dNrm, dAlb, pO, pD, pT, pDepth, accum, qOut, cnt, sO, sD, sC, cnt + 1, nullptr, s, N);
                    CK(cudaMemcpyAsync(hcnt, cnt, 8, cudaMemcpyDeviceToHost)); CK(cudaStreamSynchronize(0));
                    int nNext = hcnt[0], nSh = hcnt[1];
                    if (A.fuse == 2) S.shadowRays += nSh;
                    else if (nSh > 0) {
                        if (A.persistent) { hNK[1] = nSh; CK(cudaMemcpyAsync(cnt + 8 * DEPTHS + 1, hNK + 1, 4, cudaMemcpyHostToDevice)); }
                        cudaEvent_t c = ev(), d = ev(); CK(cudaEventRecord(c)); launchShadow(nSh); CK(cudaEventRecord(d)); pairs.push_back({ c, d }); S.shadowRays += nSh;
                    }
                    int* t = qIn; qIn = qOut; qOut = t; n = nNext;
                }
            } else {
                CK(cudaMemsetAsync(cnt, 0, DEPTHS * 8 * 4)); CK(cudaMemcpyAsync(cnt, hNK, 4, cudaMemcpyHostToDevice));
                for (int depth = 0; depth < DEPTHS; depth++) {
                    int* c = cnt + depth * 8;
                    if (A.sort && depth > 0) {
                        CK(cudaMemsetAsync(hist, 0, NBINS * 4));
                        k_hist<<<ptBlocks, TB>>>(qIn, c, c + 5, keys, hist);
                        k_scan<<<1, 1024>>>(hist);
                        k_scatter<<<ptBlocks, TB>>>(qIn, c, c + 6, keys, hist, qS);
                        int* t = qIn; qIn = qS; qS = t;
                    }
                    cudaEvent_t a = ev(), b = ev(); CK(cudaEventRecord(a));
                    DISPATCH(k_trace_dev, ptBlocks, TB, (dNodes, dTrav, qIn, c, c + 2, pO, pD, hits));
                    CK(cudaEventRecord(b)); pairs.push_back({ a, b });
                    k_shade_dev<<<ptBlocks, TB>>>(qIn, c, c + 4, hits, dTris, dNrm, dAlb, pO, pD, pT, pDepth, accum, qOut, c + 8, sO, sD, sC, c + 1, keys, s, N);
                    cudaEvent_t cc = ev(), d = ev(); CK(cudaEventRecord(cc));
                    DISPATCH(k_shadow_dev, ptBlocks, TB, (dNodes, dTrav, c + 1, c + 3, sO, sD, sC, accum));
                    CK(cudaEventRecord(d)); pairs.push_back({ cc, d });
                    int* t = qIn; qIn = qOut; qOut = t;
                }
                k_sumcounts<<<1, 32>>>(cnt, DEPTHS, dtot);
            }
        }
        CK(cudaEventRecord(f1)); CK(cudaEventSynchronize(f1));
        float ms; CK(cudaEventElapsedTime(&ms, f0, f1)); S.frameMs = ms;
        for (auto& pr : pairs) { CK(cudaEventElapsedTime(&ms, pr.first, pr.second)); S.traceMs += ms; }
        if (A.devloop) { unsigned long long tot[2]; CK(cudaMemcpy(tot, dtot, 16, cudaMemcpyDeviceToHost)); S.rays = (double)tot[0]; S.shadowRays = (double)tot[1]; }
        CK(cudaGetLastError());
    };
    Stats W; if (A.warm > 0) { int wsp = ((A.warm + K - 1) / K) * K; render(wsp, W); }
    fprintf(stderr, "warmup: %.1f ms, %.1f Mrays/s\n", W.frameMs, (W.rays + W.shadowRays) / (W.traceMs * 1e3));
    for (int r = 0; r < A.reps; r++) {
        Stats R; render(A.spp, R);
        double tot = R.rays + R.shadowRays;
        printf("{\"rep\":%d,\"variant\":\"bvh%d\",\"tri\":\"%s\",\"persistent\":%d,\"devloop\":%d,\"sort\":%d,\"fuse\":%d,\"leaf\":%d,\"batch\":%d,\"minb\":%d,\"scene\":\"%s\",\"tris\":%zu,\"bvh_nodes\":%zu,\"bvh_bytes\":%zu,\"bvh_depth\":%d,\"bvh_build_s\":%.2f,\"w\":%d,\"h\":%d,\"spp\":%d,\"bounces\":%d,"
               "\"rays_closest\":%.0f,\"rays_shadow\":%.0f,\"rays_total\":%.0f,\"t_trace_ms\":%.2f,\"t_frame_ms\":%.2f,\"mrays_s_trace\":%.1f,\"mrays_s_e2e\":%.1f,\"avg_path_len\":%.3f,\"vram_MB\":%.0f,\"pt_blocks\":%d,\"gpu\":\"%s\",\"filter\":\"%s\",\"smooth\":%d,\"seed\":%u}\n",
               r, V, A.tri.c_str(), A.persistent, A.devloop, A.sort, A.fuse, B.leafMax, K,
#ifdef MINB
               MINB,
#else
               0,
#endif
               A.synthetic ? "synthetic" : A.mesh.c_str(), M.f.size(), nNodes, gnodes.size() * 16, B.maxDepth, bvhS, A.w, A.h, A.spp, A.bounces,
               R.rays, R.shadowRays, tot, R.traceMs, R.frameMs, tot / (R.traceMs * 1e3), tot / (R.frameMs * 1e3), R.rays / ((double)N * A.spp), vramMB, ptBlocks, prop.name, A.filterBH ? "blackman-harris1.5" : "box", H.hasN, A.seed);
        fflush(stdout);
    }
    std::vector<float4> acc(N); CK(cudaMemcpy(acc.data(), accum, N * 16, cudaMemcpyDeviceToHost));
    std::vector<float> lin(N * 3); std::vector<uint8_t> rgb(N * 3);
    for (int i = 0; i < N; i++) {
        float c[3] = { acc[i].x / A.spp, acc[i].y / A.spp, acc[i].z / A.spp };
        for (int j = 0; j < 3; j++) {
            lin[3 * i + j] = c[j];
            float x = fminf(fmaxf(c[j], 0.f), 1.f); x = x <= 0.0031308f ? 12.92f * x : 1.055f * powf(x, 1.f / 2.4f) - 0.055f;
            rgb[3 * i + j] = (uint8_t)(x * 255.f + 0.5f);
        }
    }
    writePNG(A.out + ".png", A.w, A.h, rgb.data()); writePFM(A.out + ".pfm", A.w, A.h, lin.data());
    fprintf(stderr, "wrote %s.png / .pfm (tonemap: clip[0,1] + sRGB OETF, exposure 0)\n", A.out.c_str());
    return 0;
}
