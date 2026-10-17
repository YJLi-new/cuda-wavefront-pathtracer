// render/wavefront.cu - wavefront CUDA path tracer kernels and host driver.
// Kernels: raygen / trace(closest) / shade / trace(shadow, any-hit), queue compaction via atomic append.
// Options: BVH2 / BVH4 / compressed BVH8 traversal (accel/traverse.cuh); Moller-Trumbore or Woop triangles;
// fused trace+shade; host-driven loop (one sync per bounce) or experimental device-driven loop (persistent-style
// kernels reading queue counts from device memory) with optional counting-sort binning of bounce rays by
// (direction octant, Morton code of origin). Legacy shading: Lambert + sun next-event estimation + constant sky.
// Full integrator (k_trace_shade_full / split + persistent kernels): OpenPBR-subset materials, area lights, environment map, MIS, RR;
// single-level or instanced (TLAS/BLAS) scenes; optional CUDA-graph launch of the per-frame kernel sequence.
#include "render/renderer.h"
#include "accel/traverse.cuh"
#include "core/rng.h"
#include "core/sampling.h"
#include "render/bsdf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#ifndef MINB
#define LB __launch_bounds__(128)
#else
#define LB __launch_bounds__(128, MINB)
#endif

int rendererMinBlocks() {
#ifdef MINB
    return MINB;
#else
    return 0;
#endif
}

__constant__ Params P;

__device__ __forceinline__ float rnd(unsigned pix, unsigned smp, unsigned depth, unsigned dim) { return rng4(pix, smp, depth, dim, P.seed); }
__device__ __forceinline__ int warpFetch(int* fetch) { int b = 0; if ((threadIdx.x & 31) == 0) b = atomicAdd(fetch, 32); return __shfl_sync(0xffffffffu, b, 0); }
// persistent-style loop over [0, *nPtr): each warp grabs 32-element chunks
#define WARP_LOOP(nPtr, fetch, k) for (int n_ = *(nPtr), b_ = warpFetch(fetch); b_ < n_; b_ = warpFetch(fetch)) if (int k = b_ + (threadIdx.x & 31); k < n_)

// ---------------- ray generation ----------------
__global__ void k_raygen(float4* pO, float4* pD, float4* pT, int* pDepth, int* q, int smpBase, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; int N = P.cam.w * P.cam.h; if (i >= N * K) return;
    int pix = i % N, smp = smpBase + i / N;
    int x = pix % P.cam.w, y = pix / P.cam.w;
    float u0 = rnd(pix, smp, 0, 0), u1 = rnd(pix, smp, 0, 1), jx, jy;
    if (P.filterBH) { jx = sampleTable(P.ftab, kFilterTable, u0); jy = sampleTable(P.ftab, kFilterTable, u1); } else { jx = u0 - 0.5f; jy = u1 - 0.5f; }
    float dx = (x + 0.5f + jx - P.cam.px) / P.cam.fx, dy = -(y + 0.5f + jy - P.cam.py) / P.cam.fy;
    float3 d = normalize(P.cam.ax * dx + P.cam.ay * dy - P.cam.az);
    float3 o = P.cam.pos;
    if (P.cam.lensR > 0.f) {   // thin lens: origin on the lens disk, aimed at the pinhole ray's point on the focal plane
        float lx, ly; sampleConcentricDisk(rnd(pix, smp, 0, 2), rnd(pix, smp, 0, 3), lx, ly);
        float3 pf = P.cam.pos + (P.cam.ax * dx + P.cam.ay * dy - P.cam.az) * P.cam.focusD;
        o = P.cam.pos + P.cam.ax * (lx * P.cam.lensR) + P.cam.ay * (ly * P.cam.lensR);
        d = normalize(pf - o);
    }
    pO[i] = mk4(o, P.near); pD[i] = mk4(d, 1e30f); pT[i] = make_float4(1.f, 1.f, 1.f, __int_as_float(pix)); pDepth[i] = 0; q[i] = i;
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
        // sun next-event estimation: uniform direction inside the sun's cone, Lambert BRDF alb / pi
        float3 l = sampleUniformCone(P.sunTo, P.sunCos, rnd(pix, smp, depth + 1, 2), rnd(pix, smp, depth + 1, 3));
        float cl = dot(ns, l);
        if (cl > 0.f && dot(ng, l) > 0.f) {
            float3 contrib = T * alb * P.sunE * (cl * CWP_INV_PI);
            int s = atomicAdd(sCnt, 1);
            if (FS) { float st = 1e30f; int stri = -1; float su, sv; if (!trav<V, true, TRI>(nodes, trav_, po, l, 0.f, st, stri, su, sv)) { atomicAdd(&accum[pix].x, contrib.x); atomicAdd(&accum[pix].y, contrib.y); atomicAdd(&accum[pix].z, contrib.z); } }
            else { sO[s] = mk4(po, 0.f); sD[s] = mk4(l, 1e30f); sC[s] = mk4(contrib, __int_as_float(pix)); }
        }
    }
    // continuation: cosine-weighted around the shading normal (throughput *= albedo)
    float3 nd = sampleCosineHemisphere(ns, rnd(pix, smp, depth + 1, 0), rnd(pix, smp, depth + 1, 1));
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
__global__ void k_scan(int* hist) { // single block, 1024 threads, kSortBins bins -> exclusive offsets in place
    __shared__ int sh[1024]; int tid = threadIdx.x; const int per = kSortBins / 1024; int base = tid * per; int sum = 0;
    for (int j = 0; j < per; j++) sum += hist[base + j];
    sh[tid] = sum; __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) { int v = tid >= off ? sh[tid - off] : 0; __syncthreads(); sh[tid] += v; __syncthreads(); }
    int run = sh[tid] - sum;
    for (int j = 0; j < per; j++) { int c = hist[base + j]; hist[base + j] = run; run += c; }
}
__global__ void k_scatter(const int* __restrict__ q, const int* nPtr, int* fetch, const unsigned* __restrict__ keys, int* cursor, int* qOut) { WARP_LOOP(nPtr, fetch, k) { int i = q[k]; int pos = atomicAdd(&cursor[keys[i]], 1); qOut[pos] = i; } }
__global__ void k_sumcounts(const int* cnt, int depths, unsigned long long* tot) { if (threadIdx.x == 0) { unsigned long long a = 0, b = 0; for (int d = 0; d < depths; d++) { a += cnt[d * 8]; b += cnt[d * 8 + 1]; } tot[0] += a; tot[1] += b; } }

// ---------------- full material integrator ----------------
// OpenPBR-subset materials (render/bsdf.h), emissive triangles as area lights, environment map, sun; next-event
// estimation with one light sample per vertex (light type chosen with probabilities pSun/pTri/pEnv, triangle by
// power, environment texel by luminance * sin(theta)) combined with BSDF sampling by the power heuristic
// (Veach 1997). Optional Russian roulette. Path state adds pAux = pdf of the last BSDF sample (<= 0: camera ray or
// delta lobe, whose emission hits always count fully).
__constant__ FullParams PF;

__device__ __forceinline__ void envUV(float3 d, float& u, float& v) {
    u = 0.5f - (atan2f(d.y, d.x) - PF.envRot) * (1.f / CWP_TWO_PI); u -= floorf(u);
    v = acosf(fminf(fmaxf(d.z, -1.f), 1.f)) * CWP_INV_PI;
}
__device__ __forceinline__ float3 envDir(float u, float v) {
    float ph = (0.5f - u) * CWP_TWO_PI + PF.envRot, th = v * CWP_PI; float st = sinf(th);
    return mk3(st * cosf(ph), st * sinf(ph), cosf(th));
}
__device__ __forceinline__ void envTexel(float3 d, int& x, int& y, float& v) {
    float u; envUV(d, u, v); x = min((int)(u * PF.envW), PF.envW - 1); y = min((int)(v * PF.envH), PF.envH - 1);
}
__device__ __forceinline__ float3 envRadiance(float3 d) { int x, y; float v; envTexel(d, x, y, v); return xyz(PF.envTex[y * PF.envW + x]); }
__device__ __forceinline__ float envPdfTexel(int x, int y, float v) {
    const float* cr = PF.envCond + (size_t)y * (PF.envW + 1);
    float puv = (PF.envMarg[y + 1] - PF.envMarg[y]) * (cr[x + 1] - cr[x]) * PF.envW * PF.envH;
    float st = sinf(v * CWP_PI); return st > 0.f ? puv / (2.f * CWP_PI * CWP_PI * st) : 0.f;
}
__device__ __forceinline__ float envPdf(float3 d) {
    if (PF.envUniform) return 0.25f * CWP_INV_PI;
    int x, y; float v; envTexel(d, x, y, v); return envPdfTexel(x, y, v);
}
// textures: corner-uv interpolation (b1, b2 = barycentric weights of corners 1 and 2) and an RGBA lookup
__device__ __forceinline__ float2 triUV(int tri, float b1, float b2) {
    float4 a = PF.uvs[2 * tri], c = PF.uvs[2 * tri + 1]; float b0 = 1.f - b1 - b2;
    return make_float2(a.x * b0 + a.z * b1 + c.x * b2, a.y * b0 + a.w * b1 + c.y * b2);
}
__device__ __forceinline__ float4 texRGBA(int id, float2 uv) { return tex2D<float4>(PF.texObj[id], uv.x, 1.f - uv.y); }
__device__ __forceinline__ float texComp(float4 c, int ch) { return ch == 0 ? c.x : ch == 1 ? c.y : ch == 2 ? c.z : c.w; }
// largest i in [0, n) with cdf[i] <= u (cdf has n + 1 entries, cdf[0] = 0, cdf[n] = 1)
__device__ __forceinline__ int cdfFind(const float* cdf, int n, float u) { int lo = 0, hi = n; while (hi - lo > 1) { int mid = (lo + hi) >> 1; if (cdf[mid] <= u) lo = mid; else hi = mid; } return lo; }

// ---------------- full integrator: traversal of single-level or instanced (TLAS/BLAS) scenes ----------------
#include "accel/traverse_inst.cuh"
#include "accel/hploc.h"
#define FULLMASK 0xffffffffu
static constexpr int kPtFetch = 8;   // persistent traversal: a warp fetches new rays once >= kPtFetch of its lanes are idle

template <int V, bool ANY, int TRI, bool INST>
__device__ __forceinline__ bool sceneTrav(const float4* __restrict__ nodes, const float4* __restrict__ tris, float3 o, float3 d, float tmin, float& t, int& tri, float& u, float& v, int& inst) {
    if (INST) return travInst<V, ANY, TRI>(PF.tlas, PF.tlasRoot, PF.inst, nodes, tris, o, d, tmin, t, tri, u, v, inst);
    return trav<V, ANY, TRI>(nodes, tris, o, d, tmin, t, tri, u, v);
}

// Shading of one closest hit (or miss) of path i: emission / environment with MIS, one NEE sample (queued as a shadow
// ray), BSDF-sampled continuation (appended to qOut). INST: hit in instance hinst; geometry is in object space and is
// transformed here (normals by inv^T, tangents by the forward matrix, handedness and orientation by sign(det)).
template <bool INST>
__device__ __forceinline__ void shadeFull(int i, float4 O, float4 D, float hitT, int tri, float hu, float hv, int hinst,
        const float4* __restrict__ tris, const float4* __restrict__ nrm, float4* pO, float4* pD, float4* pT, int* pDepth, float* pAux, float4* accum,
        int* qOut, int* qOutCnt, float4* sO, float4* sD, float4* sC, int* sCnt, int smpBase, int N) {
    float4 T4 = pT[i]; float3 T = xyz(T4); int pix = __float_as_int(T4.w);
    float lastPdf = pAux[i]; int depth = pDepth[i]; int smp = smpBase + i / N;
    float3 d = xyz(D);
    if (tri < 0) {   // miss: environment (MIS-weighted) or constant sky (BSDF-sampled only)
        float3 L;
        if (PF.hasEnv) {
            L = envRadiance(d);
            if (lastPdf > 0.f) { if (PF.lightMode == kLightNEE) L = mk3(0.f, 0.f, 0.f); else if (PF.lightMode == kLightMIS) L = L * misPower(lastPdf, PF.pEnv * envPdf(d)); }
        } else L = P.sky;
        float3 c = T * L; if (c.x > 0.f || c.y > 0.f || c.z > 0.f) { atomicAdd(&accum[pix].x, c.x); atomicAdd(&accum[pix].y, c.y); atomicAdd(&accum[pix].z, c.z); }
        return;
    }
    float3 p = xyz(O) + d * hitT;
    float4 b4 = tris[3 * tri + 1], c4 = tris[3 * tri + 2];
    float4 xr[3], xf[3]; float dsg = 1.f;   // INST: inverse and forward rows of the hit instance
    if (INST) { const DevInstance* XI = PF.inst + hinst; xr[0] = XI->inv[0]; xr[1] = XI->inv[1]; xr[2] = XI->inv[2]; xf[0] = XI->fwd[0]; xf[1] = XI->fwd[1]; xf[2] = XI->fwd[2]; dsg = XI->detSign; }
    float3 ng0 = INST ? normalize(xfRowsN(xr, cross(xyz(b4), xyz(c4)))) * dsg : normalize(cross(xyz(b4), xyz(c4)));
    float3 ns = ng0;
    if (P.hasN) {
        float3 n0 = xyz(nrm[3 * tri]), n1 = xyz(nrm[3 * tri + 1]), n2 = xyz(nrm[3 * tri + 2]); float3 nn = n0 * (1.f - hu - hv) + n1 * hu + n2 * hv;
        if (INST) nn = xfRowsN(xr, nn);
        float l = length(nn); if (l > 0.f) ns = nn * (1.f / l); if (dot(ns, ng0) < 0.f) ns = -ns;
    }
    const int mid = __float_as_int(b4.w); float3 nsU = ns; bool mapped = false;   // nsU: unperturbed normal (bump-terminator term)
    Material M = PF.mats[mid];
    if (PF.mtex && mid < PF.nMtex) {   // textures multiply the material factors (glTF metallic-roughness semantics)
        const MaterialTextures X = PF.mtex[mid];
        if (hasTextures(X)) {
            float2 uv = triUV(tri, hu, hv);
            if (X.baseColor >= 0) M.baseColor = M.baseColor * xyz(texRGBA(X.baseColor, uv));
            if (X.emission >= 0) M.emission = M.emission * xyz(texRGBA(X.emission, uv));
            if (X.metallic >= 0) M.metallic *= texComp(texRGBA(X.metallic, uv), X.metallicCh);
            if (X.roughness >= 0) M.roughness *= texComp(texRGBA(X.roughness, uv), X.roughnessCh);
            if (X.normal >= 0 && PF.tans) {   // tangent-space normal map: n = T x + B y + N z, B = w cross(N, T)
                float w0 = 1.f - hu - hv; float4 t0 = PF.tans[3 * tri], t1 = PF.tans[3 * tri + 1], t2 = PF.tans[3 * tri + 2];
                float3 t = xyz(t0) * w0 + xyz(t1) * hu + xyz(t2) * hv; float sg = (t0.w * w0 + t1.w * hu + t2.w * hv) < 0.f ? -1.f : 1.f;
                if (INST) { t = xfRowsV(xf, t); sg *= dsg; }
                t = t - ns * dot(ns, t); float tl = length(t);
                if (tl > 0.f) {
                    t = t * (1.f / tl); float3 bt = cross(ns, t) * sg; float4 c = texRGBA(X.normal, uv);
                    float3 nm = t * ((2.f * c.x - 1.f) * X.normalScale) + bt * ((2.f * c.y - 1.f) * X.normalScale) + ns * (2.f * c.z - 1.f);
                    float nl = length(nm); if (nl > 0.f) { ns = nm * (1.f / nl); mapped = true; }
                }
            }
        }
    }
    bool front = dot(ng0, d) < 0.f;
    float3 ng = front ? ng0 : -ng0; if (!front) { ns = -ns; nsU = -nsU; }
    // emission (two-sided), MIS-weighted against light sampling of this triangle
    if (M.emission.x > 0.f || M.emission.y > 0.f || M.emission.z > 0.f) {
        float w = 1.f;
        if (lastPdf > 0.f) {
            if (PF.lightMode == kLightNEE) w = 0.f;
            else if (PF.lightMode == kLightMIS) { float cl = fabsf(dot(ng0, d)); float pa = INST ? PF.matLightPdf[mid] : PF.triLightPdf[tri]; float pl = cl > 0.f ? PF.pTri * pa * hitT * hitT / cl : 0.f; w = misPower(lastPdf, pl); }
        }
        float3 c = T * M.emission * w; if (w > 0.f) { atomicAdd(&accum[pix].x, c.x); atomicAdd(&accum[pix].y, c.y); atomicAdd(&accum[pix].z, c.z); }
    }
    if (depth > P.maxB) return;
    if (PF.rr && depth >= PF.rrDepth) { float qs = fminf(fmaxf(maxc(T), 0.05f), 1.f); if (rnd(pix, smp, depth + 1, 6) >= qs) return; T = T * (1.f / qs); }
    float3 wo = -d;
    if (dot(ns, wo) <= 1e-6f) ns = ng;
    float3 tx, ty; onb(ns, tx, ty);
    float3 woL = mk3(dot(wo, tx), dot(wo, ty), dot(wo, ns));
    float3 nrL = mk3(0.f, 0.f, 1.f);
    if (PF.bumpShadow && mapped) { nrL = mk3(dot(nsU, tx), dot(nsU, ty), dot(nsU, ns)); if (!(nrL.z > 0.f)) nrL = mk3(0.f, 0.f, 1.f); }
    float eta = front ? M.ior : 1.f / M.ior;
    float3 poUp = p + ng * P.eps, poDn = p - ng * P.eps;
    float ux1 = rnd(pix, smp, depth + 1, 7), ux2 = rng4(pix, smp, depth + 1, 7, P.seed ^ 0x68E31DA4u);   // glossy-diffuse albedo estimate
    // ---- next-event estimation (one light sample) ----
    if (bsdfHasNonDelta(M)) {
        float us = rnd(pix, smp, depth + 1, 5), u2 = rnd(pix, smp, depth + 1, 2), u3 = rnd(pix, smp, depth + 1, 3);
        float3 l = mk3(0.f, 0.f, 1.f), Le = mk3(0.f, 0.f, 0.f); float pl = 0.f, tmax = 1e30f; bool isSun = false, ok = false; int orgSide = 0;   // +1/-1: origin fixed by the triangle sample
        if (us < PF.pSun) { l = sampleUniformCone(P.sunTo, P.sunCos, u2, u3); Le = P.sunE; pl = PF.pSun; isSun = true; ok = true; }
        else if (us < PF.pSun + PF.pTri) {
            float ut = fminf((us - PF.pSun) / PF.pTri, 0.99999994f);
            int e = cdfFind(PF.emCdf, PF.nEm, ut);
            float4 a0, a1, a2; int kt;
            if (INST) { a0 = PF.emTris[3 * e]; a1 = PF.emTris[3 * e + 1]; a2 = PF.emTris[3 * e + 2]; kt = __float_as_int(a0.w); }
            else { kt = PF.emTri[e]; a0 = tris[3 * kt]; a1 = tris[3 * kt + 1]; a2 = tris[3 * kt + 2]; }
            float su = sqrtf(u2); float3 pL = xyz(a0) + xyz(a1) * (su * (1.f - u3)) + xyz(a2) * (su * u3);
            float3 nL = normalize(cross(xyz(a1), xyz(a2)));
            // connect from the shadow ray's actual (offset) origin so that tmax stops just short of the light point
            orgSide = dot(ng, pL - p) > 0.f ? 1 : -1; float3 org = orgSide > 0 ? poUp : poDn;
            float3 dv = pL - org; float dist2 = dot(dv, dv), dist = sqrtf(dist2);
            if (dist > 0.f) {
                l = dv * (1.f / dist); float cl = fabsf(dot(nL, l));
                if (cl > 0.f) {
                    int lm = __float_as_int(a1.w); float pa = INST ? PF.matLightPdf[lm] : PF.triLightPdf[kt];
                    pl = PF.pTri * pa * dist2 / cl; Le = PF.mats[lm].emission; tmax = dist * (1.f - 1e-3f); ok = true;
                    if (PF.mtex && lm < PF.nMtex && PF.mtex[lm].emission >= 0) Le = Le * xyz(texRGBA(PF.mtex[lm].emission, triUV(kt, su * (1.f - u3), su * u3)));
                }
            }
        } else if (PF.pEnv > 0.f && PF.envUniform) {
            l = sampleUniformSphere(u2, u3); pl = PF.pEnv * (0.25f * CWP_INV_PI); Le = envRadiance(l); ok = true;
        } else if (PF.pEnv > 0.f) {
            float ue = fminf((us - PF.pSun - PF.pTri) / PF.pEnv, 0.99999994f);
            int y = cdfFind(PF.envMarg, PF.envH, ue);
            const float* cr = PF.envCond + (size_t)y * (PF.envW + 1);
            int x = cdfFind(cr, PF.envW, u2);
            float v = (y + u3) / PF.envH, u = (x + 0.5f) / PF.envW;
            // continuous position inside the texel: reuse the leftover of u2 within the chosen column
            float w0 = cr[x + 1] - cr[x]; if (w0 > 0.f) u = (x + fminf((u2 - cr[x]) / w0, 0.99999994f)) / PF.envW;
            l = envDir(u, v); float pe = envPdfTexel(x, y, v);
            if (pe > 0.f) { pl = PF.pEnv * pe; Le = xyz(PF.envTex[y * PF.envW + x]); ok = true; }
        }
        if (ok) {
            float3 lL = mk3(dot(l, tx), dot(l, ty), dot(l, ns));
            float gl = dot(ng, l);
            if (lL.z != 0.f && (gl > 0.f) == (lL.z > 0.f)) {
                BsdfEval e = bsdfEval(M, woL, lL, eta, PF.gdEnergy, ux1, ux2, nrL);
                float w = (isSun || PF.lightMode == kLightNEE) ? 1.f : misPower(pl, e.pdf);
                float3 contrib = T * e.f * Le * (fabsf(lL.z) * w / pl);
                if (contrib.x > 0.f || contrib.y > 0.f || contrib.z > 0.f) {
                    int s = atomicAdd(sCnt, 1);
                    sO[s] = mk4((orgSide ? orgSide > 0 : gl > 0.f) ? poUp : poDn, 0.f); sD[s] = mk4(l, tmax); sC[s] = mk4(contrib, __int_as_float(pix));
                }
            }
        }
    }
    // ---- BSDF sampling: continuation ray ----
    float3 wiL, wgt; float pdf;
    if (!bsdfSample(M, woL, eta, PF.gdEnergy, ux1, ux2, rnd(pix, smp, depth + 1, 4), rnd(pix, smp, depth + 1, 0), rnd(pix, smp, depth + 1, 1), wiL, wgt, pdf, nrL)) return;
    float3 wi = normalize(tx * wiL.x + ty * wiL.y + ns * wiL.z);
    float gi = dot(ng, wi);
    if (gi == 0.f || (gi > 0.f) != (wiL.z > 0.f)) return;
    T = T * wgt;
    if (!(maxc(T) > 0.f)) return;
    pO[i] = mk4(gi > 0.f ? poUp : poDn, 0.f); pD[i] = mk4(wi, 1e30f); pT[i] = mk4(T, __int_as_float(pix)); pDepth[i] = depth + 1; pAux[i] = pdf;
    int s = atomicAdd(qOutCnt, 1); qOut[s] = i;
}

#define SHADE_ARGS tris, nrm, pO, pD, pT, pDepth, pAux, accum, qOut, qOutCnt, sO, sD, sC, sCnt, smpBase, N
#define SHADE_PARAMS const float4* __restrict__ tris, const float4* __restrict__ nrm, float4* pO, float4* pD, float4* pT, int* pDepth, float* pAux, float4* accum, \
        int* qOut, int* qOutCnt, float4* sO, float4* sD, float4* sC, int* sCnt, int smpBase, int N
// Every full-integrator kernel takes its queue length either from the host (nPtr == nullptr: nArg, grid sized to it)
// or from device memory (nPtr: CUDA graphs / device-driven passes, fixed grid with a grid-stride loop).
#define GRID_STRIDE(k, n) for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < (n); k += gridDim.x * blockDim.x)

// fused closest-hit traversal + shading (default)
template <int V, int TRI, bool INST>
__global__ void __launch_bounds__(128) k_trace_shade_full(const float4* __restrict__ nodes, const float4* __restrict__ trav_, const int* __restrict__ q, const int* nPtr, int nArg, SHADE_PARAMS) {
    const int n = nPtr ? *nPtr : nArg;
    GRID_STRIDE(k, n) {
        int i = q[k];
        float4 O = pO[i], D = pD[i]; float hitT = D.w; int tri = -1, hinst = -1; float hu = 0, hv = 0;
        sceneTrav<V, false, TRI, INST>(nodes, trav_, xyz(O), xyz(D), O.w, hitT, tri, hu, hv, hinst);
        shadeFull<INST>(i, O, D, hitT, tri, hu, hv, hinst, SHADE_ARGS);
    }
}
// split mode: closest-hit traversal into the hit buffer, then shading
template <int V, int TRI, bool INST>
__global__ void __launch_bounds__(128) k_trace_full(const float4* __restrict__ nodes, const float4* __restrict__ trav_, const int* __restrict__ q, const int* nPtr, int nArg, const float4* __restrict__ pO, const float4* __restrict__ pD, float4* hits, int* hitI) {
    const int n = nPtr ? *nPtr : nArg;
    GRID_STRIDE(k, n) {
        int i = q[k];
        float4 O = pO[i], D = pD[i]; float hitT = D.w; int tri = -1, hinst = -1; float hu = 0, hv = 0;
        sceneTrav<V, false, TRI, INST>(nodes, trav_, xyz(O), xyz(D), O.w, hitT, tri, hu, hv, hinst);
        hits[i] = make_float4(hitT, hu, hv, __int_as_float(tri)); if (INST) hitI[i] = hinst;
    }
}
template <bool INST>
__global__ void __launch_bounds__(128) k_shade_full(const int* __restrict__ q, const int* nPtr, int nArg, const float4* __restrict__ hits, const int* __restrict__ hitI, SHADE_PARAMS) {
    const int n = nPtr ? *nPtr : nArg;
    GRID_STRIDE(k, n) { int i = q[k]; float4 h = hits[i]; shadeFull<INST>(i, pO[i], pD[i], h.x, __float_as_int(h.w), h.y, h.z, INST ? hitI[i] : -1, SHADE_ARGS); }
}
template <int V, int TRI, bool INST>
__global__ void __launch_bounds__(128) k_shadow_full(const float4* __restrict__ nodes, const float4* __restrict__ trav_, const int* nPtr, int nArg, const float4* __restrict__ sO, const float4* __restrict__ sD, const float4* __restrict__ sC, float4* accum) {
    const int n = nPtr ? *nPtr : nArg;
    GRID_STRIDE(k, n) {
        float4 O = sO[k], D = sD[k]; float hitT = D.w; int tri = -1, hinst = -1; float u, v;
        if (!sceneTrav<V, true, TRI, INST>(nodes, trav_, xyz(O), xyz(D), O.w, hitT, tri, u, v, hinst)) { float4 c = sC[k]; int pix = __float_as_int(c.w); atomicAdd(&accum[pix].x, c.x); atomicAdd(&accum[pix].y, c.y); atomicAdd(&accum[pix].z, c.z); }
    }
}
// Persistent threads with dynamic ray fetching (Aila & Laine 2009): a grid that just fills the device; each warp keeps
// traversing until >= kPtFetch of its lanes are idle, then those lanes take the next rays from the queue with one
// atomicAdd per warp. ANY = shadow rays (queue index = ray index, contribution added when unoccluded).
template <int V, int TRI, bool INST, bool ANY>
__device__ __forceinline__ void persistentTrace(const float4* __restrict__ nodes, const float4* __restrict__ trav_, const int* __restrict__ q, int n, int* fetch,
                                                const float4* __restrict__ rO, const float4* __restrict__ rD, float4* hits, int* hitI, const float4* __restrict__ sC, float4* accum) {
    StepTrav<V, ANY, TRI, INST> st;
    const int lane = threadIdx.x & 31; const unsigned lt = (1u << lane) - 1u;
    int k = -1; bool live = true;
    while (true) {
        bool need = live && k < 0;
        unsigned nm = __ballot_sync(FULLMASK, need);
        if (nm) {
            int leader = __ffs(nm) - 1, base = 0;
            if (lane == leader) base = atomicAdd(fetch, __popc(nm));
            base = __shfl_sync(FULLMASK, base, leader);
            if (need) {
                int kk = base + __popc(nm & lt);
                if (kk < n) { k = kk; int i = ANY ? k : q[k]; float4 O = rO[i], D = rD[i]; st.init(xyz(O), xyz(D), O.w, D.w, INST ? PF.tlasRoot : 0); }
                else live = false;
            }
        }
        if (!__any_sync(FULLMASK, k >= 0)) break;
        while (true) {
            bool act = k >= 0;
            if (act && st.step(PF.tlas, PF.inst, nodes, trav_)) {
                if (ANY) { if (st.hitTri < 0) { float4 c = sC[k]; int pix = __float_as_int(c.w); atomicAdd(&accum[pix].x, c.x); atomicAdd(&accum[pix].y, c.y); atomicAdd(&accum[pix].z, c.z); } }
                else { int i = q[k]; hits[i] = make_float4(st.hitT, st.hu, st.hv, __int_as_float(st.hitTri)); if (INST) hitI[i] = st.hitInst; }
                k = -1; act = false;
            }
            unsigned idle = __ballot_sync(FULLMASK, !act && live);
            if (__popc(idle) >= kPtFetch || !__any_sync(FULLMASK, act)) break;
        }
    }
}
template <int V, int TRI, bool INST>
__global__ void __launch_bounds__(128) k_trace_full_pt(const float4* __restrict__ nodes, const float4* __restrict__ trav_, const int* __restrict__ q, const int* nPtr, int nArg, int* fetch, const float4* __restrict__ pO, const float4* __restrict__ pD, float4* hits, int* hitI) {
    persistentTrace<V, TRI, INST, false>(nodes, trav_, q, nPtr ? *nPtr : nArg, fetch, pO, pD, hits, hitI, nullptr, nullptr);
}
template <int V, int TRI, bool INST>
__global__ void __launch_bounds__(128) k_shadow_full_pt(const float4* __restrict__ nodes, const float4* __restrict__ trav_, const int* nPtr, int nArg, int* fetch, const float4* __restrict__ sO, const float4* __restrict__ sD, const float4* __restrict__ sC, float4* accum) {
    persistentTrace<V, TRI, INST, true>(nodes, trav_, nullptr, nPtr ? *nPtr : nArg, fetch, sO, sD, nullptr, nullptr, sC, accum);
}
__global__ void k_setcount(int* c, int v) { if (threadIdx.x == 0) *c = v; }
// TLAS input: world boxes of the instances (BLAS root box corners through the forward matrix)
__global__ void k_inst_boxes(const DevInstance* __restrict__ inst, const float4* __restrict__ meshBox, int n, float4* lo, float4* hi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const DevInstance& I = inst[i]; float4 bl = meshBox[2 * I.mesh], bh = meshBox[2 * I.mesh + 1];
    float3 mn = mk3(3.4e38f, 3.4e38f, 3.4e38f), mx = mk3(-3.4e38f, -3.4e38f, -3.4e38f);
    for (int c = 0; c < 8; c++) { float3 w = xfRowsP(I.fwd, mk3(c & 1 ? bh.x : bl.x, c & 2 ? bh.y : bl.y, c & 4 ? bh.z : bl.z)); mn = vmin(mn, w); mx = vmax(mx, w); }
    lo[i] = mk4(mn, 0.f); hi[i] = mk4(mx, 0.f);
}

// ---------------- host driver ----------------
static const int TB = 128;
// dispatch helpers (V, TRI runtime -> template)
#define DISPATCH(F, G, T, ARGS) do { \
        if (V_ == 2) { if (TRI_) F<2, 1><<<G, T>>>ARGS; else F<2, 0><<<G, T>>>ARGS; } \
        else if (V_ == 4) { if (TRI_) F<4, 1><<<G, T>>>ARGS; else F<4, 0><<<G, T>>>ARGS; } \
        else { if (TRI_) F<8, 1><<<G, T>>>ARGS; else F<8, 0><<<G, T>>>ARGS; } } while (0)
#define DISPATCH3(F, FS, G, T, ARGS) do { \
        if (V_ == 2) { if (TRI_) F<2, 1, FS><<<G, T>>>ARGS; else F<2, 0, FS><<<G, T>>>ARGS; } \
        else if (V_ == 4) { if (TRI_) F<4, 1, FS><<<G, T>>>ARGS; else F<4, 0, FS><<<G, T>>>ARGS; } \
        else { if (TRI_) F<8, 1, FS><<<G, T>>>ARGS; else F<8, 0, FS><<<G, T>>>ARGS; } } while (0)
// full integrator: V in {2, 8} x TRI x INST on stream st_ (V = 4 only for the fused kernel without instances)
#define DISPF_V(F, VV, G, ARGS) do { if (INST_) { if (TRI_) F<VV, 1, true><<<G, TB, 0, st_>>>ARGS; else F<VV, 0, true><<<G, TB, 0, st_>>>ARGS; } \
                                     else { if (TRI_) F<VV, 1, false><<<G, TB, 0, st_>>>ARGS; else F<VV, 0, false><<<G, TB, 0, st_>>>ARGS; } } while (0)
#define DISPF(F, G, ARGS) do { if (V_ == 2) DISPF_V(F, 2, G, ARGS); else DISPF_V(F, 8, G, ARGS); } while (0)

void WavefrontRenderer::init(const Params& H, const SceneBuffers& S, const RenderOptions& O) {
    O_ = O; if (O_.sort) O_.devloop = 1;
    V_ = O.variant; TRI_ = O.tri; INST_ = O.inst ? 1 : 0;
    int dev = 0; CK(cudaSetDevice(dev)); cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev)); gpuName = prop.name; nSM_ = prop.multiProcessorCount;
    size_t memFree0, memTot; CK(cudaMemGetInfo(&memFree0, &memTot));
    CK(cudaMalloc(&dNodes, S.nodes.size() * 16)); CK(cudaMemcpy(dNodes, S.nodes.data(), S.nodes.size() * 16, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dTris, S.tris.size() * 16)); CK(cudaMemcpy(dTris, S.tris.data(), S.tris.size() * 16, cudaMemcpyHostToDevice));
    if (TRI_ == 1) { CK(cudaMalloc(&dWoop, S.woop.size() * 16)); CK(cudaMemcpy(dWoop, S.woop.data(), S.woop.size() * 16, cudaMemcpyHostToDevice)); }
    dTrav = TRI_ == 1 ? dWoop : dTris;
    if (!S.nrm.empty()) { CK(cudaMalloc(&dNrm, S.nrm.size() * 16)); CK(cudaMemcpy(dNrm, S.nrm.data(), S.nrm.size() * 16, cudaMemcpyHostToDevice)); }
    CK(cudaMalloc(&dAlb, S.albedo.size() * sizeof(float3))); CK(cudaMemcpy(dAlb, S.albedo.data(), S.albedo.size() * sizeof(float3), cudaMemcpyHostToDevice));
    CK(cudaMemcpyToSymbol(P, &H, sizeof H)); H_ = H;
    N_ = H.cam.w * H.cam.h; K_ = O.batch < 1 ? 1 : O.batch; NK_ = (size_t)N_ * K_;
    DEPTHS_ = O.bounces + 2;
    const size_t NK = NK_;
    CK(cudaMalloc(&pO, NK * 16)); CK(cudaMalloc(&pD, NK * 16)); CK(cudaMalloc(&pT, NK * 16)); CK(cudaMalloc(&hits, NK * 16));
    CK(cudaMalloc(&sO, NK * 16)); CK(cudaMalloc(&sD, NK * 16)); CK(cudaMalloc(&sC, NK * 16)); CK(cudaMalloc(&accum, N_ * 16));
    CK(cudaMalloc(&pDepth, NK * 4)); CK(cudaMalloc(&q0, NK * 4)); CK(cudaMalloc(&q1, NK * 4)); CK(cudaMalloc(&q2, NK * 4));
    CK(cudaMalloc(&cnt, (DEPTHS_ + 1) * 8 * 4)); CK(cudaMalloc(&hist, kSortBins * 4)); CK(cudaMalloc(&dtot, 16));
    if (O_.sort) CK(cudaMalloc(&keys, NK * 4));
    size_t memFree1; CK(cudaMemGetInfo(&memFree1, &memTot));
    vramMB = (double)(memFree0 - memFree1) / 1048576.0;
    CK(cudaMallocHost(&hcnt, 64)); CK(cudaMallocHost(&hNK, 16)); hNK[0] = (int)NK;
    {
        int bpm = 0;
#define OCC(VV, TT) CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpm, k_trace_dev<VV, TT>, 128, 0))
        if (V_ == 2) { if (TRI_) OCC(2, 1); else OCC(2, 0); } else if (V_ == 4) { if (TRI_) OCC(4, 1); else OCC(4, 0); } else { if (TRI_) OCC(8, 1); else OCC(8, 0); }
#undef OCC
        ptBlocks = bpm * prop.multiProcessorCount;
        if (!O_.full) fprintf(stderr, "trace kernel: %d blocks/SM x %d SMs (occupancy %.0f%%)\n", bpm, prop.multiProcessorCount, 100.0 * bpm * 128 / prop.maxThreadsPerMultiProcessor);
    }
    if (O_.full) initFull(S);
    CK(cudaEventCreate(&f0)); CK(cudaEventCreate(&f1));
}

template <typename T> static T* upload(const std::vector<T>& v) { if (v.empty()) return nullptr; T* d; CK(cudaMalloc(&d, v.size() * sizeof(T))); CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }

template <int V, int TRI, bool INST> static int occFull(bool shadow) {
    int b = 0;
    if (shadow) CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, k_shadow_full_pt<V, TRI, INST>, TB, 0));
    else CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, k_trace_full_pt<V, TRI, INST>, TB, 0));
    return b;
}

void WavefrontRenderer::initFull(const SceneBuffers& S) {
    size_t f0m, f1m, tm; CK(cudaMemGetInfo(&f0m, &tm));
    CK(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking));
    FullParams F; memset(&F, 0, sizeof F);
    F.mats = upload(S.mats); F.triLightPdf = upload(S.triLightPdf); F.emTri = upload(S.emTri); F.emCdf = upload(S.emCdf); F.nEm = (int)S.emTri.size();
    if (INST_) {   // instanced scene: TLAS buffers (rebuilt by setInstances), per-frame emissive triangles
        nInst_ = S.nInst;
        CK(cudaMalloc(&dInst, (size_t)nInst_ * sizeof(DevInstance))); CK(cudaMalloc(&dTlas, (size_t)(nInst_ > 1 ? nInst_ - 1 : 1) * 64));
        CK(cudaMalloc(&dBoxLo, (size_t)nInst_ * 16)); CK(cudaMalloc(&dBoxHi, (size_t)nInst_ * 16)); dMeshBox = upload(S.meshBox);
        hploc_.reserve(nInst_);
        F.inst = dInst; F.tlas = dTlas; F.tlasRoot = hplocBVH2Root(nInst_);
        nEmInst_ = S.nEmInst; F.nEm = nEmInst_;
        if (nEmInst_ > 0) { CK(cudaMalloc(&dEmTris, (size_t)nEmInst_ * 48)); CK(cudaMalloc(&dEmCdf, (size_t)(nEmInst_ + 1) * 4)); F.emTris = dEmTris; F.emCdf = dEmCdf; }
        CK(cudaMalloc(&dMatLightPdf, S.mats.size() * 4)); CK(cudaMemset(dMatLightPdf, 0, S.mats.size() * 4)); F.matLightPdf = dMatLightPdf;
        CK(cudaEventCreate(&tl0)); CK(cudaEventCreate(&tl1));
    }
    F.hasEnv = S.env.w > 0 ? 1 : 0;
    if (F.hasEnv) { F.envTex = upload(S.env.texels); F.envMarg = upload(S.env.marg); F.envCond = upload(S.env.cond); F.envW = S.env.w; F.envH = S.env.h; F.envRot = S.envRotDeg * 3.14159265f / 180.f; }
    bool sun = S.sunOn, triL = F.nEm > 0 && O_.lightMode != kLightBSDF, envL = F.hasEnv && O_.lightMode != kLightBSDF;
    int nt = (sun ? 1 : 0) + (triL ? 1 : 0) + (envL ? 1 : 0);
    F.pSun = sun ? 1.f / nt : 0.f; F.pTri = triL ? 1.f / nt : 0.f; F.pEnv = envL ? 1.f / nt : 0.f;
    F.lightMode = O_.lightMode; F.rr = O_.rr; F.rrDepth = O_.rrDepth; F.gdEnergy = O_.gdEnergy; F.envUniform = O_.envUniform; F.bumpShadow = O_.bumpShadow;
    if (!S.mtex.empty()) {   // textures: one uchar4 CUDA array + texture object per image (bilinear, wrap, sRGB decode for colour)
        std::vector<cudaTextureObject_t> objs;
        for (const TextureImage& T : S.tex) {
            cudaChannelFormatDesc cd = cudaCreateChannelDesc<uchar4>(); cudaArray_t arr;
            CK(cudaMallocArray(&arr, &cd, T.w, T.h));
            CK(cudaMemcpy2DToArray(arr, 0, 0, T.rgba.data(), (size_t)T.w * 4, (size_t)T.w * 4, T.h, cudaMemcpyHostToDevice));
            cudaResourceDesc rd; memset(&rd, 0, sizeof rd); rd.resType = cudaResourceTypeArray; rd.res.array.array = arr;
            cudaTextureDesc td; memset(&td, 0, sizeof td);
            td.addressMode[0] = td.addressMode[1] = cudaAddressModeWrap; td.filterMode = O_.texNearest ? cudaFilterModePoint : cudaFilterModeLinear;
            td.readMode = cudaReadModeNormalizedFloat; td.normalizedCoords = 1; td.sRGB = T.srgb ? 1 : 0;
            cudaTextureObject_t o; CK(cudaCreateTextureObject(&o, &rd, &td, nullptr)); objs.push_back(o);
        }
        F.texObj = upload(objs); F.mtex = upload(S.mtex); F.nMtex = (int)S.mtex.size(); F.uvs = upload(S.uvs); F.tans = upload(S.tans);
    }
    CK(cudaMemcpyToSymbol(PF, &F, sizeof F)); F_ = F; nMats_ = (int)S.mats.size(); sunOn_ = S.sunOn;
    CK(cudaMalloc(&pAux, NK_ * 4));
    if (INST_) CK(cudaMalloc(&hitI, NK_ * 4));
    if (O_.persistent && V_ != 4) {
        int bt = 0, bs = 0;
        if (V_ == 2) { if (INST_) { bt = TRI_ ? occFull<2, 1, true>(false) : occFull<2, 0, true>(false); bs = TRI_ ? occFull<2, 1, true>(true) : occFull<2, 0, true>(true); }
                       else { bt = TRI_ ? occFull<2, 1, false>(false) : occFull<2, 0, false>(false); bs = TRI_ ? occFull<2, 1, false>(true) : occFull<2, 0, false>(true); } }
        else { if (INST_) { bt = TRI_ ? occFull<8, 1, true>(false) : occFull<8, 0, true>(false); bs = TRI_ ? occFull<8, 1, true>(true) : occFull<8, 0, true>(true); }
               else { bt = TRI_ ? occFull<8, 1, false>(false) : occFull<8, 0, false>(false); bs = TRI_ ? occFull<8, 1, false>(true) : occFull<8, 0, false>(true); } }
        ptTraceBlocks_ = bt * nSM_; ptShadowBlocks_ = bs * nSM_; ptBlocks = ptTraceBlocks_;
        fprintf(stderr, "persistent traversal: trace %d blocks/SM, shadow %d blocks/SM x %d SMs\n", bt, bs, nSM_);
    }
    gridCap_ = 16 * nSM_;
    CK(cudaMemGetInfo(&f1m, &tm)); vramMB += (double)(f0m - f1m) / 1048576.0;
    fprintf(stderr, "full integrator: %zu materials, %d emissive triangles, env %dx%d (%s sampling), light selection sun %.2f tri %.2f env %.2f, mode %s, rr %d, glossy-diffuse energy compensation %d, %zu textures%s, kernels %s%s\n",
            S.mats.size(), F.nEm, F.envW, F.envH, O_.envUniform ? "uniform" : "importance", F.pSun, F.pTri, F.pEnv,
            O_.lightMode == kLightMIS ? "mis" : O_.lightMode == kLightNEE ? "nee" : "bsdf", O_.rr, O_.gdEnergy, S.mtex.empty() ? (size_t)0 : S.tex.size(),
            INST_ ? ", instanced (TLAS/BLAS)" : "", O_.persistent ? "split+persistent" : O_.fuse ? "fused" : "split", O_.graph ? " + CUDA graph" : "");
}

// per-frame instance update: transforms, emissive triangles, TLAS rebuilt on the GPU (instance boxes -> H-PLOC -> BVH2)
void WavefrontRenderer::setInstances(const std::vector<DevInstance>& inst, const std::vector<float4>& emTris, const std::vector<float>& emCdf, const std::vector<float>& matLightPdf) {
    CK(cudaMemcpyAsync(dInst, inst.data(), inst.size() * sizeof(DevInstance), cudaMemcpyHostToDevice, st_));
    if (nEmInst_ > 0) { CK(cudaMemcpyAsync(dEmTris, emTris.data(), emTris.size() * 16, cudaMemcpyHostToDevice, st_)); CK(cudaMemcpyAsync(dEmCdf, emCdf.data(), emCdf.size() * 4, cudaMemcpyHostToDevice, st_)); }
    if (!matLightPdf.empty()) CK(cudaMemcpyAsync(dMatLightPdf, matLightPdf.data(), matLightPdf.size() * 4, cudaMemcpyHostToDevice, st_));
    CK(cudaEventRecord(tl0, st_));
    k_inst_boxes<<<(nInst_ + 127) / 128, 128, 0, st_>>>(dInst, dMeshBox, nInst_, dBoxLo, dBoxHi);
    hploc_.build(dBoxLo, dBoxHi, nInst_, st_);
    hplocToBVH2(hploc_.nodes, nInst_, dTlas, st_);
    CK(cudaEventRecord(tl1, st_)); CK(cudaEventSynchronize(tl1));
    float ms; CK(cudaEventElapsedTime(&ms, tl0, tl1)); tlasMs = ms;
    CK(cudaGetLastError());
}

void WavefrontRenderer::launchTrace(int* q, int n) {
    if (n <= 0) return;
    const int N = N_;
    if (O_.fuse == 1) { int g = (n + TB - 1) / TB; DISPATCH3(k_trace_shade, false, g, TB, (dNodes, dTrav, q, n, dTris, dNrm, dAlb, pO, pD, pT, pDepth, accum, qOut_, cnt, sO, sD, sC, cnt + 1, smp_, N)); return; }
    if (O_.fuse == 2) { int g = (n + TB - 1) / TB; DISPATCH3(k_trace_shade, true, g, TB, (dNodes, dTrav, q, n, dTris, dNrm, dAlb, pO, pD, pT, pDepth, accum, qOut_, cnt, sO, sD, sC, cnt + 1, smp_, N)); return; }
    if (O_.persistent) { int g = ptBlocks < (n + 31) / 32 ? ptBlocks : (n + 31) / 32; CK(cudaMemsetAsync(cnt + 2, 0, 4)); DISPATCH(k_trace_dev, g, TB, (dNodes, dTrav, q, cnt + 8 * DEPTHS_, cnt + 2, pO, pD, hits)); }
    else { int g = (n + TB - 1) / TB; DISPATCH(k_trace, g, TB, (dNodes, dTrav, q, n, pO, pD, hits)); }
}

void WavefrontRenderer::launchShadow(int n) {
    if (n <= 0) return;
    if (O_.persistent) { int g = ptBlocks < (n + 31) / 32 ? ptBlocks : (n + 31) / 32; CK(cudaMemsetAsync(cnt + 3, 0, 4)); DISPATCH(k_shadow_dev, g, TB, (dNodes, dTrav, cnt + 8 * DEPTHS_ + 1, cnt + 3, sO, sD, sC, accum)); }
    else { int g = (n + TB - 1) / TB; DISPATCH(k_shadow, g, TB, (dNodes, dTrav, n, sO, sD, sC, accum)); }
}

// full integrator, one bounce: closest-hit traversal + shading of queue (q, n | *nPtr) -> continuation queue qOut
// (count *qOutCnt) and shadow queue (count *sCnt); fetch = zeroed counter for the persistent kernel
void WavefrontRenderer::launchFullTrace(int* q, const int* nPtr, int n, int* qOut, int* qOutCnt, int* sCnt, int* fetch, int smp) {
    const int N = N_;
    const int gq = nPtr ? gridCap_ : (n + TB - 1) / TB;
    if (O_.persistent) {
        int g = nPtr ? ptTraceBlocks_ : std::min(ptTraceBlocks_, (n + 31) / 32);
        DISPF(k_trace_full_pt, g, (dNodes, dTrav, q, nPtr, n, fetch, pO, pD, hits, hitI));
        if (INST_) k_shade_full<true><<<gq, TB, 0, st_>>>(q, nPtr, n, hits, hitI, dTris, dNrm, pO, pD, pT, pDepth, pAux, accum, qOut, qOutCnt, sO, sD, sC, sCnt, smp, N);
        else k_shade_full<false><<<gq, TB, 0, st_>>>(q, nPtr, n, hits, hitI, dTris, dNrm, pO, pD, pT, pDepth, pAux, accum, qOut, qOutCnt, sO, sD, sC, sCnt, smp, N);
    } else if (O_.fuse == 0) {
        DISPF(k_trace_full, gq, (dNodes, dTrav, q, nPtr, n, pO, pD, hits, hitI));
        if (INST_) k_shade_full<true><<<gq, TB, 0, st_>>>(q, nPtr, n, hits, hitI, dTris, dNrm, pO, pD, pT, pDepth, pAux, accum, qOut, qOutCnt, sO, sD, sC, sCnt, smp, N);
        else k_shade_full<false><<<gq, TB, 0, st_>>>(q, nPtr, n, hits, hitI, dTris, dNrm, pO, pD, pT, pDepth, pAux, accum, qOut, qOutCnt, sO, sD, sC, sCnt, smp, N);
    } else if (V_ == 4) {
        if (TRI_) k_trace_shade_full<4, 1, false><<<gq, TB, 0, st_>>>(dNodes, dTrav, q, nPtr, n, dTris, dNrm, pO, pD, pT, pDepth, pAux, accum, qOut, qOutCnt, sO, sD, sC, sCnt, smp, N);
        else k_trace_shade_full<4, 0, false><<<gq, TB, 0, st_>>>(dNodes, dTrav, q, nPtr, n, dTris, dNrm, pO, pD, pT, pDepth, pAux, accum, qOut, qOutCnt, sO, sD, sC, sCnt, smp, N);
    } else DISPF(k_trace_shade_full, gq, (dNodes, dTrav, q, nPtr, n, dTris, dNrm, pO, pD, pT, pDepth, pAux, accum, qOut, qOutCnt, sO, sD, sC, sCnt, smp, N));
}
void WavefrontRenderer::launchFullShadow(const int* nPtr, int n, int* fetch) {
    if (O_.persistent) { int g = nPtr ? ptShadowBlocks_ : std::min(ptShadowBlocks_, (n + 31) / 32); DISPF(k_shadow_full_pt, g, (dNodes, dTrav, nPtr, n, fetch, sO, sD, sC, accum)); return; }
    const int g = nPtr ? gridCap_ : (n + TB - 1) / TB;
    if (V_ == 4) { if (TRI_) k_shadow_full<4, 1, false><<<g, TB, 0, st_>>>(dNodes, dTrav, nPtr, n, sO, sD, sC, accum); else k_shadow_full<4, 0, false><<<g, TB, 0, st_>>>(dNodes, dTrav, nPtr, n, sO, sD, sC, accum); return; }
    DISPF(k_shadow_full, g, (dNodes, dTrav, nPtr, n, sO, sD, sC, accum));
}
// device-driven passes (no host readback): queue counts live in cnt + 8 * depth ([0] queue, [1] shadow queue,
// [2] / [3] persistent fetch counters, [8] = next depth's queue); every bounce is launched with a fixed grid.
// This is the sequence captured into the CUDA graph.
void WavefrontRenderer::enqueueFullPasses(int spp, int sBase) {
    const int K = K_, DEPTHS = DEPTHS_; const size_t NK = NK_;
    for (int s = sBase; s < sBase + spp; s += K) {
        k_raygen<<<(int)((NK + 255) / 256), 256, 0, st_>>>(pO, pD, pT, pDepth, q0, s, K);
        CK(cudaMemsetAsync(pAux, 0, NK * 4, st_));
        CK(cudaMemsetAsync(cnt, 0, (DEPTHS + 1) * 8 * 4, st_));
        k_setcount<<<1, 32, 0, st_>>>(cnt, (int)NK);
        int* qIn = q0; int* qOut = q1;
        for (int depth = 0; depth < DEPTHS; depth++) {
            int* c = cnt + depth * 8;
            launchFullTrace(qIn, c, 0, qOut, c + 8, c + 1, c + 2, s);
            launchFullShadow(c + 1, 0, c + 3);
            int* t = qIn; qIn = qOut; qOut = t;
        }
        k_sumcounts<<<1, 32, 0, st_>>>(cnt, DEPTHS, dtot);
    }
}

void WavefrontRenderer::renderFull(int spp, FrameStats& S, int sBase, bool clear) {
    const int N = N_, K = K_, DEPTHS = DEPTHS_; const size_t NK = NK_;
    if (clear) CK(cudaMemsetAsync(accum, 0, (size_t)N * 16, st_));
    CK(cudaMemsetAsync(dtot, 0, 16, st_)); evUsed = 0;
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> pairs;
    CK(cudaEventRecord(f0, st_));
    if (O_.graph) {
        if (!gexec_ || gSpp_ != spp || gBase_ != sBase) {   // the sample range is baked into the captured launches
            if (gexec_) CK(cudaGraphExecDestroy(gexec_));
            cudaGraph_t g; CK(cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal)); enqueueFullPasses(spp, sBase); CK(cudaStreamEndCapture(st_, &g));
            CK(cudaGraphInstantiate(&gexec_, g, 0)); size_t nn = 0; CK(cudaGraphGetNodes(g, nullptr, &nn)); CK(cudaGraphDestroy(g));
            if (gSpp_ != spp) fprintf(stderr, "CUDA graph: %zu nodes for %d spp (%d passes x %d bounces)\n", nn, spp, (spp + K - 1) / K, DEPTHS);
            gSpp_ = spp; gBase_ = sBase;
        }
        CK(cudaGraphLaunch(gexec_, st_));
    } else {
        for (int s = sBase; s < sBase + spp; s += K) {
            k_raygen<<<(int)((NK + 255) / 256), 256, 0, st_>>>(pO, pD, pT, pDepth, q0, s, K);
            CK(cudaMemsetAsync(pAux, 0, NK * 4, st_));
            int* qIn = q0; int* qOut = q1; int n = (int)NK;
            for (int depth = 0; depth < DEPTHS && n > 0; depth++) {
                CK(cudaMemsetAsync(cnt, 0, 16, st_));
                cudaEvent_t a = nextEvent(), b = nextEvent(); CK(cudaEventRecord(a, st_)); launchFullTrace(qIn, nullptr, n, qOut, cnt, cnt + 1, cnt + 2, s); CK(cudaEventRecord(b, st_)); pairs.push_back({ a, b });
                S.rays += n;
                CK(cudaMemcpyAsync(hcnt, cnt, 8, cudaMemcpyDeviceToHost, st_)); CK(cudaStreamSynchronize(st_));
                int nNext = hcnt[0], nSh = hcnt[1];
                if (nSh > 0) { cudaEvent_t c = nextEvent(), d = nextEvent(); CK(cudaEventRecord(c, st_)); launchFullShadow(nullptr, nSh, cnt + 3); CK(cudaEventRecord(d, st_)); pairs.push_back({ c, d }); S.shadowRays += nSh; }
                int* t = qIn; qIn = qOut; qOut = t; n = nNext;
            }
        }
    }
    CK(cudaEventRecord(f1, st_)); CK(cudaEventSynchronize(f1));
    float ms; CK(cudaEventElapsedTime(&ms, f0, f1)); S.frameMs = ms;
    if (O_.graph) { S.traceMs = ms; unsigned long long tot[2]; CK(cudaMemcpy(tot, dtot, 16, cudaMemcpyDeviceToHost)); S.rays = (double)tot[0]; S.shadowRays = (double)tot[1]; }
    else for (auto& pr : pairs) { CK(cudaEventElapsedTime(&ms, pr.first, pr.second)); S.traceMs += ms; }
    CK(cudaGetLastError());
}

cudaEvent_t WavefrontRenderer::nextEvent() { if (evUsed == evs.size()) { cudaEvent_t e; CK(cudaEventCreate(&e)); evs.push_back(e); } return evs[evUsed++]; }

void WavefrontRenderer::render(int spp, FrameStats& S, int sBase, bool clear) {
    if (O_.full) { renderFull(spp, S, sBase, clear); return; }
    const int N = N_, K = K_, DEPTHS = DEPTHS_; const size_t NK = NK_;
    if (clear) CK(cudaMemset(accum, 0, N * 16));
    CK(cudaMemset(dtot, 0, 16)); evUsed = 0;
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> pairs;
    CK(cudaEventRecord(f0));
    for (int s = sBase; s < sBase + spp; s += K) {
        k_raygen<<<(int)((NK + 255) / 256), 256>>>(pO, pD, pT, pDepth, q0, s, K);
        int* qIn = q0; int* qOut = q1; int* qS = q2; int n = (int)NK;
        if (!O_.devloop) {
            for (int depth = 0; depth < DEPTHS && n > 0; depth++) {
                CK(cudaMemsetAsync(cnt, 0, 16));
                // persistent kernels read counts from cnt+8*DEPTHS: set them to (n, nShadow)
                if (O_.persistent) { hNK[0] = n; CK(cudaMemcpyAsync(cnt + 8 * DEPTHS, hNK, 4, cudaMemcpyHostToDevice)); }
                qOut_ = qOut; smp_ = s;
                cudaEvent_t a = nextEvent(), b = nextEvent(); CK(cudaEventRecord(a)); launchTrace(qIn, n); CK(cudaEventRecord(b)); pairs.push_back({ a, b });
                S.rays += n;
                if (!O_.fuse)
                    k_shade<<<(n + 127) / 128, 128>>>(qIn, n, hits, dTris, dNrm, dAlb, pO, pD, pT, pDepth, accum, qOut, cnt, sO, sD, sC, cnt + 1, nullptr, s, N);
                CK(cudaMemcpyAsync(hcnt, cnt, 8, cudaMemcpyDeviceToHost)); CK(cudaStreamSynchronize(0));
                int nNext = hcnt[0], nSh = hcnt[1];
                if (O_.fuse == 2) S.shadowRays += nSh;
                else if (nSh > 0) {
                    if (O_.persistent) { hNK[1] = nSh; CK(cudaMemcpyAsync(cnt + 8 * DEPTHS + 1, hNK + 1, 4, cudaMemcpyHostToDevice)); }
                    cudaEvent_t c = nextEvent(), d = nextEvent(); CK(cudaEventRecord(c)); launchShadow(nSh); CK(cudaEventRecord(d)); pairs.push_back({ c, d }); S.shadowRays += nSh;
                }
                int* t = qIn; qIn = qOut; qOut = t; n = nNext;
            }
        } else {
            CK(cudaMemsetAsync(cnt, 0, DEPTHS * 8 * 4)); CK(cudaMemcpyAsync(cnt, hNK, 4, cudaMemcpyHostToDevice));
            for (int depth = 0; depth < DEPTHS; depth++) {
                int* c = cnt + depth * 8;
                if (O_.sort && depth > 0) {
                    CK(cudaMemsetAsync(hist, 0, kSortBins * 4));
                    k_hist<<<ptBlocks, TB>>>(qIn, c, c + 5, keys, hist);
                    k_scan<<<1, 1024>>>(hist);
                    k_scatter<<<ptBlocks, TB>>>(qIn, c, c + 6, keys, hist, qS);
                    int* t = qIn; qIn = qS; qS = t;
                }
                cudaEvent_t a = nextEvent(), b = nextEvent(); CK(cudaEventRecord(a));
                DISPATCH(k_trace_dev, ptBlocks, TB, (dNodes, dTrav, qIn, c, c + 2, pO, pD, hits));
                CK(cudaEventRecord(b)); pairs.push_back({ a, b });
                k_shade_dev<<<ptBlocks, TB>>>(qIn, c, c + 4, hits, dTris, dNrm, dAlb, pO, pD, pT, pDepth, accum, qOut, c + 8, sO, sD, sC, c + 1, keys, s, N);
                cudaEvent_t cc = nextEvent(), d = nextEvent(); CK(cudaEventRecord(cc));
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
    if (O_.devloop) { unsigned long long tot[2]; CK(cudaMemcpy(tot, dtot, 16, cudaMemcpyDeviceToHost)); S.rays = (double)tot[0]; S.shadowRays = (double)tot[1]; }
    CK(cudaGetLastError());
}

// ---------------- runtime updates (interactive viewer); called between render() calls ----------------
void WavefrontRenderer::setParams(const Params& H) { CK(cudaDeviceSynchronize()); H_ = H; CK(cudaMemcpyToSymbol(P, &H, sizeof H)); }

void WavefrontRenderer::setMaterials(const std::vector<Material>& mats) {
    CK(cudaDeviceSynchronize());
    std::vector<float3> alb; for (const Material& m : mats) alb.push_back(m.baseColor);
    if (!alb.empty()) CK(cudaMemcpy(dAlb, alb.data(), alb.size() * sizeof(float3), cudaMemcpyHostToDevice));
    if (O_.full && F_.mats && (int)mats.size() == nMats_) CK(cudaMemcpy((void*)F_.mats, mats.data(), mats.size() * sizeof(Material), cudaMemcpyHostToDevice));
}

void WavefrontRenderer::setEnvironment(const std::vector<float4>* texels, float envRotDeg, bool sunOn) {
    if (!O_.full) return;
    CK(cudaDeviceSynchronize());
    if (texels && F_.hasEnv && texels->size() == (size_t)F_.envW * F_.envH) CK(cudaMemcpy((void*)F_.envTex, texels->data(), texels->size() * 16, cudaMemcpyHostToDevice));
    F_.envRot = envRotDeg * 3.14159265f / 180.f;
    if (sunOn != sunOn_) {   // light-type selection probabilities, same rule as initFull
        sunOn_ = sunOn;
        bool triL = F_.nEm > 0 && O_.lightMode != kLightBSDF, envL = F_.hasEnv && O_.lightMode != kLightBSDF;
        int nt = (sunOn ? 1 : 0) + (triL ? 1 : 0) + (envL ? 1 : 0);
        F_.pSun = sunOn ? 1.f / nt : 0.f; F_.pTri = triL ? 1.f / nt : 0.f; F_.pEnv = envL ? 1.f / nt : 0.f;
    }
    CK(cudaMemcpyToSymbol(PF, &F_, sizeof F_));
}

void WavefrontRenderer::readAccum(std::vector<float4>& acc) const { acc.resize(N_); CK(cudaMemcpy(acc.data(), accum, (size_t)N_ * 16, cudaMemcpyDeviceToHost)); }
