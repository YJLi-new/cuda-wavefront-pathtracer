// render/wavefront.cu - wavefront CUDA path tracer kernels and host driver.
// Kernels: raygen / trace(closest) / shade / trace(shadow, any-hit), queue compaction via atomic append.
// Options: BVH2 / BVH4 / compressed BVH8 traversal (accel/traverse.cuh); Moller-Trumbore or Woop triangles;
// fused trace+shade; host-driven loop (one sync per bounce) or experimental device-driven loop (persistent-style
// kernels reading queue counts from device memory) with optional counting-sort binning of bounce rays by
// (direction octant, Morton code of origin). Shading: Lambert + sun next-event estimation + constant sky.
#include "render/renderer.h"
#include "accel/traverse.cuh"
#include "core/rng.h"
#include "core/sampling.h"
#include <cstdio>
#include <cstdlib>

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
    pO[i] = mk4(P.cam.pos, P.near); pD[i] = mk4(d, 1e30f); pT[i] = make_float4(1.f, 1.f, 1.f, __int_as_float(pix)); pDepth[i] = 0; q[i] = i;
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

void WavefrontRenderer::init(const Params& H, const SceneBuffers& S, const RenderOptions& O) {
    O_ = O; if (O_.sort) O_.devloop = 1;
    V_ = O.variant; TRI_ = O.tri;
    int dev = 0; CK(cudaSetDevice(dev)); cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev)); gpuName = prop.name;
    size_t memFree0, memTot; CK(cudaMemGetInfo(&memFree0, &memTot));
    CK(cudaMalloc(&dNodes, S.nodes.size() * 16)); CK(cudaMemcpy(dNodes, S.nodes.data(), S.nodes.size() * 16, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dTris, S.tris.size() * 16)); CK(cudaMemcpy(dTris, S.tris.data(), S.tris.size() * 16, cudaMemcpyHostToDevice));
    if (TRI_ == 1) { CK(cudaMalloc(&dWoop, S.woop.size() * 16)); CK(cudaMemcpy(dWoop, S.woop.data(), S.woop.size() * 16, cudaMemcpyHostToDevice)); }
    dTrav = TRI_ == 1 ? dWoop : dTris;
    if (!S.nrm.empty()) { CK(cudaMalloc(&dNrm, S.nrm.size() * 16)); CK(cudaMemcpy(dNrm, S.nrm.data(), S.nrm.size() * 16, cudaMemcpyHostToDevice)); }
    CK(cudaMalloc(&dAlb, S.albedo.size() * sizeof(float3))); CK(cudaMemcpy(dAlb, S.albedo.data(), S.albedo.size() * sizeof(float3), cudaMemcpyHostToDevice));
    CK(cudaMemcpyToSymbol(P, &H, sizeof H));
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
        fprintf(stderr, "trace kernel: %d blocks/SM x %d SMs (occupancy %.0f%%)\n", bpm, prop.multiProcessorCount, 100.0 * bpm * 128 / prop.maxThreadsPerMultiProcessor);
    }
    CK(cudaEventCreate(&f0)); CK(cudaEventCreate(&f1));
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

cudaEvent_t WavefrontRenderer::nextEvent() { if (evUsed == evs.size()) { cudaEvent_t e; CK(cudaEventCreate(&e)); evs.push_back(e); } return evs[evUsed++]; }

void WavefrontRenderer::render(int spp, FrameStats& S) {
    const int N = N_, K = K_, DEPTHS = DEPTHS_; const size_t NK = NK_;
    CK(cudaMemset(accum, 0, N * 16)); CK(cudaMemset(dtot, 0, 16)); evUsed = 0;
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> pairs;
    CK(cudaEventRecord(f0));
    for (int s = 0; s < spp; s += K) {
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

void WavefrontRenderer::readAccum(std::vector<float4>& acc) const { acc.resize(N_); CK(cudaMemcpy(acc.data(), accum, (size_t)N_ * 16, cudaMemcpyDeviceToHost)); }
