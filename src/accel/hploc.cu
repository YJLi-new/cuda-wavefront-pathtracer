// accel/hploc.cu - H-PLOC GPU BVH builder (Benthin et al. 2024), written from scratch from the paper (see hploc.h).
#include "accel/hploc.h"
#include <cub/device/device_radix_sort.cuh>
#include <cstdio>
#include <cstdlib>

#define HCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#define FULLMASK 0xffffffffu

// order-preserving float <-> uint mapping for atomicMin/atomicMax on floats
__device__ __forceinline__ unsigned f2o(float f) { unsigned u = __float_as_uint(f); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
__device__ __forceinline__ float o2f(unsigned u) { return __uint_as_float((u & 0x80000000u) ? (u & 0x7fffffffu) : ~u); }

__global__ void k_hp_init(unsigned* cbox, int* counter, int n) {
    if (threadIdx.x < 3) { cbox[threadIdx.x] = 0xffffffffu; cbox[3 + threadIdx.x] = 0u; }
    if (threadIdx.x == 0) *counter = n;
}
__global__ void k_hp_centroid_box(const float4* __restrict__ lo, const float4* __restrict__ hi, int n, unsigned* cbox) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float3 c = mk3(0.f, 0.f, 0.f); bool ok = i < n;
    if (ok) c = (xyz(lo[i]) + xyz(hi[i])) * 0.5f;
    float mn[3] = { ok ? c.x : 3.4e38f, ok ? c.y : 3.4e38f, ok ? c.z : 3.4e38f }, mx[3] = { ok ? c.x : -3.4e38f, ok ? c.y : -3.4e38f, ok ? c.z : -3.4e38f };
    for (int a = 0; a < 3; a++)
        for (int o = 16; o > 0; o >>= 1) { mn[a] = fminf(mn[a], __shfl_xor_sync(FULLMASK, mn[a], o)); mx[a] = fmaxf(mx[a], __shfl_xor_sync(FULLMASK, mx[a], o)); }
    if ((threadIdx.x & 31) == 0) for (int a = 0; a < 3; a++) { atomicMin(&cbox[a], f2o(mn[a])); atomicMax(&cbox[3 + a], f2o(mx[a])); }
}
__device__ __forceinline__ unsigned long long expandBits21(unsigned v) {   // 21 bits -> every third bit of 63
    unsigned long long x = v & 0x1fffffu;
    x = (x | x << 32) & 0x1f00000000ffffull; x = (x | x << 16) & 0x1f0000ff0000ffull; x = (x | x << 8) & 0x100f00f00f00f00full;
    x = (x | x << 4) & 0x10c30c30c30c30c3ull; x = (x | x << 2) & 0x1249249249249249ull; return x;
}
// key = (Morton code with mb bits per axis) << ib | primitive index (ib index bits): unique 64-bit keys
__global__ void k_hp_keys(const float4* __restrict__ lo, const float4* __restrict__ hi, int n, const unsigned* cbox, unsigned long long* keys, int mb, int ib) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float3 bl = mk3(o2f(cbox[0]), o2f(cbox[1]), o2f(cbox[2])), bh = mk3(o2f(cbox[3]), o2f(cbox[4]), o2f(cbox[5]));
    float3 c = (xyz(lo[i]) + xyz(hi[i])) * 0.5f, e = bh - bl;
    float qx = e.x > 0.f ? (c.x - bl.x) / e.x : 0.f, qy = e.y > 0.f ? (c.y - bl.y) / e.y : 0.f, qz = e.z > 0.f ? (c.z - bl.z) / e.z : 0.f;
    float s = (float)(1u << mb), smax = s - 1.f;
    unsigned ix = (unsigned)fminf(fmaxf(qx * s, 0.f), smax), iy = (unsigned)fminf(fmaxf(qy * s, 0.f), smax), iz = (unsigned)fminf(fmaxf(qz * s, 0.f), smax);
    unsigned long long m = (expandBits21(ix) << 2) | (expandBits21(iy) << 1) | expandBits21(iz);
    keys[i] = (m << ib) | (unsigned long long)i;
}
__global__ void k_hp_leaves(const float4* __restrict__ lo, const float4* __restrict__ hi, const unsigned long long* __restrict__ keys, int n, int ib, float4* nodes, int* C, int* bounds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    int p = (int)(keys[i] & ((1ull << ib) - 1ull));
    nodes[2 * i] = mk4(xyz(lo[p]), __int_as_float(p)); nodes[2 * i + 1] = mk4(xyz(hi[p]), __int_as_float(-1));
    C[i] = i; bounds[i] = -1;
}

__device__ __forceinline__ float halfArea(float3 lo, float3 hi) { float3 d = hi - lo; return d.x * d.y + d.y * d.z + d.z * d.x; }
__device__ __forceinline__ float3 shfl3(float3 v, int src) { return mk3(__shfl_sync(FULLMASK, v.x, src), __shfl_sync(FULLMASK, v.y, src), __shfl_sync(FULLMASK, v.z, src)); }
__device__ __forceinline__ float3 shfl3down(float3 v, int d) { return mk3(__shfl_down_sync(FULLMASK, v.x, d), __shfl_down_sync(FULLMASK, v.y, d), __shfl_down_sync(FULLMASK, v.z, d)); }

// Warp-cooperative PLOC on the clusters of LBVH node [L, R] with split s (left child [L, s], right [s + 1, R]).
// All 32 lanes must call it together. Cluster ids live in C at the start of each child's range (invalid slots = -1).
__device__ void plocMerge(int lane, int L, int R, int s, bool root, float4* nodes, int* C, int* counter, int* shId, float* shB) {
    const unsigned lt = (1u << lane) - 1u;
    int nL = min(s - L + 1, HPLOC_T), nR = min(R - s, HPLOC_T);
    int slot = lane < nL ? L + lane : (lane < nL + nR ? s + 1 + (lane - nL) : -1);
    int id = slot >= 0 ? __ldcg(&C[slot]) : -1;
    unsigned vm = __ballot_sync(FULLMASK, id >= 0);
    int num = __popc(vm);
    if (id >= 0) shId[__popc(vm & lt)] = id;
    __syncwarp();
    id = lane < num ? shId[lane] : -1;
    __syncwarp();
    float3 lo = mk3(0.f, 0.f, 0.f), hi = lo;
    if (id >= 0) { float4 a = __ldcg(&nodes[2 * id]), b = __ldcg(&nodes[2 * id + 1]); lo = xyz(a); hi = xyz(b); }
    const int target = root ? 1 : HPLOC_T;
    while (num > target) {
        // nearest neighbour within the radius; pairs are totally ordered by (area, lower position, upper position) so
        // the globally best pair is always mutual and every iteration merges at least one pair
        unsigned long long best = ~0ull;
        for (int r = 1; r <= HPLOC_R; r++) {
            float3 olo = shfl3down(lo, r), ohi = shfl3down(hi, r);
            unsigned long long key = ~0ull;
            if (lane + r < num) key = ((unsigned long long)__float_as_uint(halfArea(vmin(lo, olo), vmax(hi, ohi))) << 32) | ((unsigned)lane << 5) | (unsigned)(lane + r);
            if (key < best) best = key;
            unsigned long long k2 = __shfl_up_sync(FULLMASK, key, r);
            if (lane >= r && k2 < best) best = k2;
        }
        int pa = (int)((best >> 5) & 31u), pb = (int)(best & 31u);
        int nn = lane < num ? (pa == lane ? pb : pa) : lane;
        int nnn = __shfl_sync(FULLMASK, nn, nn);
        bool mutual = lane < num && nn != lane && nnn == lane;
        bool lead = mutual && lane < nn;
        float3 plo = shfl3(lo, nn), phi = shfl3(hi, nn); int pid = __shfl_sync(FULLMASK, id, nn);
        unsigned lm = __ballot_sync(FULLMASK, lead);
        int base = 0;
        if (lane == 0) base = atomicAdd(counter, __popc(lm));
        base = __shfl_sync(FULLMASK, base, 0);
        if (lead) {
            int nid = base + __popc(lm & lt);
            lo = vmin(lo, plo); hi = vmax(hi, phi);
            nodes[2 * nid] = mk4(lo, __int_as_float(id)); nodes[2 * nid + 1] = mk4(hi, __int_as_float(pid));
            id = nid;
        }
        bool keep = lane < num && !(mutual && lane > nn);
        unsigned km = __ballot_sync(FULLMASK, keep);
        if (keep) { int dd = __popc(km & lt); shId[dd] = id; shB[6 * dd] = lo.x; shB[6 * dd + 1] = lo.y; shB[6 * dd + 2] = lo.z; shB[6 * dd + 3] = hi.x; shB[6 * dd + 4] = hi.y; shB[6 * dd + 5] = hi.z; }
        __syncwarp();
        num = __popc(km);
        if (lane < num) { id = shId[lane]; lo = mk3(shB[6 * lane], shB[6 * lane + 1], shB[6 * lane + 2]); hi = mk3(shB[6 * lane + 3], shB[6 * lane + 4], shB[6 * lane + 5]); }
        else id = -1;
        __syncwarp();
    }
    if (slot >= 0) C[slot] = -1;
    __syncwarp();
    if (lane < num) C[L + lane] = id;
    __syncwarp();
}

__global__ void __launch_bounds__(64) k_hploc(float4* nodes, const unsigned long long* __restrict__ keys, int* C, int* bounds, int* counter, int n) {
    __shared__ int shId[2][32]; __shared__ float shB[2][32 * 6];
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = i < n; int L = i, R = i;
    while (__any_sync(FULLMASK, active)) {
        __syncwarp(); __threadfence();   // this warp's cluster/node writes become visible before any parent hand-over below
        int s = 0; bool mrg = false;
        if (active) {
            int prev;
            if (L == 0 || (R != n - 1 && (keys[R] ^ keys[R + 1]) < (keys[L - 1] ^ keys[L]))) { s = R; prev = atomicExch(&bounds[R], L); if (prev != -1) R = prev; }
            else { s = L - 1; prev = atomicExch(&bounds[L - 1], R); if (prev != -1) L = prev; }
            if (prev == -1) active = false;   // first child to arrive: the sibling's thread continues
            else { __threadfence(); mrg = (R - L + 1 > HPLOC_T) || (L == 0 && R == n - 1); }
        }
        unsigned mm = __ballot_sync(FULLMASK, mrg);
        while (mm) {
            int src = __ffs(mm) - 1; mm &= mm - 1;
            int l = __shfl_sync(FULLMASK, L, src), r = __shfl_sync(FULLMASK, R, src), sp = __shfl_sync(FULLMASK, s, src);
            plocMerge(lane, l, r, sp, l == 0 && r == n - 1, nodes, C, counter, shId[w], shB[w]);
        }
        if (active && L == 0 && R == n - 1) active = false;
    }
}

__global__ void k_hp_to_bvh2(const float4* __restrict__ hn, int n, float4* out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    const float E = 3e37f;
    if (n == 1) {
        if (j == 0) {
            float4 a = hn[0], b = hn[1]; int p = __float_as_int(a.w), m1 = -1, k0 = 1, k1 = 0;
            out[0] = make_float4(a.x, b.x, a.y, b.y); out[1] = make_float4(E, E, E, E); out[2] = make_float4(a.z, b.z, E, E);
            out[3] = make_float4(__int_as_float(p), __int_as_float(m1), __int_as_float(k0), __int_as_float(k1));
        }
        return;
    }
    if (j >= n - 1) return;
    int id = n + j; float4 a = hn[2 * id], b = hn[2 * id + 1];
    int ch[2] = { __float_as_int(a.w), __float_as_int(b.w) }; float4 cl[2], chh[2]; int c[2], k[2];
    for (int q = 0; q < 2; q++) {
        int x = ch[q]; cl[q] = hn[2 * x]; chh[q] = hn[2 * x + 1];
        if (x < n) { c[q] = __float_as_int(cl[q].w); k[q] = 1; } else { c[q] = x - n; k[q] = 0; }
    }
    out[4 * j] = make_float4(cl[0].x, chh[0].x, cl[0].y, chh[0].y);
    out[4 * j + 1] = make_float4(cl[1].x, chh[1].x, cl[1].y, chh[1].y);
    out[4 * j + 2] = make_float4(cl[0].z, chh[0].z, cl[1].z, chh[1].z);
    out[4 * j + 3] = make_float4(__int_as_float(c[0]), __int_as_float(c[1]), __int_as_float(k[0]), __int_as_float(k[1]));
}

void HplocBuilder::reserve(int n) {
    if (n <= capacity) return;
    release(); capacity = n;
    HCK(cudaMalloc(&nodes, (size_t)2 * (2 * n - 1) * sizeof(float4)));
    HCK(cudaMalloc(&keys, (size_t)n * 8)); HCK(cudaMalloc(&keysSorted, (size_t)n * 8));
    HCK(cudaMalloc(&clusters, (size_t)n * 4)); HCK(cudaMalloc(&bounds, (size_t)n * 4)); HCK(cudaMalloc(&counter, 4)); HCK(cudaMalloc(&cbox, 24));
    tmpBytes = 0; HCK(cub::DeviceRadixSort::SortKeys(nullptr, tmpBytes, keys, keysSorted, n, 0, 64));
    HCK(cudaMalloc(&tmp, tmpBytes > 0 ? tmpBytes : 16));
}
void HplocBuilder::release() {
    if (!capacity) return;
    cudaFree(nodes); cudaFree(keys); cudaFree(keysSorted); cudaFree(clusters); cudaFree(bounds); cudaFree(counter); cudaFree(cbox); cudaFree(tmp);
    capacity = 0;
}
int HplocBuilder::build(const float4* lo, const float4* hi, int n, cudaStream_t s) {
    reserve(n);
    int g = (n + 255) / 256;
    k_hp_init<<<1, 32, 0, s>>>(cbox, counter, n);
    k_hp_centroid_box<<<g, 256, 0, s>>>(lo, hi, n, cbox);
    int ib = 1; while ((1ll << ib) < n) ib++;
    const int mb = (64 - ib) / 3 < 21 ? (64 - ib) / 3 : 21;
    k_hp_keys<<<g, 256, 0, s>>>(lo, hi, n, cbox, keys, mb, ib);
    size_t tb = tmpBytes; HCK(cub::DeviceRadixSort::SortKeys(tmp, tb, keys, keysSorted, n, 0, 3 * mb + ib, s));
    k_hp_leaves<<<g, 256, 0, s>>>(lo, hi, keysSorted, n, ib, nodes, clusters, bounds);
    if (n > 1) k_hploc<<<(n + 63) / 64, 64, 0, s>>>(nodes, keysSorted, clusters, bounds, counter, n);
    return n > 1 ? 2 * n - 2 : 0;
}

void hplocToBVH2(const float4* hnodes, int n, float4* out, cudaStream_t s) {
    int m = n > 1 ? n - 1 : 1;
    k_hp_to_bvh2<<<(m + 127) / 128, 128, 0, s>>>(hnodes, n, out);
}

int hplocBuildHost(const std::vector<float4>& lo, const std::vector<float4>& hi, std::vector<float4>& nodesOut, double& gpuMs) {
    int n = (int)lo.size();
    float4 *dLo, *dHi; HCK(cudaMalloc(&dLo, (size_t)n * 16)); HCK(cudaMalloc(&dHi, (size_t)n * 16));
    HCK(cudaMemcpy(dLo, lo.data(), (size_t)n * 16, cudaMemcpyHostToDevice)); HCK(cudaMemcpy(dHi, hi.data(), (size_t)n * 16, cudaMemcpyHostToDevice));
    HplocBuilder H; H.reserve(n);
    cudaEvent_t e0, e1; HCK(cudaEventCreate(&e0)); HCK(cudaEventCreate(&e1));
    H.build(dLo, dHi, n, 0);   // first build: warms up the kernels / module load (not timed)
    HCK(cudaEventRecord(e0)); int root = H.build(dLo, dHi, n, 0); HCK(cudaEventRecord(e1)); HCK(cudaEventSynchronize(e1));
    float ms = 0; HCK(cudaEventElapsedTime(&ms, e0, e1)); gpuMs = ms;
    HCK(cudaGetLastError());
    nodesOut.resize((size_t)2 * (2 * n - 1));
    HCK(cudaMemcpy(nodesOut.data(), H.nodes, nodesOut.size() * 16, cudaMemcpyDeviceToHost));
    H.release(); cudaFree(dLo); cudaFree(dHi); cudaEventDestroy(e0); cudaEventDestroy(e1);
    return root;
}
