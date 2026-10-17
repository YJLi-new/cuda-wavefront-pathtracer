// accel/hploc.h - GPU BVH2 construction by hierarchical PLOC (H-PLOC, Benthin et al. 2024), written from the paper.
//   1. Morton codes of the primitive-box centroids extended by the primitive index -> unique 64-bit keys (ib =
//      ceil(log2 n) index bits, min(21, (64 - ib) / 3) Morton bits per axis), radix-sorted (CUB).
//   2. One kernel, one thread per sorted primitive: bottom-up construction of the implicit LBVH over the sorted keys
//      (Apetrei 2014: a node's parent is chosen by comparing the key deltas at the two ends of its range, the second
//      thread to reach a parent continues with the union range). Every LBVH node holds a list of clusters stored at
//      the start of its primitive range. When a range holds more than HPLOC_T clusters (or is the root) the whole warp
//      loads the clusters of both children (<= 2 HPLOC_T = 32), and runs PLOC iterations on them (nearest neighbour =
//      smallest merged-box surface area within a radius of HPLOC_R positions, mutual nearest neighbours are merged
//      into a new BVH node) until HPLOC_T clusters remain (1 at the root); the survivors are written back.
// Output: 2n-1 nodes of 2 float4: (lo.xyz, left), (hi.xyz, right) with child ids as int bits. Nodes [0, n) are the
// leaves in Morton order (left = primitive id, right = -1); internal nodes [n, 2n-1) are allocated by an atomic counter,
// so every child id is smaller than its parent's and the root (the last merge) is node 2n-2 (node 0 when n == 1).
#pragma once
#include "core/vec.h"
#include <vector>

static constexpr int HPLOC_T = 16;   // merge threshold: clusters kept per LBVH subtree (half a warp)
static constexpr int HPLOC_R = 8;    // PLOC nearest-neighbour search radius (positions in Morton order)

struct HplocBuilder {
    int capacity = 0;
    float4* nodes = nullptr;                                 // 2 (2 capacity - 1) float4
    unsigned long long *keys = nullptr, *keysSorted = nullptr;
    int *clusters = nullptr, *bounds = nullptr, *counter = nullptr; unsigned* cbox = nullptr;
    void* tmp = nullptr; size_t tmpBytes = 0;
    void reserve(int n);
    // primitive boxes lo/hi (n float4 each, device). Stream-ordered (capturable); returns the root node id.
    int build(const float4* primLo, const float4* primHi, int n, cudaStream_t s);
    void release();
};

// BVH2 GPU layout (64 B nodes, accel/bvh_layouts.h) built directly from the H-PLOC nodes on the device: internal node
// j <- H-PLOC node n + j, root = n - 2; leaf children -> (primitive id, count 1). n == 1: node 0 = the leaf + an empty
// child (root 0). Used for the per-frame TLAS.
void hplocToBVH2(const float4* hnodes, int n, float4* out, cudaStream_t s);
inline int hplocBVH2Root(int n) { return n > 1 ? n - 2 : 0; }

// Host convenience for static meshes: uploads the boxes, builds, downloads the node array (2n-1 nodes). gpuMs = time of
// the device build (Morton codes + sort + H-PLOC kernel) measured with CUDA events. Returns the root id.
int hplocBuildHost(const std::vector<float4>& lo, const std::vector<float4>& hi, std::vector<float4>& nodesOut, double& gpuMs);
