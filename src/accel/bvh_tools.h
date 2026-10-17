// accel/bvh_tools.h - BVH tree utilities: H-PLOC node arrays -> BVHBuilder trees, SAH cost, CWBVH collapse.
#pragma once
#include "accel/bvh_builder.h"
#include <vector>

// H-PLOC node array (host copy, accel/hploc.h layout) -> BVHBuilder tree with the root at nodes[0] and the primitives
// of every leaf contiguous in idx. Subtrees of at most leafMax primitives become leaves where that lowers the SAH
// cost (exact bottom-up dynamic programme: C(n) = min(A(n) p(n), A(n) + C(left) + C(right)), i.e. c_node = c_prim = 1).
void treeFromHploc(const std::vector<float4>& hn, int nPrim, int root, int leafMax, BVHBuilder& B);

// SAH cost of a binary tree normalised by the root surface area: sum over internal nodes A/A_root * cNode + sum over
// leaves A/A_root * count * cPrim.
double sahCost(const BVHBuilder& B, double cNode, double cPrim);

// Compressed wide BVH (Ylitie et al. 2017), written from the paper. The binary tree is collapsed into an 8-wide tree by
// the paper's SAH dynamic programme over (binary node, number of wide-tree roots i = 1..7):
//   C(n,1) = min(C_leaf(n), C_dist(n,8) + A(n) c_node), C(n,i) = min(C(n,i-1), C_dist(n,i)),
//   C_dist(n,j) = min_k C(left,k) + C(right,j-k), C_leaf(n) = A(n) p(n) c_prim if p(n) <= pMax, else infinite,
// with c_node = 1, c_prim = 0.3 and leaves of at most pMax = 3 triangles. Child slots are assigned greedily by the cost
// -dot(child centre - node centre, octant direction of the slot) so that octant-ordered traversal is near front to
// back. Node encoding (80 B) as accel/bvh_layouts.h BVH8Flat / accel/traverse.cuh trav8: origin + power-of-two scale
// exponents, 8-bit quantised child boxes, imask, child and triangle base indices, per-child meta byte.
struct CWBVH {
    std::vector<float4> nodes;   // 5 float4 per node, root = node 0
    std::vector<int> order;      // triangle order referenced by the leaves
    int maxLeafTris = 0; double sah = 0;   // collapsed-tree SAH cost (c_node 1, c_prim 0.3) / A(root)
    int pMax = 3; float cNode = 1.f, cPrim = 0.3f;
    void build(const BVHBuilder& B);
};

// concatenation of per-mesh layouts into one node array (instanced scenes): child/triangle indices are offset so that
// every BLAS uses absolute indices; returns the root node index of the appended BLAS
int appendBVH2(std::vector<float4>& all, const std::vector<float4>& blas, int triBase);
int appendBVH8(std::vector<float4>& all, const std::vector<float4>& blas, int triBase);
