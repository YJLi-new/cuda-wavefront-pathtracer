// accel/bvh_layouts.h - GPU node layouts flattened from a BVHBuilder tree
//   BVH2: "children boxes in parent" 64 B nodes (Aila & Laine 2009)
//   BVH4: collapsed from the BVH2, 128 B nodes
//   BVH8: compressed 8-wide, 80 B nodes (in the style of Ylitie et al. 2017: per-node origin + power-of-two scale,
//         8-bit quantized child boxes, octant-ordered child slots, imask + popcount child addressing)
#pragma once
#include "accel/bvh_builder.h"
#include <vector>

static const float EMPTY_BOX = 3e37f;

// BVH2 GPU node (64 B): n0xy=(c0.lo.x,c0.hi.x,c0.lo.y,c0.hi.y) n1xy=(same for c1) nz=(c0.lo.z,c0.hi.z,c1.lo.z,c1.hi.z)
// meta=(c0, c1, cnt0, cnt1): cnt>0 -> leaf, c = first triangle; cnt==0 -> c = internal node index (c=-1 empty)
struct BVH2Flat {
    std::vector<float4> nodes; // 4 float4 per node
    void build(BVHBuilder& B);
private:
    int emit(const BVHBuilder& B, int bn);
};

// BVH4 GPU node (128 B): bminx,bmaxx,bminy,bmaxy,bminz,bmaxz (float4 each, lane = child), child int4, cnt int4
struct BVH4Flat {
    std::vector<float4> nodes; // 8 float4 per node
    void build(BVHBuilder& B);
private:
    int emit(const BVHBuilder& B, int bn);
};

// BVH8 compressed node (80 B, 5 float4). Leaves of at most 4 triangles stored contiguously per node (order).
struct BVH8Flat {
    std::vector<float4> nodes;   // 5 float4 per node
    std::vector<int> order;      // triangle order (index into mesh triangles)
    const BVHBuilder* B = nullptr;
    int maxLeafTris = 0;
    void build(BVHBuilder& Bb);
private:
    static int expo(float ext, float lo, float hi);
    static unsigned pack4(const int* q, int base);
    void emit(int bn, int out);
};
