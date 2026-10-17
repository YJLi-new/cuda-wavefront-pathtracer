// accel/bvh_builder.h - CPU binned-SAH BVH2 builder (16 bins, leaves of at most leafMax triangles)
#pragma once
#include "accel/aabb.h"
#include "scene/mesh.h"
#include <vector>

struct BuildNode { AABB box; int left = -1, right = -1, first = 0, count = 0; bool leaf() const { return count > 0; } };

struct BVHBuilder {
    std::vector<AABB> tb; std::vector<float3> cent; std::vector<int> idx; std::vector<BuildNode> nodes;
    int leafMax = 4; int maxDepth = 0;
    static const int NB = 16;
    void build(const Mesh& m);          // nodes[0] is the root; idx is the triangle order referenced by leaves
private:
    int rec(int begin, int end, int depth);
};
