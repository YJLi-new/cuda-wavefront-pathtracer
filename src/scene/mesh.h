// scene/mesh.h - indexed triangle mesh with per-triangle material ids and a per-material albedo table
#pragma once
#include "core/vec.h"
#include <vector>

struct Mesh {
    std::vector<float3> v;      // vertices
    std::vector<float3> n;      // optional per-vertex normals (same size as v) or empty
    std::vector<int3>   f;      // triangles
    std::vector<int>    fm;     // per-triangle material id
    std::vector<float3> kd;     // material albedo table (id 0 = default)
    bool hasN() const { return !n.empty() && n.size() == v.size(); }
};
