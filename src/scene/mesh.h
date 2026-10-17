// scene/mesh.h - indexed triangle mesh with per-triangle material ids and a per-material albedo table
#pragma once
#include "core/vec.h"
#include "scene/material.h"
#include <string>
#include <vector>

struct Mesh {
    std::vector<float3> v;      // vertices
    std::vector<float3> n;      // optional per-vertex normals (same size as v) or empty
    std::vector<int3>   f;      // triangles
    std::vector<int>    fm;     // per-triangle material id
    std::vector<float3> kd;     // material albedo table (id 0 = default)
    std::vector<Material> mat;  // full materials by id (optional; ids missing here are Lambertian kd)
    std::vector<std::string> matName;   // material names by id where known (OBJ: usemtl names; id 0 = unnamed default)
    bool hasN() const { return !n.empty() && n.size() == v.size(); }
};
