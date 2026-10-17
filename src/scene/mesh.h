// scene/mesh.h - indexed triangle mesh with per-triangle material ids and a per-material albedo table
#pragma once
#include "core/vec.h"
#include "scene/material.h"
#include "scene/texture.h"
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
    // textured models (Assimp loader): per-vertex uv (origin bottom-left) and tangents (xyz, handedness w; empty if no
    // material has a normal map), texture images, and texture bindings by material id (may be shorter than the table)
    std::vector<float2> uv;
    std::vector<float4> tan;
    std::vector<TextureImage> tex;
    std::vector<MaterialTextures> mtex;
    bool hasN() const { return !n.empty() && n.size() == v.size(); }
    bool hasUV() const { return !uv.empty() && uv.size() == v.size(); }
    bool hasTan() const { return !tan.empty() && tan.size() == v.size(); }
};
