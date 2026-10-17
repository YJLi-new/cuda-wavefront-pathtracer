// accel/tri_pack.h - GPU triangle layouts in BVH leaf order
#pragma once
#include "scene/mesh.h"
#include <vector>

// 3 float4 per triangle: (v0, origTri), (e1, mat), (e2, 0); per-corner normals if the mesh has them
void packTriangles(const Mesh& m, const std::vector<int>& order, std::vector<float4>& tris, std::vector<float4>& nrm);
// Woop unit-triangle transform: rows of M = inverse([e1 e2 n]) with translation, stored (row0,row1,row2) as 3 float4
void packWoop(const Mesh& m, const std::vector<int>& order, std::vector<float4>& w);
