// scene/tangents.h - per-vertex tangent frames for normal mapping when a model has none.
#pragma once
#include "core/vec.h"
#include <vector>

// Tangents from texture-coordinate derivatives (Lengyel, "Computing Tangent Space Basis Vectors for an Arbitrary
// Mesh", 2001): per triangle T = dP/du, B = dP/dv, accumulated per vertex, Gram-Schmidt against the vertex normal
// (or the area-weighted face normal if n is empty). Output tan[i] = (t.xyz, w) with bitangent = w * cross(n, t).
void computeTangents(const std::vector<float3>& p, const std::vector<float3>& n, const std::vector<float2>& uv,
                     const std::vector<int3>& f, std::vector<float4>& tan);
