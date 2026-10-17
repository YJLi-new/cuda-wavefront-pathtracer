// accel/instance.h - device instance record of two-level (TLAS/BLAS) scenes, shared by host and device
#pragma once
#include "core/vec.h"

// world-to-object (inv) and object-to-world (fwd) 3x4 rows, BLAS root node (absolute index into the node array),
// mesh id, sign of det(fwd)
struct DevInstance { float4 inv[3]; float4 fwd[3]; int root; int mesh; float detSign; int pad; };

HD float3 xfRowsP(const float4* r, float3 p) { return mk3(r[0].x * p.x + r[0].y * p.y + r[0].z * p.z + r[0].w, r[1].x * p.x + r[1].y * p.y + r[1].z * p.z + r[1].w, r[2].x * p.x + r[2].y * p.y + r[2].z * p.z + r[2].w); }
HD float3 xfRowsV(const float4* r, float3 p) { return mk3(r[0].x * p.x + r[0].y * p.y + r[0].z * p.z, r[1].x * p.x + r[1].y * p.y + r[1].z * p.z, r[2].x * p.x + r[2].y * p.y + r[2].z * p.z); }
// normal transform with the inverse rows: n' = inv^T n
HD float3 xfRowsN(const float4* r, float3 n) { return mk3(r[0].x * n.x + r[1].x * n.y + r[2].x * n.z, r[0].y * n.x + r[1].y * n.y + r[2].y * n.z, r[0].z * n.x + r[1].z * n.y + r[2].z * n.z); }
