// accel/aabb.h - axis-aligned bounding box (host)
#pragma once
#include "core/vec.h"
#include <cfloat>

struct AABB {
    float3 lo, hi;
    AABB() : lo(mk3(FLT_MAX, FLT_MAX, FLT_MAX)), hi(mk3(-FLT_MAX, -FLT_MAX, -FLT_MAX)) {}
    void grow(float3 p) { lo = vmin(lo, p); hi = vmax(hi, p); }
    void grow(const AABB& b) { lo = vmin(lo, b.lo); hi = vmax(hi, b.hi); }
    float area() const { if (hi.x < lo.x) return 0.f; float3 d = hi - lo; return 2.f * (d.x * d.y + d.y * d.z + d.z * d.x); }
    float3 extent() const { return hi - lo; }
};
