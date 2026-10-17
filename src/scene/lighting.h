// scene/lighting.h - sun (directional light with a finite angular diameter) plus a constant sky
#pragma once
#include "core/vec.h"

struct Lighting {
    float3 sunTo = mk3(0.3f, -0.5f, 0.8f);   // direction toward the sun, world z-up (normalized by the renderer)
    float3 sunE = mk3(3.f, 3.f, 3.f);        // irradiance on a surface facing the sun, W/m^2
    float sunAngleDeg = 0.526f;              // angular diameter
    float3 sky = mk3(0.3f, 0.3f, 0.3f);      // constant sky radiance
};
