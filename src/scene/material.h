// scene/material.h - OpenPBR-style material subset (Academy Software Foundation OpenPBR Surface, base + specular +
// transmission + emission layers only). Shared by host and device (64 bytes, plain data).
//   metallic = 1          : rough conductor, F0 = base_color, F82 tint = specular_color (Kutz et al. 2021)
//   metallic = 0, transm. : rough dielectric BSDF (Walter et al. 2007), refraction tinted by base_color
//   metallic = 0, opaque  : glossy-diffuse = GGX dielectric specular (weight specular_weight, ior) mixed
//                           non-physically with a Lambertian base_color lobe weighted by 1 - F(cos theta_o)
//   emission              : emission_color * emission_strength, two-sided, added on top
// metallic and transmission blend the three bases linearly; roughness r gives GGX alpha = r^2.
#pragma once
#include "core/vec.h"

struct Material {
    float3 baseColor; float metallic;
    float3 specularColor; float roughness;
    float3 emission; float ior;               // emission radiance (color * strength)
    float specularWeight; float transmission; float pad0, pad1;
};

// pure Lambertian material (what OBJ Kd / PLY material_index / lighting-JSON albedos give)
inline Material lambertMaterial(float3 kd) {
    Material m; m.baseColor = kd; m.metallic = 0.f; m.specularColor = mk3(1.f, 1.f, 1.f); m.roughness = 1.f;
    m.emission = mk3(0.f, 0.f, 0.f); m.ior = 1.5f; m.specularWeight = 0.f; m.transmission = 0.f; m.pad0 = m.pad1 = 0.f;
    return m;
}
// OpenPBR defaults (specular on): what a materials-JSON entry starts from
inline Material defaultMaterial() { Material m = lambertMaterial(mk3(0.8f, 0.8f, 0.8f)); m.roughness = 0.5f; m.specularWeight = 1.f; return m; }
inline bool isLambert(const Material& m) { return m.metallic == 0.f && m.transmission == 0.f && m.specularWeight == 0.f; }
inline bool isEmissive(const Material& m) { return m.emission.x > 0.f || m.emission.y > 0.f || m.emission.z > 0.f; }
