// scene/texture.h - 8-bit RGBA texture images and per-material texture bindings (glTF 2.0 metallic-roughness model).
// Texture coordinates follow the bottom-left-origin convention (v up, as Assimp delivers them); images are stored
// top row first, so a lookup uses (u, 1 - v). Shared by host and device (MaterialTextures is plain data, 32 bytes).
#pragma once
#include "core/vec.h"
#include <cstdint>
#include <string>
#include <vector>

struct TextureImage {
    std::string path;              // source file (or "<embedded k>")
    int w = 0, h = 0;
    bool srgb = false;             // true: colour data decoded from sRGB on lookup (base colour, emission); false: linear data
    std::vector<uint8_t> rgba;     // w * h * 4 bytes, top row first
};

// Texture ids index Mesh::tex (-1 = none). Lookups multiply the material's factors (glTF semantics):
//   base_color *= tex.rgb, emission *= tex.rgb, metallic *= tex[metallicCh], roughness *= tex[roughnessCh],
//   normal: tangent-space normal map, n = normalize((2 tex.rgb - 1) * (normalScale, normalScale, 1)).
struct MaterialTextures {
    int baseColor = -1, emission = -1, normal = -1, metallic = -1, roughness = -1;
    int metallicCh = 2, roughnessCh = 1;   // glTF packs roughness in G and metalness in B of one image
    float normalScale = 1.f;
};
HD bool hasTextures(const MaterialTextures& t) { return t.baseColor >= 0 || t.emission >= 0 || t.normal >= 0 || t.metallic >= 0 || t.roughness >= 0; }
