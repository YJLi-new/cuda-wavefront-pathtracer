// io/assimp_loader.h - model loading through Assimp (obj, ply, fbx, gltf/glb, ...) with materials and textures.
// Optional: built only with -DCWP_WITH_ASSIMP=ON (the default); otherwise loadAssimp() reports that and fails, and the
// dependency-free OBJ/PLY loaders (io/mesh_loader.h) remain.
#pragma once
#include "scene/mesh.h"
#include <string>

bool assimpAvailable();
// Flattens the node hierarchy (node transforms applied), triangulates, keeps uv set 0, normals (flat ones are generated
// where a mesh has none) and tangents (taken from the file, or computed when missing, if a material has a normal map).
// Materials:
//   glTF metallic-roughness: base colour, metallic, roughness factors x textures; KHR_materials_transmission (factor),
//   KHR_materials_ior, emissive factor x texture x KHR_materials_emissive_strength; normal texture with its scale.
//   other formats: diffuse colour/texture, metallic/roughness factors if present (else roughness from the Phong
//   shininess n via GGX alpha = sqrt(2 / (n + 2))), specular weight = luminance of the specular colour.
// Textures are decoded with stb_image (8 bits per channel); base colour and emission are flagged sRGB, the rest linear.
bool loadAssimp(const std::string& path, Mesh& m);
