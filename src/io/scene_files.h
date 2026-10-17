// io/scene_files.h - camera and lighting JSON files (formats documented in README.md)
#pragma once
#include "scene/lighting.h"
#include "scene/mesh.h"
#include <string>

// Reads the lighting file into L; if it has "albedo_rgb_linear_by_material_index" the mesh albedo table is replaced.
// The table is then padded so every material id used by the mesh has an entry. Returns false if unreadable.
bool loadLightingFile(const std::string& path, Lighting& L, Mesh& m);

// Reads the camera file. fx/fy are taken from it only when they are still 0 and the file's width/height match w/h;
// otherwise hfov_deg is used (if hfov is still 0 and fx is present). c2w comes from key[frame].
bool loadCameraFile(const std::string& path, const std::string& key, int frame, int w, int h, float& fx, float& fy, float& hfov, float c2w[16]);

// Reads {"materials": [ {name, base_color, metallic, roughness, ior, specular_weight, specular_color, transmission,
// emission_color, emission_strength}, ... ]}. Missing keys take OpenPBR defaults. An entry whose "name" matches a mesh
// material name (OBJ usemtl) is assigned to that id; other entries are assigned by position (entry i -> id i).
bool loadMaterialsFile(const std::string& path, Mesh& mesh);

// Optional thin-lens keys of the camera file: "aperture_radius" (lens radius, world units) and "focus_distance"
// (distance of the plane in focus along the view axis). Values already set (>= 0) are kept.
void loadCameraLens(const std::string& path, float& apertureR, float& focusD);
