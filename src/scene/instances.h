// scene/instances.h - multi-mesh scenes: meshes, instances with per-instance transforms, per-frame animation.
// Scene file (JSON), all keys optional except "meshes" and "instances":
// {
//   "frames": 30,                                   // default number of animation frames
//   "meshes": [ {"file": "a.gltf", "materials": "m.json", "yup_to_zup": 1, "loader": "auto"}, ... ],
//   "instances": [
//     {"mesh": 0, "transform": [12 or 16 numbers, row-major object-to-world]},
//     {"mesh": 1, "translate": [x,y,z], "axis": [x,y,z], "angle_deg": a, "scale": s or [sx,sy,sz],
//                 "translate_per_frame": [..], "angle_deg_per_frame": da, "scale_per_frame": [..]},
//     {"mesh": 1, "transforms": [[12 or 16], [..], ...]}   // explicit per-frame matrices (frame f uses min(f, n-1))
//   ]
// }
// Parametric instances: M(f) = T(translate + f translate_per_frame) R(axis, angle + f angle_per_frame) S(scale + f scale_per_frame).
// Paths are relative to the scene file unless absolute. Materials: as for single meshes (Assimp / OBJ MTL / PLY), an
// optional per-mesh materials JSON (io/scene_files.h format).
#pragma once
#include "scene/mesh.h"
#include <string>
#include <vector>

struct Xform { float m[12]; };   // row-major 3x4 object-to-world

struct InstanceDesc {
    int mesh = 0;
    bool hasMatrix = false; Xform matrix;
    std::vector<Xform> perFrame;
    float3 translate = mk3(0, 0, 0), translateVel = mk3(0, 0, 0), axis = mk3(0, 0, 1), scale = mk3(1, 1, 1), scaleVel = mk3(0, 0, 0);
    float angleDeg = 0.f, angleVel = 0.f;
};
struct MeshDesc { std::string file, materials, loader = "auto"; int yup = -1; };
struct SceneDesc { std::vector<MeshDesc> meshes; std::vector<InstanceDesc> instances; int frames = 1; };

bool loadSceneFile(const std::string& path, SceneDesc& S);
Xform instanceXform(const InstanceDesc& I, int frame);
// inverse in double precision; detSign = sign of the determinant of the linear part
void invertXform(const Xform& M, Xform& inv, float& detSign);
inline float3 xfPoint(const Xform& X, float3 p) { const float* m = X.m; return mk3(m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3], m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7], m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]); }
inline float3 xfVector(const Xform& X, float3 p) { const float* m = X.m; return mk3(m[0] * p.x + m[1] * p.y + m[2] * p.z, m[4] * p.x + m[5] * p.y + m[6] * p.z, m[8] * p.x + m[9] * p.y + m[10] * p.z); }
// normal transform with the inverse: n' = inv^T n (not normalised)
inline float3 xfNormal(const Xform& inv, float3 n) { const float* m = inv.m; return mk3(m[0] * n.x + m[4] * n.y + m[8] * n.z, m[1] * n.x + m[5] * n.y + m[9] * n.z, m[2] * n.x + m[6] * n.y + m[10] * n.z); }

// Normalises a mesh's material table to one Material per id (materials-file entries, else Lambertian kd) and its texture
// bindings to the same length - what the single-mesh path does at render setup.
void normalizeMaterials(Mesh& m);
// Appends the materials/textures of `src` (normalised) to `dst`; returns the material id offset.
int appendMaterials(Mesh& dst, const Mesh& src);
// Appends the geometry of `src` (optionally transformed: positions by X, normals by inv^T and tangents by X, both left
// unnormalised so that interpolation matches the instanced renderer; tangent handedness times sign(det)) with material
// ids offset by matOffset. Missing per-vertex attributes are padded (zero normals = use the geometric normal).
void appendGeometry(Mesh& dst, const Mesh& src, int matOffset, const Xform* X);
