// io/mesh_loader.h - dependency-free OBJ/MTL and PLY loaders
#pragma once
#include "scene/mesh.h"
#include <string>

bool loadOBJ(const std::string& path, Mesh& m);   // positions, optional vn normals, usemtl -> Kd albedo
bool loadPLY(const std::string& path, Mesh& m);   // ascii / binary_little_endian, optional nx ny nz, per-face material_index
bool loadMesh(const std::string& path, Mesh& m);  // by extension: .ply -> PLY, anything else -> OBJ
