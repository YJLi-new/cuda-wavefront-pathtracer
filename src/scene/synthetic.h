// scene/synthetic.h - procedural test scene: ground plane + grid of bumpy spheres, z-up world like Blender
#pragma once
#include "scene/mesh.h"

// gridN x gridN spheres of (nu x nv) quads each -> 2*nu*nv triangles per sphere, plus a (gs x gs) ground.
void makeSynthetic(Mesh& m, int gridN, int nu, int nv, int gs);
// camera-to-world matrix (row-major 4x4, OpenGL camera axes) of the default synthetic-scene view
void syntheticCamera(float c2w[16]);
