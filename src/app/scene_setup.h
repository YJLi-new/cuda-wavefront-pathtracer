// app/scene_setup.h - command-line options and scene setup shared by the batch renderer (app/pt.cpp) and the
// interactive viewer (app/viewer.cpp): load the mesh / scene file / synthetic scene, materials, lighting and camera, build
// the BVH (CPU binned SAH or GPU H-PLOC) in the selected layout and fill the renderer's upload buffers and options.
#pragma once
#include "render/renderer.h"
#include "scene/instances.h"
#include "scene/lighting.h"
#include "scene/mesh.h"
#include <string>
#include <vector>

struct Args {
    std::string mesh, camera, out = "out", variant = "bvh2", c2wKey = "c2w_blender_zup", tri = "mt";
    int fuse = -1, leaf = 4, batch = 64, synthetic = 0, frame = 0, w = 512, h = 512, spp = 256, bounces = 8, persistent = 0, filterBH = 1, smooth = 1, warm = 8, yupToZup = -1, reps = 1, devloop = 0, sort = 0;
    float fx = 0, fy = 0, hfov = 0; unsigned seed = 1000;
    float3 albedo = mk3(-1, 0, 0);
    Lighting light;
    float epsScale = 1e-4f, near = 0.f;
    std::string c2w, lighting;
    std::string materials, env, integrator = "auto", lightSampling = "mis";
    float envStrength = -1.f, envRot = -1e30f; int rr = 0, rrDepth = 3, gdEnergy = 1, bumpShadow = 0;
    std::string loader = "auto", envSampling = "importance", texFilter = "linear", textures = "base,emission,normal,metallic,roughness";
    float aperture = -1.f, focusDist = -1.f;   // thin lens (< 0: from the camera file, else pinhole / unused)
    // P3: scenes with instances / animation, BVH builder and CWBVH collapse, CUDA graphs
    std::string scene, builder = "sah", collapse = "sah";
    int frames = -1, frameStart = 0, flatten = 0, graph = -1;
};

// parses the batch-renderer options into A; false (after a message) on an unknown option or an invalid value
bool parseArgs(int argc, char** argv, Args& A);

struct SceneSetup {
    Args A;
    Mesh M;                                   // the render mesh (flattened scene or all BLAS meshes back to back)
    SceneDesc SD; std::vector<Mesh> meshes; std::vector<int> meshMatOff, meshTriOff;   // scene files
    std::vector<Material> mats;               // material table as uploaded
    SceneBuffers S; Params H; RenderOptions O;
    std::vector<int> order, meshRoot, posOf;  // BVH leaf order of the triangles, BLAS roots, leaf position per triangle
    float c2w[16]; float extent = 0.f;
    bool sceneMode = false, inst = false, full = false, textured = false;
    int V = 2, TRI = 0, K = 1, F = 1, maxDepth = 0, reportLeaf = 0;
    double treeMs = 0, postMs = 0, sah = 0, bvhS = 0; size_t nNodes = 0, nodeBytes = 0;
    // instanced scenes, before each frame f: instance records, world-space emissive triangles + their power CDF and the
    // per-material light pdf (same arithmetic as the flattened mesh)
    void frameData(int f, std::vector<DevInstance>& di, std::vector<float4>& emT, std::vector<float>& emC, std::vector<float>& mlp) const;
};

// Runs the whole setup from Z.A (logs to stderr). autoCamera: without a camera, frame the scene instead of failing (viewer).
// Returns 0 on success, else the process exit code.
int setupScene(SceneSetup& Z, bool autoCamera = false);
