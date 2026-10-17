// render/renderer.h - wavefront path tracer on the GPU (Laine et al. 2013): raygen -> per-bounce
// [trace (closest hit) -> shade (sun NEE + cosine continuation) -> shadow (any hit)] with atomic-append queues.
#pragma once
#include "render/params.h"
#include "scene/environment.h"
#include <string>
#include <utility>
#include <vector>

struct RenderOptions {
    int variant = 2;      // BVH layout: 2, 4 or 8 (compressed)
    int tri = 0;          // triangle test: 0 Moller-Trumbore, 1 Woop
    int fuse = 0;         // 1 fused trace+shade kernel, 2 also traces the shadow ray inline
    int persistent = 0;   // experimental persistent-thread trace/shadow kernels
    int devloop = 0;      // experimental device-driven bounce loop (no host sync)
    int sort = 0;         // experimental bounce-ray sort (implies devloop)
    int batch = 64;       // samples per pixel in flight per wavefront pass
    int bounces = 8;      // max bounce depth (camera hit = depth 0)
    int full = 0;         // 1: full material integrator (always fused; persistent/devloop/sort are legacy-only)
    int lightMode = kLightMIS; int rr = 0, rrDepth = 3;
    int gdEnergy = 1;     // glossy-diffuse energy compensation (albedo scaling); 0 = plain Fresnel-weighted mix
    int envUniform = 0;   // environment light sampling: 0 importance (luminance x sin theta), 1 uniform sphere
    int texNearest = 0;   // texture filtering: 0 bilinear, 1 nearest texel
    int bumpShadow = 0;   // bump-terminator shadowing term (Chiang et al. 2019) on normal-mapped diffuse lobes
};

// host copies of the scene, uploaded once by init()
struct SceneBuffers {
    std::vector<float4> nodes; int nodeStride = 4;   // float4s per node
    std::vector<float4> tris, nrm, woop;             // see accel/tri_pack.h; nrm/woop may be empty
    std::vector<float3> albedo;
    // full integrator only
    std::vector<Material> mats;
    std::vector<float> triLightPdf; std::vector<int> emTri; std::vector<float> emCdf;
    EnvMapHost env; float envRotDeg = 0.f; bool sunOn = true;
    // textures (full integrator; empty when no material is textured), see render/params.h
    std::vector<MaterialTextures> mtex; std::vector<TextureImage> tex; std::vector<float4> uvs, tans;
};

struct FrameStats { double rays = 0, shadowRays = 0, traceMs = 0, frameMs = 0; };

class WavefrontRenderer {
public:
    // Selects device 0, uploads the scene and allocates path state for w*h*batch paths. Exits on CUDA errors.
    void init(const Params& P, const SceneBuffers& S, const RenderOptions& O);
    // renders spp samples per pixel (multiple of batch) into a cleared accumulation buffer
    void render(int spp, FrameStats& st);
    void readAccum(std::vector<float4>& acc) const;   // per-pixel radiance sums (divide by spp)

    double vramMB = 0; int ptBlocks = 0; std::string gpuName;

private:
    void launchTrace(int* q, int n);
    void launchShadow(int n);
    cudaEvent_t nextEvent();
    void initFull(const SceneBuffers& S);

    RenderOptions O_; int V_ = 2, TRI_ = 0, N_ = 0, K_ = 1, DEPTHS_ = 0; size_t NK_ = 0;
    float4 *dNodes = nullptr, *dTris = nullptr, *dNrm = nullptr, *dWoop = nullptr; const float4* dTrav = nullptr; float3* dAlb = nullptr;
    float4 *pO, *pD, *pT, *hits, *sO, *sD, *sC, *accum; int *pDepth, *q0, *q1, *q2, *cnt, *hist; unsigned* keys = nullptr; unsigned long long* dtot;
    int *hcnt, *hNK; float* pAux = nullptr;
    int* qOut_ = nullptr; int smp_ = 0;
    std::vector<cudaEvent_t> evs; size_t evUsed = 0; cudaEvent_t f0, f1;
};

// __launch_bounds__ min-blocks value the renderer was compiled with (0 = no register cap)
int rendererMinBlocks();
