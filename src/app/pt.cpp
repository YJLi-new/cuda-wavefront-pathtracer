// app/pt.cpp - batch renderer CLI: load a mesh, a multi-mesh instanced scene file (--scene) or the synthetic scene,
// build the BVH (CPU binned SAH or GPU H-PLOC), render on the GPU with the wavefront path tracer, print one JSON line
// of statistics per timed frame, and write <out>.png (sRGB) and <out>.pfm (linear radiance); animated scenes write
// <out>_fNNNN.png/.pfm per frame.
#include "app/scene_setup.h"
#include "io/image.h"
#include "render/renderer.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    SceneSetup Z;
    if (!parseArgs(argc, argv, Z.A)) return 1;
    if (int rc = setupScene(Z)) return rc;
    Args& A = Z.A; const Mesh& M = Z.M; const SceneDesc& SD = Z.SD; const Params& H = Z.H; const SceneBuffers& S = Z.S; const RenderOptions& O = Z.O;
    const bool inst = Z.inst, sceneMode = Z.sceneMode, full = Z.full; const int V = Z.V, K = Z.K, F = Z.F, maxDepth = Z.maxDepth, reportLeaf = Z.reportLeaf;
    const double bvhS = Z.bvhS, treeMs = Z.treeMs, postMs = Z.postMs, sah = Z.sah; const size_t nNodes = Z.nNodes, nodeBytes = Z.nodeBytes;
    // ---- GPU ----
    if (A.spp % K) { fprintf(stderr, "spp must be a multiple of batch\n"); return 1; }
    WavefrontRenderer R; R.init(H, S, O);
    const int N = A.w * A.h;
    std::vector<DevInstance> di; std::vector<float4> emT; std::vector<float> emC, mlp;
    for (int fi = 0; fi < F; fi++) {
        const int f = A.frameStart + fi;
        if (inst) { Z.frameData(f, di, emT, emC, mlp); R.setInstances(di, emT, emC, mlp); }
        if (fi == 0) {
            FrameStats W; if (A.warm > 0) { int wsp = ((A.warm + K - 1) / K) * K; R.render(wsp, W); }
            fprintf(stderr, "warmup: %.1f ms, %.1f Mrays/s\n", W.frameMs, (W.rays + W.shadowRays) / (W.traceMs * 1e3));
        }
        for (int r = 0; r < A.reps; r++) {
            FrameStats st; R.render(A.spp, st);
            double tot = st.rays + st.shadowRays;
            printf("{\"rep\":%d,\"variant\":\"bvh%d\",\"tri\":\"%s\",\"persistent\":%d,\"devloop\":%d,\"sort\":%d,\"fuse\":%d,\"leaf\":%d,\"batch\":%d,\"minb\":%d,\"scene\":\"%s\",\"tris\":%zu,\"bvh_nodes\":%zu,\"bvh_bytes\":%zu,\"bvh_depth\":%d,\"bvh_build_s\":%.2f,\"w\":%d,\"h\":%d,\"spp\":%d,\"bounces\":%d,"
                   "\"rays_closest\":%.0f,\"rays_shadow\":%.0f,\"rays_total\":%.0f,\"t_trace_ms\":%.2f,\"t_frame_ms\":%.2f,\"mrays_s_trace\":%.1f,\"mrays_s_e2e\":%.1f,\"avg_path_len\":%.3f,\"vram_MB\":%.0f,\"pt_blocks\":%d,\"gpu\":\"%s\",\"filter\":\"%s\",\"smooth\":%d,\"seed\":%u,\"integrator\":\"%s\",\"lens_r\":%.4f,\"focus_d\":%.4f,\"env_sampling\":\"%s\",\"textures\":%zu,"
                   "\"builder\":\"%s\",\"collapse\":\"%s\",\"bvh_tree_ms\":%.2f,\"bvh_post_ms\":%.2f,\"bvh_sah\":%.4f,\"graph\":%d,\"instances\":%d,\"flatten\":%d,\"frame\":%d,\"tlas_ms\":%.3f}\n",
                   r, V, A.tri.c_str(), A.persistent, A.devloop, A.sort, A.fuse, reportLeaf, K, rendererMinBlocks(),
                   A.synthetic ? "synthetic" : sceneMode ? A.scene.c_str() : A.mesh.c_str(), M.f.size(), nNodes, nodeBytes, maxDepth, bvhS, A.w, A.h, A.spp, A.bounces,
                   st.rays, st.shadowRays, tot, st.traceMs, st.frameMs, tot / (st.traceMs * 1e3), tot / (st.frameMs * 1e3), st.rays / ((double)N * A.spp), R.vramMB, R.ptBlocks, R.gpuName.c_str(), A.filterBH ? "blackman-harris1.5" : "box", H.hasN, A.seed, full ? (A.lightSampling == "mis" ? "full-mis" : A.lightSampling == "nee" ? "full-nee" : "full-bsdf") : "legacy", H.cam.lensR, H.cam.focusD, A.envSampling.c_str(), S.tex.size(),
                   A.builder.c_str(), V == 8 ? A.collapse.c_str() : "-", treeMs, postMs, sah, A.graph, inst ? S.nInst : sceneMode ? (int)SD.instances.size() : 0, A.flatten, f, R.tlasMs);
            fflush(stdout);
        }
        std::vector<float4> acc; R.readAccum(acc);
        std::vector<float> lin((size_t)N * 3); std::vector<uint8_t> rgb((size_t)N * 3);
        for (int i = 0; i < N; i++) { lin[3 * i] = acc[i].x / A.spp; lin[3 * i + 1] = acc[i].y / A.spp; lin[3 * i + 2] = acc[i].z / A.spp; }
        tonemapSRGB8(lin.data(), (size_t)N, rgb.data());
        std::string name = A.out; if (F > 1) { char b[32]; snprintf(b, sizeof b, "_f%04d", f); name += b; }
        writePNG(name + ".png", A.w, A.h, rgb.data()); writePFM(name + ".pfm", A.w, A.h, lin.data());
        fprintf(stderr, "wrote %s.png / .pfm (tonemap: clip[0,1] + sRGB OETF, exposure 0)\n", name.c_str());
    }
    return 0;
}
