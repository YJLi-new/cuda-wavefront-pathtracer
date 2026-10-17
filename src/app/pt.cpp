// app/pt.cpp - batch renderer CLI: load a mesh (or build the synthetic scene), build the BVH on the CPU,
// render on the GPU with the wavefront path tracer, print one JSON line of statistics per timed frame,
// and write <out>.png (sRGB) and <out>.pfm (linear radiance).
#include "accel/bvh_builder.h"
#include "accel/bvh_layouts.h"
#include "accel/tri_pack.h"
#include "core/sampling.h"
#include "io/assimp_loader.h"
#include "io/image.h"
#include "io/mesh_loader.h"
#include "io/scene_files.h"
#include "render/renderer.h"
#include "scene/camera.h"
#include "scene/environment.h"
#include "scene/synthetic.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include "render/bsdf.h"

struct Args {
    std::string mesh, camera, out = "out", variant = "bvh2", c2wKey = "c2w_blender_zup", tri = "mt";
    int fuse = 0, leaf = 4, batch = 64, synthetic = 0, frame = 0, w = 512, h = 512, spp = 256, bounces = 8, persistent = 0, filterBH = 1, smooth = 1, warm = 8, yupToZup = -1, reps = 1, devloop = 0, sort = 0;
    float fx = 0, fy = 0, hfov = 0; unsigned seed = 1000;
    float3 albedo = mk3(-1, 0, 0);
    Lighting light;
    float epsScale = 1e-4f, near = 0.f;
    std::string c2w, lighting;
    std::string materials, env, integrator = "auto", lightSampling = "mis";
    float envStrength = -1.f, envRot = -1e30f; int rr = 0, rrDepth = 3, gdEnergy = 1, bumpShadow = 0;
    std::string loader = "auto", envSampling = "importance", texFilter = "linear", textures = "base,emission,normal,metallic,roughness";
    float aperture = -1.f, focusDist = -1.f;   // thin lens (< 0: from the camera file, else pinhole / unused)
};

static float3 parse3(char** a, int& i) { float3 v = mk3((float)atof(a[i + 1]), (float)atof(a[i + 2]), (float)atof(a[i + 3])); i += 3; return v; }

static bool parseArgs(int argc, char** argv, Args& A) {
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto nf = [&]() { return (float)atof(argv[++i]); }; auto ni = [&]() { return atoi(argv[++i]); };
        if (k == "--mesh") A.mesh = argv[++i]; else if (k == "--camera") A.camera = argv[++i]; else if (k == "--out") A.out = argv[++i];
        else if (k == "--variant") A.variant = argv[++i]; else if (k == "--c2w_key") A.c2wKey = argv[++i]; else if (k == "--c2w") A.c2w = argv[++i]; else if (k == "--tri") A.tri = argv[++i];
        else if (k == "--synthetic") A.synthetic = ni(); else if (k == "--frame") A.frame = ni(); else if (k == "--width") A.w = ni(); else if (k == "--height") A.h = ni();
        else if (k == "--spp") A.spp = ni(); else if (k == "--bounces") A.bounces = ni(); else if (k == "--persistent") A.persistent = ni(); else if (k == "--filter") A.filterBH = std::string(argv[++i]) == "bh";
        else if (k == "--smooth") A.smooth = ni(); else if (k == "--warm") A.warm = ni(); else if (k == "--yup_to_zup") A.yupToZup = ni(); else if (k == "--reps") A.reps = ni(); else if (k == "--batch") A.batch = ni();
        else if (k == "--devloop") A.devloop = ni(); else if (k == "--sort") A.sort = ni(); else if (k == "--fuse") A.fuse = ni(); else if (k == "--leaf") A.leaf = ni();
        else if (k == "--fx") A.fx = nf(); else if (k == "--fy") A.fy = nf(); else if (k == "--hfov") A.hfov = nf(); else if (k == "--seed") A.seed = (unsigned)ni();
        else if (k == "--sun_to") A.light.sunTo = parse3(argv, i); else if (k == "--sun_rgb") A.light.sunE = parse3(argv, i); else if (k == "--sky") A.light.sky = parse3(argv, i); else if (k == "--albedo") A.albedo = parse3(argv, i);
        else if (k == "--materials") A.materials = argv[++i]; else if (k == "--env") A.env = argv[++i]; else if (k == "--env_strength") A.envStrength = nf(); else if (k == "--env_rot") A.envRot = nf();
        else if (k == "--integrator") A.integrator = argv[++i]; else if (k == "--light_sampling") A.lightSampling = argv[++i]; else if (k == "--rr") A.rr = ni(); else if (k == "--rr_depth") A.rrDepth = ni(); else if (k == "--gd_energy") A.gdEnergy = ni();
        else if (k == "--bump_shadow") A.bumpShadow = ni(); else if (k == "--tex_filter") A.texFilter = argv[++i]; else if (k == "--textures") A.textures = argv[++i];
        else if (k == "--loader") A.loader = argv[++i]; else if (k == "--env_sampling") A.envSampling = argv[++i]; else if (k == "--aperture") A.aperture = nf(); else if (k == "--focus_dist") A.focusDist = nf();
        else if (k == "--sun_angle") A.light.sunAngleDeg = nf(); else if (k == "--eps") A.epsScale = nf(); else if (k == "--near") A.near = nf(); else if (k == "--lighting") A.lighting = argv[++i];
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return false; }
    }
    if (A.sort) A.devloop = 1;
    if (A.integrator != "auto" && A.integrator != "legacy" && A.integrator != "full") { fprintf(stderr, "--integrator must be auto, legacy or full\n"); return false; }
    if (A.lightSampling != "mis" && A.lightSampling != "nee" && A.lightSampling != "bsdf") { fprintf(stderr, "--light_sampling must be mis, nee or bsdf\n"); return false; }
    if (A.loader != "auto" && A.loader != "builtin" && A.loader != "assimp") { fprintf(stderr, "--loader must be auto, builtin or assimp\n"); return false; }
    if (A.envSampling != "importance" && A.envSampling != "uniform") { fprintf(stderr, "--env_sampling must be importance or uniform\n"); return false; }
    if (A.texFilter != "linear" && A.texFilter != "nearest") { fprintf(stderr, "--tex_filter must be linear or nearest\n"); return false; }
    return true;
}

int main(int argc, char** argv) {
    Args A;
    if (!parseArgs(argc, argv, A)) return 1;
    // ---- scene ----
    Mesh M; auto t0 = std::chrono::steady_clock::now();
    float c2w[16]; identityC2W(c2w); bool haveC2w = false;
    if (A.synthetic) {
        makeSynthetic(M, A.synthetic, 200, 100, 200);
        syntheticCamera(c2w); haveC2w = true;
        if (A.hfov == 0 && A.fx == 0) A.hfov = 50.f;
    } else {
        // loader: built-in OBJ/PLY (dependency-free; default for .obj/.ply) or Assimp (default for everything else)
        std::string ext = A.mesh.substr(A.mesh.find_last_of('.') + 1); for (auto& ch : ext) ch = (char)tolower((unsigned char)ch);
        bool useAssimp = A.loader == "assimp" || (A.loader == "auto" && ext != "obj" && ext != "ply");
        if (!(useAssimp ? loadAssimp(A.mesh, M) : loadMesh(A.mesh, M))) { fprintf(stderr, "failed to load mesh %s (%s loader)\n", A.mesh.c_str(), useAssimp ? "assimp" : "built-in"); return 1; }
        // glTF is y-up by definition: converted to the z-up world unless --yup_to_zup 0
        int yup = A.yupToZup >= 0 ? A.yupToZup : (ext == "gltf" || ext == "glb") ? 1 : 0;
        if (yup) {
            for (auto& v : M.v) v = mk3(v.x, -v.z, v.y);
            for (auto& n : M.n) n = mk3(n.x, -n.z, n.y);
            for (auto& t : M.tan) t = make_float4(t.x, -t.z, t.y, t.w);
        }
    }
    if (!A.smooth) M.n.clear();
    {   // --textures: texture kinds to use (the others fall back to the material factors)
        auto on = [&](const char* k) { return ("," + A.textures + ",").find(std::string(",") + k + ",") != std::string::npos; };
        for (auto& t : M.mtex) {
            if (!on("base")) t.baseColor = -1; if (!on("emission")) t.emission = -1; if (!on("normal")) t.normal = -1;
            if (!on("metallic")) t.metallic = -1; if (!on("roughness")) t.roughness = -1;
        }
    }
    bool textured = false; for (auto& t : M.mtex) textured |= hasTextures(t);
    if (textured && !M.hasUV()) { fprintf(stderr, "warn: textures without texture coordinates ignored\n"); textured = false; }
    if (textured && A.albedo.x >= 0) for (auto& t : M.mtex) t.baseColor = -1;   // --albedo replaces textured base colours too
    if (!A.lighting.empty() && !loadLightingFile(A.lighting, A.light, M)) return 1;
    if (A.albedo.x >= 0) for (auto& kd : M.kd) kd = A.albedo;
    if (M.kd.empty()) M.kd.push_back(mk3(0.8f, 0.8f, 0.8f));
    if (!A.materials.empty() && !loadMaterialsFile(A.materials, M)) return 1;
    if (!A.env.empty()) A.light.envMap = A.env;
    if (A.envStrength >= 0.f) A.light.envStrength = A.envStrength;
    if (A.envRot > -1e29f) A.light.envRotDeg = A.envRot;
    // material table: materials-file entries, else Lambertian kd; --albedo overrides every base colour
    std::vector<Material> mats;
    { int maxMat = 0; for (int t : M.fm) if (t > maxMat) maxMat = t;
      size_t nm = std::max(std::max(M.kd.size(), M.mat.size()), (size_t)maxMat + 1);
      for (size_t j = 0; j < nm; j++) { Material m = j < M.mat.size() ? M.mat[j] : lambertMaterial(j < M.kd.size() ? M.kd[j] : mk3(0.8f, 0.8f, 0.8f)); if (A.albedo.x >= 0) m.baseColor = A.albedo; mats.push_back(m); } }
    bool needFull = A.rr != 0 || !A.light.envMap.empty() || textured;
    for (auto& m : mats) if (!isLambert(m) || isEmissive(m)) needFull = true;
    const bool full = A.integrator == "full" || (A.integrator == "auto" && needFull);
    if (A.integrator == "legacy" && needFull) fprintf(stderr, "warn: --integrator legacy ignores specular/transmissive/emissive materials, the environment map and RR\n");
    if (full && (A.persistent || A.devloop || A.sort || A.fuse != 1)) { fprintf(stderr, "note: the full integrator always uses the fused kernel; --persistent/--devloop/--sort/--fuse ignored\n"); A.persistent = A.devloop = A.sort = 0; A.fuse = 1; }
    double loadS = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    AABB sb; for (auto& v : M.v) sb.grow(v);
    float extent = maxc(sb.extent());
    fprintf(stderr, "mesh: %zu verts %zu tris, normals=%d, mats=%zu, extent=%.3f load %.2fs\n", M.v.size(), M.f.size(), (int)M.hasN(), M.kd.size(), extent, loadS);

    // ---- camera ----
    if (!A.camera.empty()) {
        if (!loadCameraFile(A.camera, A.c2wKey, A.frame, A.w, A.h, A.fx, A.fy, A.hfov, c2w)) return 1;
        haveC2w = true;
        loadCameraLens(A.camera, A.aperture, A.focusDist);
    }
    if (!A.c2w.empty()) { std::string s = A.c2w; for (auto& ch : s) if (ch == ',') ch = ' '; const char* p = s.c_str(); char* e; for (int j = 0; j < 16; j++) { c2w[j] = (float)strtod(p, &e); p = e; } haveC2w = true; }
    if (!haveC2w) { fprintf(stderr, "no camera\n"); return 1; }
    if (A.fx == 0) { if (A.hfov == 0) A.hfov = 50.f; A.fx = focalFromHfov(A.w, A.hfov); }
    if (A.fy == 0) A.fy = A.fx;
    Params H; memset(&H, 0, sizeof H);
    H.cam = makeCamera(c2w, A.fx, A.fy, A.w, A.h);
    if (A.aperture > 0.f) {
        if (!(A.focusDist > 0.f)) { fprintf(stderr, "thin lens: --aperture needs --focus_dist (or \"focus_distance\" in the camera file)\n"); return 1; }
        H.cam.lensR = A.aperture; H.cam.focusD = A.focusDist;
        fprintf(stderr, "thin lens: aperture radius %.4f, focus distance %.4f\n", A.aperture, A.focusDist);
    }
    H.sunTo = normalize(A.light.sunTo); H.sunE = A.light.sunE; H.sky = A.light.sky; H.sunCos = cosf(0.5f * A.light.sunAngleDeg * 3.14159265f / 180.f);
    H.maxB = A.bounces; H.seed = A.seed; H.eps = A.epsScale * extent; H.near = A.near; H.hasN = M.hasN() ? 1 : 0; H.filterBH = A.filterBH;
    H.bmin = sb.lo; float3 ex3 = sb.extent(); H.binv = mk3(16.f / fmaxf(ex3.x, 1e-6f), 16.f / fmaxf(ex3.y, 1e-6f), 16.f / fmaxf(ex3.z, 1e-6f));
    buildBlackmanHarrisTable(H.ftab, kFilterTable, 1.5);
    fprintf(stderr, "cam pos (%.3f %.3f %.3f) fx %.2f %dx%d spp %d bounces %d eps %.2e sunTo (%.3f %.3f %.3f)\n", H.cam.pos.x, H.cam.pos.y, H.cam.pos.z, A.fx, A.w, A.h, A.spp, A.bounces, H.eps, H.sunTo.x, H.sunTo.y, H.sunTo.z);

    // ---- BVH ----
    t0 = std::chrono::steady_clock::now();
    BVHBuilder B; B.leafMax = (A.variant == "bvh8" && A.leaf > 4) ? 4 : A.leaf; B.build(M);
    SceneBuffers S; int V = A.variant == "bvh4" ? 4 : A.variant == "bvh8" ? 8 : 2; int TRI = A.tri == "woop" ? 1 : 0;
    std::vector<int> order;
    if (V == 2) { BVH2Flat F; F.build(B); S.nodes.swap(F.nodes); S.nodeStride = 4; order = B.idx; }
    else if (V == 4) { BVH4Flat F; F.build(B); S.nodes.swap(F.nodes); S.nodeStride = 8; order = B.idx; }
    else { BVH8Flat F; F.build(B); S.nodes.swap(F.nodes); S.nodeStride = 5; order.swap(F.order); }
    packTriangles(M, order, S.tris, S.nrm);
    if (TRI == 1) packWoop(M, order, S.woop);
    for (auto& m : mats) S.albedo.push_back(m.baseColor);
    if (full) {
        S.mats = mats; S.sunOn = luminance(A.light.sunE) > 0.f;
        // emissive triangles (BVH leaf order) sampled proportionally to power = luminance(emission) * area
        std::vector<double> pw; double tot = 0;
        for (size_t kk = 0; kk < order.size(); kk++) {
            int t = order[kk]; int mi = M.fm.empty() ? 0 : M.fm[t]; const Material& m = mats[mi];
            if (!isEmissive(m)) continue;
            float3 e1 = M.v[M.f[t].y] - M.v[M.f[t].x], e2 = M.v[M.f[t].z] - M.v[M.f[t].x]; double area = 0.5 * length(cross(e1, e2));
            double pwr = luminance(m.emission) * area; if (!(pwr > 0)) continue;
            S.emTri.push_back((int)kk); pw.push_back(pwr); tot += pwr;
        }
        if (!S.emTri.empty()) {
            S.triLightPdf.assign(order.size(), 0.f); S.emCdf.assign(S.emTri.size() + 1, 0.f); double c = 0;
            for (size_t e = 0; e < S.emTri.size(); e++) {
                int kk = S.emTri[e], t = order[kk]; float3 e1 = M.v[M.f[t].y] - M.v[M.f[t].x], e2 = M.v[M.f[t].z] - M.v[M.f[t].x]; double area = 0.5 * length(cross(e1, e2));
                S.triLightPdf[kk] = (float)(pw[e] / tot / area); c += pw[e]; S.emCdf[e + 1] = (float)(c / tot);
            }
            S.emCdf.back() = 1.f;
        }
        if (!A.light.envMap.empty()) { if (!buildEnvMap(A.light.envMap, A.light.envStrength, S.env)) return 1; S.envRotDeg = A.light.envRotDeg; }
        if (textured) {
            S.mtex = M.mtex; S.mtex.resize(mats.size()); S.tex.swap(M.tex);
            packAttributes(M, order, S.uvs, S.tans);
        }
    }
    double bvhS = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    size_t nNodes = S.nodes.size() / S.nodeStride, nodeBytes = S.nodes.size() * 16;
    fprintf(stderr, "BVH%d: %zu nodes (%.1f MB), depth %d, build %.2fs, tri=%s\n", V, nNodes, nodeBytes / 1e6, B.maxDepth, bvhS, A.tri.c_str());

    // ---- GPU ----
    int K = A.batch < 1 ? 1 : A.batch;
    if (A.spp % K) { fprintf(stderr, "spp must be a multiple of batch\n"); return 1; }
    RenderOptions O; O.full = full ? 1 : 0; O.lightMode = A.lightSampling == "nee" ? kLightNEE : A.lightSampling == "bsdf" ? kLightBSDF : kLightMIS; O.rr = A.rr; O.rrDepth = A.rrDepth; O.gdEnergy = A.gdEnergy; O.envUniform = A.envSampling == "uniform" ? 1 : 0; O.texNearest = A.texFilter == "nearest" ? 1 : 0; O.bumpShadow = A.bumpShadow;
    O.variant = V; O.tri = TRI; O.fuse = A.fuse; O.persistent = A.persistent; O.devloop = A.devloop; O.sort = A.sort; O.batch = K; O.bounces = A.bounces;
    WavefrontRenderer R; R.init(H, S, O);
    const int N = A.w * A.h;

    FrameStats W; if (A.warm > 0) { int wsp = ((A.warm + K - 1) / K) * K; R.render(wsp, W); }
    fprintf(stderr, "warmup: %.1f ms, %.1f Mrays/s\n", W.frameMs, (W.rays + W.shadowRays) / (W.traceMs * 1e3));
    for (int r = 0; r < A.reps; r++) {
        FrameStats st; R.render(A.spp, st);
        double tot = st.rays + st.shadowRays;
        printf("{\"rep\":%d,\"variant\":\"bvh%d\",\"tri\":\"%s\",\"persistent\":%d,\"devloop\":%d,\"sort\":%d,\"fuse\":%d,\"leaf\":%d,\"batch\":%d,\"minb\":%d,\"scene\":\"%s\",\"tris\":%zu,\"bvh_nodes\":%zu,\"bvh_bytes\":%zu,\"bvh_depth\":%d,\"bvh_build_s\":%.2f,\"w\":%d,\"h\":%d,\"spp\":%d,\"bounces\":%d,"
               "\"rays_closest\":%.0f,\"rays_shadow\":%.0f,\"rays_total\":%.0f,\"t_trace_ms\":%.2f,\"t_frame_ms\":%.2f,\"mrays_s_trace\":%.1f,\"mrays_s_e2e\":%.1f,\"avg_path_len\":%.3f,\"vram_MB\":%.0f,\"pt_blocks\":%d,\"gpu\":\"%s\",\"filter\":\"%s\",\"smooth\":%d,\"seed\":%u,\"integrator\":\"%s\",\"lens_r\":%.4f,\"focus_d\":%.4f,\"env_sampling\":\"%s\",\"textures\":%zu}\n",
               r, V, A.tri.c_str(), A.persistent, A.devloop, A.sort, A.fuse, B.leafMax, K, rendererMinBlocks(),
               A.synthetic ? "synthetic" : A.mesh.c_str(), M.f.size(), nNodes, nodeBytes, B.maxDepth, bvhS, A.w, A.h, A.spp, A.bounces,
               st.rays, st.shadowRays, tot, st.traceMs, st.frameMs, tot / (st.traceMs * 1e3), tot / (st.frameMs * 1e3), st.rays / ((double)N * A.spp), R.vramMB, R.ptBlocks, R.gpuName.c_str(), A.filterBH ? "blackman-harris1.5" : "box", H.hasN, A.seed, full ? (A.lightSampling == "mis" ? "full-mis" : A.lightSampling == "nee" ? "full-nee" : "full-bsdf") : "legacy", H.cam.lensR, H.cam.focusD, A.envSampling.c_str(), S.tex.size());
        fflush(stdout);
    }
    std::vector<float4> acc; R.readAccum(acc);
    std::vector<float> lin((size_t)N * 3); std::vector<uint8_t> rgb((size_t)N * 3);
    for (int i = 0; i < N; i++) { lin[3 * i] = acc[i].x / A.spp; lin[3 * i + 1] = acc[i].y / A.spp; lin[3 * i + 2] = acc[i].z / A.spp; }
    tonemapSRGB8(lin.data(), (size_t)N, rgb.data());
    writePNG(A.out + ".png", A.w, A.h, rgb.data()); writePFM(A.out + ".pfm", A.w, A.h, lin.data());
    fprintf(stderr, "wrote %s.png / .pfm (tonemap: clip[0,1] + sRGB OETF, exposure 0)\n", A.out.c_str());
    return 0;
}
