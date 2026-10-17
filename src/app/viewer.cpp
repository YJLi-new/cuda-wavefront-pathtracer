// app/viewer.cpp - interactive viewer: the batch renderer's scene setup and wavefront path tracer with progressive
// accumulation, displayed through GLFW + OpenGL (GLEW) + Dear ImGui; file dialogs through tinyfiledialogs.
//   Display: CUDA-GL interop (the display kernel writes into a mapped pixel-buffer object, then glTexSubImage2D from it);
//   when interop is unavailable (e.g. WSL) or --interop 0, the kernel writes a device buffer that is copied to the host.
//   Controls: right mouse drag = look, W/A/S/D = move, Q/E = down/up, Shift = x5, Alt = x0.2, mouse wheel = speed,
//   middle click or Ctrl + left click = focus the thin lens at the cursor, P = screenshot (PNG), Ctrl+O = open a scene.
//   Every change of the camera, lighting, materials or render settings restarts the accumulation.
// Options: every batch-renderer option (app/scene_setup.h; --spp = accumulation target, default unlimited) plus
//   --frames N            render exactly N viewer frames (each --passes passes of --batch samples) and then
//   --screenshot out.png  save the displayed render image (read back from the display texture), --exit: quit after N
//   --screenshot_window f save the whole window (render + UI) after N frames; --screenshot_pfm f: linear sums / spp
//   --passes P            batch passes per viewer frame (default 1); --interop auto|0|1; --exposure EV;
//   --tonemap clip|reinhard|aces; --win_width / --win_height; --hide_ui 1
//   --test_updates 1      (test) after the N frames, change camera, lens, sun, environment and material 0 through the
//                         runtime-update path, render N more frames and save the outputs again with suffix _upd
// With --frames N, the image equals the batch renderer's with --spp N*P*batch and the same options (same sample indices;
// bitwise with --batch 1, where accumulation order is deterministic).
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "tinyfiledialogs.h"
#include "accel/aabb.h"
#include "app/display.h"
#include "app/scene_setup.h"
#include "io/image.h"
#include "render/bsdf.h"
#include "render/renderer.h"
#include "scene/camera.h"
#include "scene/environment.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#define CKV(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
static const float kPi = 3.14159265f;

struct ViewerOpts {
    int frames = 0, passes = 1, tonemap = kDisplayClip, winW = 0, winH = 0, hideUi = 0, testUpdates = 0; bool exitAfter = false, sppGiven = false;
    float exposure = 0.f; std::string screenshot, screenshotWindow, screenshotPfm, interop = "auto";
};

// splits argv into viewer options and the batch-renderer options passed to parseArgs
static bool splitArgs(int argc, char** argv, ViewerOpts& V, std::vector<char*>& rest) {
    rest.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i]; bool hasV = i + 1 < argc;
        if (k == "--exit") V.exitAfter = true;
        else if (k == "--frames" && hasV) V.frames = atoi(argv[++i]);
        else if (k == "--passes" && hasV) V.passes = std::max(1, atoi(argv[++i]));
        else if (k == "--screenshot" && hasV) V.screenshot = argv[++i];
        else if (k == "--screenshot_window" && hasV) V.screenshotWindow = argv[++i];
        else if (k == "--screenshot_pfm" && hasV) V.screenshotPfm = argv[++i];
        else if (k == "--interop" && hasV) V.interop = argv[++i];
        else if (k == "--exposure" && hasV) V.exposure = (float)atof(argv[++i]);
        else if (k == "--win_width" && hasV) V.winW = atoi(argv[++i]);
        else if (k == "--win_height" && hasV) V.winH = atoi(argv[++i]);
        else if (k == "--hide_ui" && hasV) V.hideUi = atoi(argv[++i]);
        else if (k == "--test_updates" && hasV) V.testUpdates = atoi(argv[++i]);
        else if (k == "--tonemap" && hasV) { std::string t = argv[++i]; V.tonemap = t == "reinhard" ? kDisplayReinhard : t == "aces" ? kDisplayACES : kDisplayClip; }
        else { if (k == "--spp") V.sppGiven = true; rest.push_back(argv[i]); }
    }
    if (V.interop != "auto" && V.interop != "0" && V.interop != "1") { fprintf(stderr, "--interop must be auto, 0 or 1\n"); return false; }
    return true;
}

// restarts the process on another model / scene file: scene-specific options are dropped, render settings kept
static void reopenWith(int argc, char** argv, const std::string& file) {
    static const char* drop1[] = { "--mesh", "--scene", "--synthetic", "--camera", "--c2w", "--c2w_key", "--frame", "--fx", "--fy", "--materials", "--yup_to_zup",
                                   "--frames", "--screenshot", "--screenshot_window", "--screenshot_pfm", "--frame_start" };
    std::vector<std::string> a; a.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i]; bool d = false;
        for (const char* x : drop1) if (k == x) { d = true; i++; break; }
        if (k == "--exit") d = true;
        if (!d) a.push_back(argv[i]);
    }
    std::string ext = file.substr(file.find_last_of('.') + 1); for (auto& c : ext) c = (char)tolower((unsigned char)c);
    a.push_back(ext == "json" ? "--scene" : "--mesh"); a.push_back(file);
    std::vector<char*> av; for (auto& s : a) av.push_back(const_cast<char*>(s.c_str())); av.push_back(nullptr);
    fprintf(stderr, "opening %s\n", file.c_str()); fflush(stderr);
#ifdef _WIN32
    _execv(argv[0], av.data());
#else
    execv("/proc/self/exe", av.data());
#endif
    fprintf(stderr, "could not restart the viewer for %s\n", file.c_str());
}

// ---- CPU ray cast for click-to-focus (brute force over the triangles; instanced scenes in object space) ----
static bool hitTri(float3 o, float3 d, float3 a, float3 b, float3 c, float& t) {
    float3 e1 = b - a, e2 = c - a, p = cross(d, e2); float det = dot(e1, p); if (fabsf(det) < 1e-12f) return false;
    float inv = 1.f / det; float3 s = o - a; float u = dot(s, p) * inv; if (u < 0.f || u > 1.f) return false;
    float3 q = cross(s, e1); float v = dot(d, q) * inv; if (v < 0.f || u + v > 1.f) return false;
    float tt = dot(e2, q) * inv; if (tt > 1e-6f && tt < t) { t = tt; return true; } return false;
}
static float castRay(const SceneSetup& Z, int frame, float3 o, float3 d) {   // d unnormalized; returns t (inf if no hit)
    float t = INFINITY;
    if (!Z.inst) { const Mesh& M = Z.M; for (const int3& f : M.f) hitTri(o, d, M.v[f.x], M.v[f.y], M.v[f.z], t); return t; }
    for (const InstanceDesc& I : Z.SD.instances) {
        Xform X = instanceXform(I, frame), Xi; float ds; invertXform(X, Xi, ds);
        float3 oo = xfPoint(Xi, o), dd = xfVector(Xi, d); const Mesh& m = Z.meshes[I.mesh];
        for (const int3& f : m.f) hitTri(oo, dd, m.v[f.x], m.v[f.y], m.v[f.z], t);
    }
    return t;
}

int main(int argc, char** argv) {
    ViewerOpts VO; std::vector<char*> rest;
    if (!splitArgs(argc, argv, VO, rest)) return 1;
    SceneSetup Z;
    if (!parseArgs((int)rest.size(), rest.data(), Z.A)) return 1;
    if (int rc = setupScene(Z, true)) return rc;
    Args& A = Z.A; const int W = A.w, Hh = A.h, N = W * Hh, K = Z.K;
    int targetSpp = VO.frames > 0 ? 0 : (VO.sppGiven ? A.spp : 0);

    // ---- window + GL ----
    if (!glfwInit()) { fprintf(stderr, "glfwInit failed (no display?)\n"); return 1; }
    const int panelW = VO.hideUi ? 0 : 360;
    float fit = std::min(1.f, std::min(1600.f / W, 1000.f / Hh));
    int winW = VO.winW > 0 ? VO.winW : panelW + std::max(640, (int)(W * fit)), winH = VO.winH > 0 ? VO.winH : std::max(VO.hideUi ? 1 : 720, (int)(Hh * fit));
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    std::string title = "cwp viewer - " + (A.synthetic ? std::string("synthetic") : !A.scene.empty() ? A.scene : A.mesh);
    GLFWwindow* win = glfwCreateWindow(winW, winH, title.c_str(), nullptr, nullptr);
    if (!win) { fprintf(stderr, "could not create an OpenGL 3.3 core window\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win); glfwSwapInterval(0);
    glewExperimental = GL_TRUE; GLenum ge = glewInit();
    if (ge != GLEW_OK) { fprintf(stderr, "glewInit failed: %s\n", (const char*)glewGetErrorString(ge)); return 1; }
    while (glGetError() != GL_NO_ERROR) {}
    fprintf(stderr, "OpenGL %s, renderer %s\n", (const char*)glGetString(GL_VERSION), (const char*)glGetString(GL_RENDERER));
    IMGUI_CHECKVERSION(); ImGui::CreateContext(); ImGuiIO& io = ImGui::GetIO(); io.IniFilename = nullptr;
    ImGui::StyleColorsDark(); ImGui_ImplGlfw_InitForOpenGL(win, true); ImGui_ImplOpenGL3_Init("#version 330");

    // ---- renderer ----
    WavefrontRenderer R; R.init(Z.H, Z.S, Z.O);
    std::vector<DevInstance> di; std::vector<float4> emT; std::vector<float> emC, mlp;
    int animFrame = A.frameStart, animShown = -1; bool animPlay = false;
    auto applyFrame = [&]() { if (Z.inst && animShown != animFrame) { Z.frameData(animFrame, di, emT, emC, mlp); R.setInstances(di, emT, emC, mlp); animShown = animFrame; } };
    applyFrame();

    // ---- display texture, PBO interop or host-copy fallback ----
    GLuint tex = 0; glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, Hh, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    GLuint pbo = 0; cudaGraphicsResource* res = nullptr; bool interop = VO.interop != "0"; std::string interopNote = "host copy (--interop 0)";
    if (interop) {
        glGenBuffers(1, &pbo); glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo); glBufferData(GL_PIXEL_UNPACK_BUFFER, (GLsizeiptr)N * 4, nullptr, GL_STREAM_DRAW); glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        cudaError_t e = cudaGraphicsGLRegisterBuffer(&res, pbo, cudaGraphicsRegisterFlagsWriteDiscard);
        if (e != cudaSuccess) {
            cudaGetLastError(); glDeleteBuffers(1, &pbo); pbo = 0; res = nullptr; interop = false;
            interopNote = std::string("host copy (CUDA-GL interop unavailable: ") + cudaGetErrorString(e) + ")";
            if (VO.interop == "1") { fprintf(stderr, "--interop 1: %s\n", interopNote.c_str()); return 1; }
        } else interopNote = "CUDA-GL interop (PBO)";
    }
    fprintf(stderr, "display path: %s\n", interopNote.c_str());
    uchar4* dDisp = nullptr; std::vector<uchar4> hDisp;
    if (!interop) { CKV(cudaMalloc(&dDisp, (size_t)N * 4)); hDisp.resize(N); }
    float exposure = VO.exposure; int tonemap = VO.tonemap;
    int spp = 0;   // samples accumulated
    auto updateDisplay = [&]() {
        if (interop) {
            uchar4* p = nullptr; size_t sz = 0;
            CKV(cudaGraphicsMapResources(1, &res, 0)); CKV(cudaGraphicsResourceGetMappedPointer((void**)&p, &sz, res));
            launchDisplay(R.accumDevice(), p, N, spp, exposure, tonemap); CKV(cudaGetLastError());
            CKV(cudaGraphicsUnmapResources(1, &res, 0));
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo); glBindTexture(GL_TEXTURE_2D, tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, Hh, GL_RGBA, GL_UNSIGNED_BYTE, nullptr); glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        } else {
            launchDisplay(R.accumDevice(), dDisp, N, spp, exposure, tonemap); CKV(cudaGetLastError());
            CKV(cudaMemcpy(hDisp.data(), dDisp, (size_t)N * 4, cudaMemcpyDeviceToHost));
            glBindTexture(GL_TEXTURE_2D, tex); glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, Hh, GL_RGBA, GL_UNSIGNED_BYTE, hDisp.data());
        }
    };
    // the displayed render image, read back from the texture
    auto saveRenderPNG = [&](const std::string& path) {
        std::vector<uint8_t> rgba((size_t)N * 4), rgb((size_t)N * 3);
        glBindTexture(GL_TEXTURE_2D, tex); glPixelStorei(GL_PACK_ALIGNMENT, 1); glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
        for (int i = 0; i < N; i++) for (int c = 0; c < 3; c++) rgb[3 * i + c] = rgba[4 * i + c];
        bool ok = writePNG(path, W, Hh, rgb.data()); fprintf(stderr, "%s %s (%dx%d, %d spp, displayed image)\n", ok ? "wrote" : "FAILED to write", path.c_str(), W, Hh, spp); return ok;
    };
    auto saveWindowPNG = [&](const std::string& path) {
        int fw, fh; glfwGetFramebufferSize(win, &fw, &fh); std::vector<uint8_t> buf((size_t)fw * fh * 3), flip(buf.size());
        glPixelStorei(GL_PACK_ALIGNMENT, 1); glReadBuffer(GL_BACK); glReadPixels(0, 0, fw, fh, GL_RGB, GL_UNSIGNED_BYTE, buf.data());
        for (int y = 0; y < fh; y++) memcpy(&flip[(size_t)y * fw * 3], &buf[(size_t)(fh - 1 - y) * fw * 3], (size_t)fw * 3);
        bool ok = writePNG(path, fw, fh, flip.data()); fprintf(stderr, "%s %s (window %dx%d)\n", ok ? "wrote" : "FAILED to write", path.c_str(), fw, fh);
    };
    auto savePFM = [&](const std::string& path) {
        std::vector<float4> acc; R.readAccum(acc); std::vector<float> lin((size_t)N * 3);
        for (int i = 0; i < N; i++) { lin[3 * i] = acc[i].x / spp; lin[3 * i + 1] = acc[i].y / spp; lin[3 * i + 2] = acc[i].z / spp; }
        writePFM(path, W, Hh, lin.data()); fprintf(stderr, "wrote %s\n", path.c_str());
    };

    // ---- interactive state (applied to the renderer only when changed, so an untouched viewer renders the CLI setup exactly) ----
    Params H = Z.H; std::vector<Material> mats = Z.mats;
    float3 f0 = -H.cam.az; float yaw = atan2f(f0.y, f0.x), pitch = asinf(std::max(-1.f, std::min(1.f, f0.z)));
    float hfov = 2.f * atanf(0.5f * W / H.cam.fx) * 180.f / kPi, fyRatio = H.cam.fy / H.cam.fx;
    float speed = std::max(Z.extent, 1e-3f) * 0.25f, lensR = H.cam.lensR, focusD = H.cam.focusD;
    if (lensR <= 0.f) { AABB b; for (auto& v : Z.M.v) b.grow(v); float d = dot((b.lo + b.hi) * 0.5f - H.cam.pos, f0); focusD = d > 0.f ? d : Z.extent; }
    float sunInt = maxc(H.sunE); float3 sunCol = sunInt > 0.f ? H.sunE * (1.f / sunInt) : mk3(1, 1, 1);
    float3 st0 = normalize(H.sunTo); float sunAz = atan2f(st0.y, st0.x) * 180.f / kPi, sunEl = asinf(std::max(-1.f, std::min(1.f, st0.z))) * 180.f / kPi;
    const bool hasEnv = Z.full && Z.S.env.w > 0; float envStrength = A.light.envStrength, envRot = A.light.envRotDeg; EnvMapHost env1; std::vector<float4> envTex;
    bool camDirty = false, lensDirty = false, lightDirty = false, envDirty = false, envTexDirty = false, matDirty = false, reset = false, paused = false;
    int curMat = 0; FrameStats last; double lastPassMs = 0; int frameCount = 0, shot = 0; bool automationDone = false;
    auto t0 = std::chrono::steady_clock::now();

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        auto t1 = std::chrono::steady_clock::now(); float dt = std::chrono::duration<float>(t1 - t0).count(); t0 = t1; dt = std::min(dt, 0.1f);
        ImGui_ImplOpenGL3_NewFrame(); ImGui_ImplGlfw_NewFrame(); ImGui::NewFrame();
        int fbw, fbh; glfwGetFramebufferSize(win, &fbw, &fbh);
        // image rectangle (letterboxed in the area right of the panel)
        float ax0 = (float)panelW, aw = std::max(1.f, (float)fbw - ax0), ah = std::max(1.f, (float)fbh);
        float sc = std::min(aw / W, ah / Hh); float iw = W * sc, ih = Hh * sc, ix = ax0 + 0.5f * (aw - iw), iy = 0.5f * (ah - ih);

        // ---- camera input ----
        if (VO.frames <= 0) {
            if (!io.WantCaptureMouse) {
                if (ImGui::IsMouseDown(ImGuiMouseButton_Right) && (io.MouseDelta.x != 0.f || io.MouseDelta.y != 0.f)) {
                    yaw -= io.MouseDelta.x * 0.15f * kPi / 180.f; pitch -= io.MouseDelta.y * 0.15f * kPi / 180.f;
                    pitch = std::max(-1.55f, std::min(1.55f, pitch)); camDirty = true;
                }
                if (io.MouseWheel != 0.f) speed *= powf(1.2f, io.MouseWheel);
                bool pick = ImGui::IsMouseClicked(ImGuiMouseButton_Middle) || (io.KeyCtrl && ImGui::IsMouseClicked(ImGuiMouseButton_Left));
                if (pick && io.MousePos.x >= ix && io.MousePos.x < ix + iw && io.MousePos.y >= iy && io.MousePos.y < iy + ih) {
                    const Cam& c = H.cam; float px = (io.MousePos.x - ix) / sc, py = (io.MousePos.y - iy) / sc;
                    float3 d = c.ax * ((px - c.px) / c.fx) + c.ay * (-(py - c.py) / c.fy) - c.az;   // camera-space z = -1: t = depth
                    float t = castRay(Z, animFrame, c.pos, d);
                    if (std::isfinite(t)) { focusD = t; lensDirty = true; fprintf(stderr, "focus distance %.4f\n", t); }
                }
            }
            if (!io.WantCaptureKeyboard) {
                float v = speed * dt * (io.KeyShift ? 5.f : 1.f) * (io.KeyAlt ? 0.2f : 1.f);
                float3 fwd = -H.cam.az, rt = H.cam.ax, mv = mk3(0, 0, 0);
                if (ImGui::IsKeyDown(ImGuiKey_W)) mv += fwd * v; if (ImGui::IsKeyDown(ImGuiKey_S)) mv += fwd * -v;
                if (ImGui::IsKeyDown(ImGuiKey_D)) mv += rt * v; if (ImGui::IsKeyDown(ImGuiKey_A)) mv += rt * -v;
                if (ImGui::IsKeyDown(ImGuiKey_E)) mv += mk3(0, 0, v); if (ImGui::IsKeyDown(ImGuiKey_Q)) mv += mk3(0, 0, -v);
                if (!io.KeyCtrl && (mv.x != 0.f || mv.y != 0.f || mv.z != 0.f)) { H.cam.pos += mv; lensDirty = true; }
                if (ImGui::IsKeyPressed(ImGuiKey_P, false)) { char b[64]; snprintf(b, sizeof b, "cwp_screenshot_%03d.png", shot++); saveRenderPNG(b); }
                if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_O, false)) {
                    const char* pat[] = { "*.obj", "*.ply", "*.gltf", "*.glb", "*.fbx", "*.json" };
                    if (const char* f = tinyfd_openFileDialog("Open model or scene", "", 6, pat, "models / scene files", 0)) reopenWith(argc, argv, f);
                }
            }
        }

        // ---- UI ----
        if (!VO.hideUi) {
            ImGui::SetNextWindowPos(ImVec2(0, 0)); ImGui::SetNextWindowSize(ImVec2((float)panelW, (float)fbh));
            ImGui::Begin("cwp viewer", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
            ImGui::PushItemWidth(190.f);
            ImGui::Text("%s", R.gpuName.c_str());
            ImGui::Text("%dx%d  %d spp  %.1f ms/pass  %.0f Mrays/s", W, Hh, spp, lastPassMs, (last.rays + last.shadowRays) / std::max(1e-9, last.frameMs * 1e3));
            ImGui::TextWrapped("display: %s", interopNote.c_str());
            if (ImGui::CollapsingHeader("Render", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("integrator %s, batch %d spp/pass", Z.full ? "full (materials, MIS)" : "legacy (Lambert + sun + sky)", K);
                ImGui::InputInt("target spp (0 = inf)", &targetSpp, K, 64 * K); targetSpp = std::max(0, targetSpp);
                ImGui::SliderInt("passes / frame", &VO.passes, 1, 64);
                if (ImGui::SliderInt("max bounces", &H.maxB, 0, A.bounces)) lensDirty = true;
                ImGui::Checkbox("pause", &paused); ImGui::SameLine(); if (ImGui::Button("restart")) reset = true;
                ImGui::SliderFloat("exposure (EV)", &exposure, -8.f, 8.f, "%.2f");
                const char* tms[] = { "clip + sRGB (= batch PNG)", "Reinhard", "ACES fit" }; ImGui::Combo("tone map", &tonemap, tms, 3);
                if (ImGui::Button("Save PNG...")) {
                    const char* pat[] = { "*.png" };
                    if (const char* f = tinyfd_saveFileDialog("Save screenshot", "screenshot.png", 1, pat, "PNG image")) saveRenderPNG(f);
                }
                ImGui::SameLine(); ImGui::TextDisabled("(P: quick save)");
            }
            if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("pos %.3f %.3f %.3f", H.cam.pos.x, H.cam.pos.y, H.cam.pos.z);
                ImGui::SliderFloat("speed", &speed, std::max(Z.extent, 1e-3f) * 1e-3f, std::max(Z.extent, 1e-3f) * 4.f, "%.3f", ImGuiSliderFlags_Logarithmic);
                if (ImGui::SliderFloat("hfov (deg)", &hfov, 5.f, 150.f, "%.1f")) lensDirty = true;
                if (ImGui::SliderFloat("aperture radius", &lensR, 0.f, std::max(Z.extent, 1e-3f) * 0.05f, "%.4f", ImGuiSliderFlags_Logarithmic)) lensDirty = true;
                if (ImGui::SliderFloat("focus distance", &focusD, std::max(Z.extent, 1e-3f) * 1e-3f, std::max(Z.extent, 1e-3f) * 8.f, "%.3f", ImGuiSliderFlags_Logarithmic)) lensDirty = true;
                ImGui::TextDisabled("RMB look, WASD/QE move, wheel speed,\nMMB or Ctrl+LMB: focus at cursor");
            }
            if (ImGui::CollapsingHeader("Lighting")) {
                bool ch = false;
                ch |= ImGui::SliderFloat("sun azimuth", &sunAz, -180.f, 180.f, "%.1f");
                ch |= ImGui::SliderFloat("sun elevation", &sunEl, -10.f, 90.f, "%.1f");
                ch |= ImGui::DragFloat("sun irradiance", &sunInt, 0.05f, 0.f, 1000.f, "%.3f");
                ch |= ImGui::ColorEdit3("sun colour", &sunCol.x);
                if (!hasEnv) ch |= ImGui::ColorEdit3("sky radiance", &H.sky.x, ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);
                if (ch) lightDirty = true;
                if (hasEnv) {
                    if (ImGui::DragFloat("env strength", &envStrength, 0.01f, 0.f, 100.f, "%.3f")) { envDirty = envTexDirty = true; }
                    if (ImGui::SliderFloat("env rotation", &envRot, -180.f, 180.f, "%.1f")) envDirty = true;
                }
            }
            if (ImGui::CollapsingHeader("Materials") && !mats.empty()) {
                auto name = [&](int i) { return i < (int)Z.M.matName.size() && !Z.M.matName[i].empty() ? Z.M.matName[i] : "material " + std::to_string(i); };
                curMat = std::min(curMat, (int)mats.size() - 1);
                if (ImGui::BeginCombo("material", name(curMat).c_str())) {
                    for (int i = 0; i < (int)mats.size(); i++) if (ImGui::Selectable(name(i).c_str(), i == curMat)) curMat = i;
                    ImGui::EndCombo();
                }
                Material& m = mats[curMat]; bool ch = false;
                ch |= ImGui::ColorEdit3("base colour", &m.baseColor.x);
                if (Z.full) {
                    ch |= ImGui::SliderFloat("metallic", &m.metallic, 0.f, 1.f);
                    ch |= ImGui::SliderFloat("roughness", &m.roughness, 0.f, 1.f);
                    ch |= ImGui::SliderFloat("specular weight", &m.specularWeight, 0.f, 1.f);
                    ch |= ImGui::ColorEdit3("specular colour", &m.specularColor.x);
                    ch |= ImGui::SliderFloat("ior", &m.ior, 1.f, 3.f);
                    ch |= ImGui::SliderFloat("transmission", &m.transmission, 0.f, 1.f);
                    ImGui::Text("emission %.3f %.3f %.3f (fixed: light tables)", m.emission.x, m.emission.y, m.emission.z);
                } else ImGui::TextWrapped("legacy integrator: base colour only (start with --integrator full for all parameters)");
                if (ch) matDirty = true;
            }
            if (ImGui::CollapsingHeader("Scene")) {
                if (ImGui::Button("Open model / scene... (Ctrl+O)")) {
                    const char* pat[] = { "*.obj", "*.ply", "*.gltf", "*.glb", "*.fbx", "*.json" };
                    if (const char* f = tinyfd_openFileDialog("Open model or scene", "", 6, pat, "models / scene files", 0)) reopenWith(argc, argv, f);
                }
                ImGui::Text("%zu triangles, %zu materials%s", Z.M.f.size(), mats.size(), Z.inst ? ", instanced" : "");
                if (Z.inst && Z.SD.frames > 1) { ImGui::SliderInt("frame", &animFrame, 0, Z.SD.frames - 1); ImGui::SameLine(); ImGui::Checkbox("play", &animPlay); }
            }
            ImGui::PopItemWidth(); ImGui::End();
        }
        if (animPlay && Z.inst && Z.SD.frames > 1) animFrame = (animFrame + 1) % Z.SD.frames;

        // ---- apply changes -> restart accumulation ----
        if (camDirty) {
            float3 f = mk3(cosf(pitch) * cosf(yaw), cosf(pitch) * sinf(yaw), sinf(pitch)), r = normalize(cross(f, mk3(0, 0, 1))), u = cross(r, f);
            H.cam.ax = r; H.cam.ay = u; H.cam.az = -f; lensDirty = true; camDirty = false;
        }
        if (lensDirty) { H.cam.fx = focalFromHfov(W, hfov); H.cam.fy = H.cam.fx * fyRatio; H.cam.lensR = lensR; H.cam.focusD = focusD; }
        if (lightDirty) {
            float a = sunAz * kPi / 180.f, e = sunEl * kPi / 180.f; H.sunTo = mk3(cosf(e) * cosf(a), cosf(e) * sinf(a), sinf(e)); H.sunE = sunCol * sunInt;
            envDirty = true;   // sun on/off changes the light-type selection probabilities
        }
        if (lensDirty || lightDirty) { R.setParams(H); reset = true; lensDirty = lightDirty = false; }
        if (envDirty) {
            if (envTexDirty) {
                if (env1.w == 0) buildEnvMap(A.light.envMap, 1.f, env1);
                envTex.resize(env1.texels.size()); for (size_t i = 0; i < envTex.size(); i++) { float4 t = env1.texels[i]; envTex[i] = make_float4(t.x * envStrength, t.y * envStrength, t.z * envStrength, t.w); }
            }
            R.setEnvironment(envTexDirty ? &envTex : nullptr, envRot, luminance(H.sunE) > 0.f); reset = true; envDirty = envTexDirty = false;
        }
        if (matDirty) { R.setMaterials(mats); reset = true; matDirty = false; }
        if (Z.inst && animShown != animFrame) { applyFrame(); reset = true; }
        if (reset) { spp = 0; reset = false; }

        // ---- render passes ----
        bool rendered = false;
        bool want = VO.frames > 0 ? (frameCount < VO.frames) : (!paused && (targetSpp == 0 || spp < targetSpp));
        if (want) {
            auto r0 = std::chrono::steady_clock::now();
            for (int p = 0; p < VO.passes && (VO.frames > 0 || targetSpp == 0 || spp < targetSpp); p++) { R.render(K, last, spp, spp == 0); spp += K; }
            lastPassMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - r0).count() / VO.passes;
            frameCount++; rendered = true;
        }
        static float lastExp = exposure; static int lastTm = tonemap;
        if (rendered || lastExp != exposure || lastTm != tonemap) { if (spp > 0) updateDisplay(); lastExp = exposure; lastTm = tonemap; }
        bool autoNow = VO.frames > 0 && frameCount == VO.frames && !automationDone;
        if (autoNow) {
            if (!VO.screenshot.empty()) saveRenderPNG(VO.screenshot);
            if (!VO.screenshotPfm.empty()) savePFM(VO.screenshotPfm);
            fprintf(stderr, "automation: %d frames x %d passes x %d spp = %d spp\n", VO.frames, VO.passes, K, spp);
        }
        bool nextRound = autoNow && VO.testUpdates == 1;

        // ---- draw ----
        if (spp > 0) ImGui::GetBackgroundDrawList()->AddImage((ImTextureID)(intptr_t)tex, ImVec2(ix, iy), ImVec2(ix + iw, iy + ih));
        ImGui::Render();
        glViewport(0, 0, fbw, fbh); glClearColor(0.08f, 0.08f, 0.09f, 1.f); glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        if (autoNow) { if (!VO.screenshotWindow.empty()) saveWindowPNG(VO.screenshotWindow); automationDone = true; }
        if (nextRound) {   // --test_updates: one more round through every runtime-update path
            auto suf = [](std::string& f) { if (f.empty()) return; size_t d = f.find_last_of('.'); f = f.substr(0, d) + "_upd" + f.substr(d); };
            suf(VO.screenshot); suf(VO.screenshotPfm); suf(VO.screenshotWindow);
            yaw += 0.08f; camDirty = true; lensR = std::max(Z.extent, 1e-3f) * 0.01f; hfov *= 0.9f;
            sunAz += 40.f; sunEl = std::max(sunEl, 30.f); lightDirty = true;
            if (hasEnv) { envStrength *= 2.f; envRot += 30.f; envDirty = envTexDirty = true; }
            if (!mats.empty()) { mats[0].baseColor = mk3(0.9f, 0.2f, 0.1f); mats[0].roughness = 0.2f; mats[0].specularWeight = 1.f; matDirty = true; }
            frameCount = 0; automationDone = false; VO.testUpdates = 2;
        }
        glfwSwapBuffers(win);
        char tb[256]; snprintf(tb, sizeof tb, "%s - %d spp", title.c_str(), spp); glfwSetWindowTitle(win, tb);
        if (automationDone && VO.exitAfter) break;
    }
    if (res) cudaGraphicsUnregisterResource(res);
    ImGui_ImplOpenGL3_Shutdown(); ImGui_ImplGlfw_Shutdown(); ImGui::DestroyContext();
    glfwDestroyWindow(win); glfwTerminate();
    return 0;
}
