// io/assimp_loader.cpp - Assimp-based model loader (written from scratch against the Assimp public API)
#include "io/assimp_loader.h"
#include <cstdio>

#ifndef CWP_HAVE_ASSIMP
bool assimpAvailable() { return false; }
bool loadAssimp(const std::string& path, Mesh&) {
    fprintf(stderr, "%s: built without Assimp (-DCWP_WITH_ASSIMP=OFF); only the built-in OBJ/PLY loaders are available\n", path.c_str());
    return false;
}
#else
#include "scene/tangents.h"
#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <map>
#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#define STB_IMAGE_STATIC   // private copy: no symbol clash with any stb inside Assimp
#include "stb_image.h"

bool assimpAvailable() { return true; }

namespace {
std::string dirOf(const std::string& p) { size_t k = p.find_last_of("/\\"); return k == std::string::npos ? std::string(".") : p.substr(0, k); }
std::string lower(std::string s) { for (auto& c : s) c = (char)tolower((unsigned char)c); return s; }
std::string urlDecode(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '%' && i + 2 < s.size() && isxdigit((unsigned char)s[i + 1]) && isxdigit((unsigned char)s[i + 2])) { o += (char)strtol(s.substr(i + 1, 2).c_str(), nullptr, 16); i += 2; }
        else o += s[i];
    }
    return o;
}

struct TexCache {
    const aiScene* sc; std::string dir; Mesh* m; std::map<std::pair<std::string, bool>, int> ids;
    // texture id for (file, srgb), loaded on first use; -1 if it cannot be read
    int get(const std::string& ref, bool srgb) {
        auto key = std::make_pair(ref, srgb); auto it = ids.find(key); if (it != ids.end()) return it->second;
        TextureImage T; T.srgb = srgb; int c = 0; unsigned char* px = nullptr;
        if (const aiTexture* e = sc->GetEmbeddedTexture(ref.c_str())) {
            T.path = "<embedded " + ref + ">";
            if (e->mHeight == 0) px = stbi_load_from_memory((const stbi_uc*)e->pcData, (int)e->mWidth, &T.w, &T.h, &c, 4);
            else {   // raw BGRA texels
                T.w = (int)e->mWidth; T.h = (int)e->mHeight; T.rgba.resize((size_t)T.w * T.h * 4);
                for (size_t i = 0; i < (size_t)T.w * T.h; i++) { const aiTexel& t = e->pcData[i]; T.rgba[4 * i] = t.r; T.rgba[4 * i + 1] = t.g; T.rgba[4 * i + 2] = t.b; T.rgba[4 * i + 3] = t.a; }
            }
        } else {
            std::string r = urlDecode(ref); for (auto& ch : r) if (ch == '\\') ch = '/';
            T.path = (!r.empty() && r[0] == '/') ? r : dir + "/" + r;
            px = stbi_load(T.path.c_str(), &T.w, &T.h, &c, 4);
        }
        if (px) { T.rgba.assign(px, px + (size_t)T.w * T.h * 4); stbi_image_free(px); }
        int id = -1;
        if (!T.rgba.empty()) {
            id = (int)m->tex.size();
            fprintf(stderr, "texture %d: %s %dx%d %s\n", id, T.path.c_str(), T.w, T.h, srgb ? "sRGB" : "linear");
            m->tex.push_back(std::move(T));
        } else fprintf(stderr, "warn: texture %s unreadable (%s)\n", T.path.c_str(), stbi_failure_reason() ? stbi_failure_reason() : "?");
        ids[key] = id; return id;
    }
};

bool texPath(const aiMaterial* a, aiTextureType t, std::string& out) {
    aiString s; if (a->GetTextureCount(t) == 0 || a->GetTexture(t, 0, &s) != AI_SUCCESS || s.length == 0) return false;
    out = s.C_Str(); return true;
}
float lum3(float3 c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }

void convertMaterial(const aiMaterial* a, bool gltf, TexCache& tc, Material& M, MaterialTextures& X, std::string& name) {
    aiString nm; name = a->Get(AI_MATKEY_NAME, nm) == AI_SUCCESS ? nm.C_Str() : "";
    M = lambertMaterial(mk3(0.8f, 0.8f, 0.8f));
    aiColor4D c4; aiColor3D c3; float x;
    if (a->Get(AI_MATKEY_BASE_COLOR, c4) == AI_SUCCESS) M.baseColor = mk3(c4.r, c4.g, c4.b);
    else if (a->Get(AI_MATKEY_COLOR_DIFFUSE, c3) == AI_SUCCESS) M.baseColor = mk3(c3.r, c3.g, c3.b);
    bool hasMetal = a->Get(AI_MATKEY_METALLIC_FACTOR, x) == AI_SUCCESS; if (hasMetal) M.metallic = x;
    bool hasRough = a->Get(AI_MATKEY_ROUGHNESS_FACTOR, x) == AI_SUCCESS; if (hasRough) M.roughness = x;
    if (gltf) {
        M.specularWeight = 1.f;   // glTF dielectric: F0 = 0.04 at ior 1.5 (KHR_materials_specular is not supported)
        if (!hasMetal) M.metallic = 1.f;
        if (!hasRough) M.roughness = 1.f;   // glTF defaults
    } else {
        if (!hasRough) { float n = 0.f; M.roughness = (a->Get(AI_MATKEY_SHININESS, n) == AI_SUCCESS && n > 0.f) ? sqrtf(sqrtf(2.f / (n + 2.f))) : 0.5f; }
        if (a->Get(AI_MATKEY_COLOR_SPECULAR, c3) == AI_SUCCESS) M.specularWeight = fminf(fmaxf(lum3(mk3(c3.r, c3.g, c3.b)), 0.f), 1.f);
    }
    if (a->Get(AI_MATKEY_TRANSMISSION_FACTOR, x) == AI_SUCCESS) M.transmission = x;
    if (a->Get(AI_MATKEY_REFRACTI, x) == AI_SUCCESS && x >= 1.f) M.ior = x;
    float3 em = mk3(0, 0, 0); if (a->Get(AI_MATKEY_COLOR_EMISSIVE, c3) == AI_SUCCESS) em = mk3(c3.r, c3.g, c3.b);
    if (a->Get(AI_MATKEY_EMISSIVE_INTENSITY, x) == AI_SUCCESS) em = em * x;
    M.emission = em;
    std::string p;
    if (texPath(a, aiTextureType_BASE_COLOR, p) || texPath(a, aiTextureType_DIFFUSE, p)) X.baseColor = tc.get(p, true);
    if (texPath(a, aiTextureType_EMISSIVE, p)) X.emission = tc.get(p, true);
    if (texPath(a, aiTextureType_NORMALS, p)) {
        X.normal = tc.get(p, false);
        float s; if (a->Get("$tex.scale", aiTextureType_NORMALS, 0, s) == AI_SUCCESS) X.normalScale = s;
    }
    std::string pm, pr;
    bool hm = texPath(a, aiTextureType_METALNESS, pm), hr = texPath(a, aiTextureType_DIFFUSE_ROUGHNESS, pr);
    if (gltf && !hm && !hr && texPath(a, aiTextureType_UNKNOWN, p)) { pm = pr = p; hm = hr = true; }   // glTF metallicRoughness
    bool packed = gltf || (hm && hr && pm == pr);
    if (hm) { X.metallic = tc.get(pm, false); X.metallicCh = packed ? 2 : 0; }
    if (hr) { X.roughness = tc.get(pr, false); X.roughnessCh = packed ? 1 : 0; }
}

void walk(const aiScene* sc, const aiNode* nd, const aiMatrix4x4& parent, bool wantTan, Mesh& m, size_t& nMeshes) {
    aiMatrix4x4 T = parent * nd->mTransformation;
    aiMatrix3x3 R(T); aiMatrix3x3 Nm = R; Nm.Inverse().Transpose();
    float hand = R.Determinant() < 0.f ? -1.f : 1.f;   // a mirroring transform flips the bitangent
    for (unsigned k = 0; k < nd->mNumMeshes; k++) {
        const aiMesh* am = sc->mMeshes[nd->mMeshes[k]];
        if (!(am->mPrimitiveTypes & aiPrimitiveType_TRIANGLE)) continue;
        int base = (int)m.v.size(); nMeshes++;
        const unsigned nv = am->mNumVertices;
        std::vector<float3> lp(nv), ln(am->HasNormals() ? nv : 0); std::vector<float2> luv(nv, make_float2(0.f, 0.f));
        for (unsigned i = 0; i < nv; i++) {
            lp[i] = mk3(am->mVertices[i].x, am->mVertices[i].y, am->mVertices[i].z);
            if (am->HasNormals()) ln[i] = mk3(am->mNormals[i].x, am->mNormals[i].y, am->mNormals[i].z);
            if (am->HasTextureCoords(0)) luv[i] = make_float2(am->mTextureCoords[0][i].x, am->mTextureCoords[0][i].y);
        }
        std::vector<int3> lf;
        for (unsigned j = 0; j < am->mNumFaces; j++) { const aiFace& f = am->mFaces[j]; if (f.mNumIndices == 3) lf.push_back(make_int3((int)f.mIndices[0], (int)f.mIndices[1], (int)f.mIndices[2])); }
        std::vector<float4> lt;
        if (wantTan) {
            if (am->HasTangentsAndBitangents() && am->HasNormals()) {
                lt.resize(nv);
                for (unsigned i = 0; i < nv; i++) {
                    float3 t = mk3(am->mTangents[i].x, am->mTangents[i].y, am->mTangents[i].z), b = mk3(am->mBitangents[i].x, am->mBitangents[i].y, am->mBitangents[i].z);
                    lt[i] = make_float4(t.x, t.y, t.z, dot(cross(ln[i], t), b) < 0.f ? -1.f : 1.f);
                }
            } else computeTangents(lp, ln, luv, lf, lt);
        }
        for (unsigned i = 0; i < nv; i++) {
            aiVector3D p = T * am->mVertices[i]; m.v.push_back(mk3(p.x, p.y, p.z));
            if (!ln.empty()) { aiVector3D n = Nm * am->mNormals[i]; m.n.push_back(normalize(mk3(n.x, n.y, n.z))); } else m.n.push_back(mk3(0, 0, 0));
            m.uv.push_back(luv[i]);
            if (wantTan) {
                aiVector3D t(lt[i].x, lt[i].y, lt[i].z); t = R * t; float3 tt = normalize(mk3(t.x, t.y, t.z));
                m.tan.push_back(make_float4(tt.x, tt.y, tt.z, lt[i].w * hand));
            }
        }
        for (const int3& f : lf) { m.f.push_back(make_int3(base + f.x, base + f.y, base + f.z)); m.fm.push_back((int)am->mMaterialIndex); }
    }
    for (unsigned c = 0; c < nd->mNumChildren; c++) walk(sc, nd->mChildren[c], T, wantTan, m, nMeshes);
}
}  // namespace

bool loadAssimp(const std::string& path, Mesh& m) {
    Assimp::Importer imp;
    imp.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_POINT | aiPrimitiveType_LINE);
    const aiScene* sc = imp.ReadFile(path, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_GenNormals | aiProcess_SortByPType);
    if (!sc || !sc->mRootNode) { fprintf(stderr, "assimp: %s: %s\n", path.c_str(), imp.GetErrorString()); return false; }
    std::string ext = lower(path.substr(path.find_last_of('.') + 1)); bool gltf = ext == "gltf" || ext == "glb";
    m = Mesh();
    TexCache tc{ sc, dirOf(path), &m, {} };
    bool wantTan = false;
    for (unsigned i = 0; i < sc->mNumMaterials; i++) {
        Material M; MaterialTextures X; std::string name; convertMaterial(sc->mMaterials[i], gltf, tc, M, X, name);
        m.mat.push_back(M); m.mtex.push_back(X); m.matName.push_back(name); m.kd.push_back(M.baseColor);
        wantTan |= X.normal >= 0;
        fprintf(stderr, "material %u '%s': base (%.3f %.3f %.3f) metallic %.3f roughness %.3f ior %.3f transmission %.3f emission (%.3f %.3f %.3f) "
                        "tex base %d emis %d normal %d (scale %.2f) metal %d[ch%d] rough %d[ch%d]\n",
                i, name.c_str(), M.baseColor.x, M.baseColor.y, M.baseColor.z, M.metallic, M.roughness, M.ior, M.transmission, M.emission.x, M.emission.y, M.emission.z,
                X.baseColor, X.emission, X.normal, X.normalScale, X.metallic, X.metallicCh, X.roughness, X.roughnessCh);
    }
    if (m.mat.empty()) { m.mat.push_back(defaultMaterial()); m.mtex.emplace_back(); m.matName.push_back(""); m.kd.push_back(m.mat[0].baseColor); }
    size_t nMeshes = 0; aiMatrix4x4 I;
    walk(sc, sc->mRootNode, I, wantTan, m, nMeshes);
    bool anyN = false; for (auto& n : m.n) if (n.x != 0.f || n.y != 0.f || n.z != 0.f) { anyN = true; break; }
    if (!anyN) m.n.clear();
    fprintf(stderr, "assimp: %s: %zu meshes, %zu verts, %zu tris, %zu materials, %zu textures, tangents %s\n", path.c_str(), nMeshes, m.v.size(), m.f.size(), m.mat.size(), m.tex.size(), wantTan ? "yes" : "no");
    return !m.f.empty();
}
#endif
