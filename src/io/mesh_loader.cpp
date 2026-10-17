// io/mesh_loader.cpp - OBJ / PLY loaders (written from scratch)
#include "io/mesh_loader.h"
#include <map>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>

static std::string dirOf(const std::string& p) {
    size_t k = p.find_last_of("/\\");
    return k == std::string::npos ? std::string(".") : p.substr(0, k);
}

static void loadMTL(const std::string& path, Mesh& m, std::map<std::string, int>& mats) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) { fprintf(stderr, "warn: mtl %s not found\n", path.c_str()); return; }
    char line[1024]; int cur = -1;
    while (fgets(line, sizeof line, fp)) {
        if (strncmp(line, "newmtl", 6) == 0) {
            char name[512]; if (sscanf(line + 6, "%511s", name) == 1) { cur = (int)m.kd.size(); mats[name] = cur; m.kd.push_back(mk3(0.8f, 0.8f, 0.8f)); m.matName.resize(m.kd.size()); m.matName[cur] = name; }
        } else if (cur >= 0 && strncmp(line, "Kd", 2) == 0) {
            float r, g, b; if (sscanf(line + 2, "%f %f %f", &r, &g, &b) == 3) m.kd[cur] = mk3(r, g, b);
        }
    }
    fclose(fp);
}

bool loadOBJ(const std::string& path, Mesh& m) {
    FILE* fp = fopen(path.c_str(), "rb"); if (!fp) return false;
    std::vector<float3> vn; std::map<std::string, int> mats; int curMat = 0;
    if (m.kd.empty()) m.kd.push_back(mk3(0.8f, 0.8f, 0.8f));
    std::vector<char> line(1 << 16);
    std::vector<int> vi, ni;
    while (fgets(line.data(), (int)line.size(), fp)) {
        char* s = line.data();
        if (s[0] == 'v' && s[1] == ' ') { float x, y, z; if (sscanf(s + 2, "%f %f %f", &x, &y, &z) == 3) m.v.push_back(mk3(x, y, z)); }
        else if (s[0] == 'v' && s[1] == 'n') { float x, y, z; if (sscanf(s + 2, "%f %f %f", &x, &y, &z) == 3) vn.push_back(mk3(x, y, z)); }
        else if (s[0] == 'f' && (s[1] == ' ' || s[1] == '\t')) {
            vi.clear(); ni.clear();
            char* p = s + 1;
            while (*p) {
                while (*p == ' ' || *p == '\t') p++;
                if (*p == '\n' || *p == '\r' || *p == 0) break;
                int a = atoi(p); int b = 0; bool hasN = false;
                char* q = p; while (*q && *q != ' ' && *q != '\t' && *q != '\n' && *q != '\r') q++;
                char* sl = (char*)memchr(p, '/', q - p);
                if (sl) { char* sl2 = (char*)memchr(sl + 1, '/', q - sl - 1); if (sl2 && sl2 + 1 < q) { b = atoi(sl2 + 1); hasN = true; } }
                int vidx = a > 0 ? a - 1 : (int)m.v.size() + a;
                int nidx = hasN ? (b > 0 ? b - 1 : (int)vn.size() + b) : -1;
                vi.push_back(vidx); ni.push_back(nidx);
                p = q;
            }
            for (size_t k = 1; k + 1 < vi.size(); k++) {
                m.f.push_back(make_int3(vi[0], vi[k], vi[k + 1])); m.fm.push_back(curMat);
            }
            for (size_t k = 0; k < vi.size(); k++) if (ni[k] >= 0 && ni[k] < (int)vn.size() && vi[k] >= 0 && vi[k] < (int)m.v.size()) {
                if (m.n.size() != m.v.size()) m.n.resize(m.v.size(), mk3(0, 0, 0));
                m.n[vi[k]] = vn[ni[k]];
            }
        }
        else if (strncmp(s, "mtllib", 6) == 0) { char name[1024]; if (sscanf(s + 6, "%1023s", name) == 1) loadMTL(dirOf(path) + "/" + name, m, mats); }
        else if (strncmp(s, "usemtl", 6) == 0) { char name[1024]; curMat = 0; if (sscanf(s + 6, "%1023s", name) == 1) { auto it = mats.find(name); if (it != mats.end()) curMat = it->second; } }
    }
    fclose(fp);
    if (!m.n.empty()) { // vertices never assigned a normal: fall back to zero -> treated as geometric later
        bool any = false; for (auto& x : m.n) if (length(x) > 0) { any = true; break; }
        if (!any) m.n.clear();
    }
    return !m.f.empty();
}

// ---- PLY (ascii / binary_little_endian) ----
struct PlyProp { std::string name, type, ltype, itype; bool list = false; };
struct PlyElem { std::string name; long count = 0; std::vector<PlyProp> props; };
static int plySize(const std::string& t) {
    if (t == "char" || t == "uchar" || t == "int8" || t == "uint8") return 1;
    if (t == "short" || t == "ushort" || t == "int16" || t == "uint16") return 2;
    if (t == "int" || t == "uint" || t == "int32" || t == "uint32" || t == "float" || t == "float32") return 4;
    if (t == "double" || t == "float64") return 8;
    return 4;
}
static double plyReadBin(FILE* fp, const std::string& t) {
    unsigned char b[8]; int n = plySize(t); if (fread(b, 1, n, fp) != (size_t)n) return 0;
    if (t == "char" || t == "int8") return (double)(int8_t)b[0];
    if (t == "uchar" || t == "uint8") return (double)b[0];
    if (t == "short" || t == "int16") { int16_t x; memcpy(&x, b, 2); return x; }
    if (t == "ushort" || t == "uint16") { uint16_t x; memcpy(&x, b, 2); return x; }
    if (t == "int" || t == "int32") { int32_t x; memcpy(&x, b, 4); return x; }
    if (t == "uint" || t == "uint32") { uint32_t x; memcpy(&x, b, 4); return x; }
    if (t == "float" || t == "float32") { float x; memcpy(&x, b, 4); return x; }
    if (t == "double" || t == "float64") { double x; memcpy(&x, b, 8); return x; }
    return 0;
}
static double plyReadAscii(FILE* fp) { double x = 0; if (fscanf(fp, "%lf", &x) != 1) return 0; return x; }

bool loadPLY(const std::string& path, Mesh& m) {
    FILE* fp = fopen(path.c_str(), "rb"); if (!fp) return false;
    char line[1024]; bool ascii = false, ok = false; std::vector<PlyElem> elems;
    if (!fgets(line, sizeof line, fp) || strncmp(line, "ply", 3) != 0) { fclose(fp); return false; }
    while (fgets(line, sizeof line, fp)) {
        if (strncmp(line, "format", 6) == 0) { ascii = strstr(line, "ascii") != nullptr; if (strstr(line, "big_endian")) { fprintf(stderr, "PLY big endian unsupported\n"); fclose(fp); return false; } }
        else if (strncmp(line, "element", 7) == 0) { PlyElem e; char name[256]; long c; if (sscanf(line + 7, "%255s %ld", name, &c) == 2) { e.name = name; e.count = c; elems.push_back(e); } }
        else if (strncmp(line, "property", 8) == 0 && !elems.empty()) {
            PlyProp p; char a[64], b[64], c[64];
            if (sscanf(line + 8, "%63s %63s %63s", a, b, c) >= 2) {
                if (strcmp(a, "list") == 0) { p.list = true; p.ltype = b; p.itype = c; char d[64]; if (sscanf(line + 8, "%*s %*s %*s %63s", d) == 1) p.name = d; }
                else { p.type = a; p.name = b; }
                elems.back().props.push_back(p);
            }
        }
        else if (strncmp(line, "end_header", 10) == 0) { ok = true; break; }
    }
    if (!ok) { fclose(fp); return false; }
    if (m.kd.empty()) m.kd.push_back(mk3(0.8f, 0.8f, 0.8f));
    auto rd = [&](const std::string& t) { return ascii ? plyReadAscii(fp) : plyReadBin(fp, t); };
    for (auto& e : elems) {
        if (e.name == "vertex") {
            int ix = -1, iy = -1, iz = -1, inx = -1, iny = -1, inz = -1;
            for (size_t k = 0; k < e.props.size(); k++) { const std::string& nm = e.props[k].name; if (nm == "x") ix = k; else if (nm == "y") iy = k; else if (nm == "z") iz = k; else if (nm == "nx") inx = k; else if (nm == "ny") iny = k; else if (nm == "nz") inz = k; }
            bool hasN = inx >= 0 && iny >= 0 && inz >= 0;
            m.v.reserve(m.v.size() + e.count); if (hasN) m.n.reserve(m.v.size() + e.count);
            std::vector<double> vals(e.props.size());
            for (long i = 0; i < e.count; i++) {
                for (size_t k = 0; k < e.props.size(); k++) {
                    if (e.props[k].list) { int cnt = (int)rd(e.props[k].ltype); for (int j = 0; j < cnt; j++) rd(e.props[k].itype); vals[k] = 0; }
                    else vals[k] = rd(e.props[k].type);
                }
                m.v.push_back(mk3((float)vals[ix], (float)vals[iy], (float)vals[iz]));
                if (hasN) m.n.push_back(mk3((float)vals[inx], (float)vals[iny], (float)vals[inz]));
            }
        } else if (e.name == "face") {
            m.f.reserve(m.f.size() + e.count);
            std::vector<int> idx; int maxMat = 0;
            for (long i = 0; i < e.count; i++) {
                int mat = 0;
                for (size_t k = 0; k < e.props.size(); k++) {
                    const PlyProp& p = e.props[k];
                    if (p.list) {
                        int cnt = (int)rd(p.ltype);
                        bool isIdx = p.name == "vertex_indices" || p.name == "vertex_index";
                        if (isIdx) { idx.resize(cnt); for (int j = 0; j < cnt; j++) idx[j] = (int)rd(p.itype); }
                        else for (int j = 0; j < cnt; j++) rd(p.itype);
                    } else { double x = rd(p.type); if (p.name == "material_index") mat = (int)x; }
                }
                if (mat > maxMat) maxMat = mat;
                for (size_t k = 1; k + 1 < idx.size(); k++) { m.f.push_back(make_int3(idx[0], idx[k], idx[k + 1])); m.fm.push_back(mat); }
            }
            while ((int)m.kd.size() <= maxMat) m.kd.push_back(mk3(0.8f, 0.8f, 0.8f));
        } else {
            for (long i = 0; i < e.count; i++) for (auto& p : e.props) {
                if (p.list) { int cnt = (int)rd(p.ltype); for (int j = 0; j < cnt; j++) rd(p.itype); } else rd(p.type);
            }
        }
    }
    fclose(fp);
    if (!m.n.empty() && m.n.size() != m.v.size()) m.n.clear();
    return !m.f.empty();
}

bool loadMesh(const std::string& path, Mesh& m) {
    if (path.size() > 4 && (path.substr(path.size() - 4) == ".ply" || path.substr(path.size() - 4) == ".PLY")) return loadPLY(path, m);
    return loadOBJ(path, m);
}
