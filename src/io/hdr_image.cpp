// io/hdr_image.cpp (written from scratch)
#include "io/hdr_image.h"
#include <cmath>
#include <cstdio>
#include <cstring>

static void rgbeToFloat(const unsigned char* e, float* out) {
    if (e[3] == 0) { out[0] = out[1] = out[2] = 0.f; return; }
    float f = ldexpf(1.f, (int)e[3] - (128 + 8));   // mantissa * 2^(exp - 136), the decoding Blender uses (no +0.5 offset)
    out[0] = e[0] * f; out[1] = e[1] * f; out[2] = e[2] * f;
}

bool loadHDR(const std::string& path, int& w, int& h, std::vector<float>& rgb) {
    FILE* fp = fopen(path.c_str(), "rb"); if (!fp) { fprintf(stderr, "hdr: cannot open %s\n", path.c_str()); return false; }
    char line[512];
    if (!fgets(line, sizeof line, fp) || (strncmp(line, "#?RADIANCE", 10) != 0 && strncmp(line, "#?RGBE", 6) != 0)) { fprintf(stderr, "hdr: %s is not a Radiance file\n", path.c_str()); fclose(fp); return false; }
    bool fmtOk = true;
    while (fgets(line, sizeof line, fp)) {
        if (line[0] == '\n' || line[0] == '\r') break;
        if (strncmp(line, "FORMAT=", 7) == 0 && strncmp(line + 7, "32-bit_rle_rgbe", 15) != 0) fmtOk = false;
    }
    if (!fmtOk) { fprintf(stderr, "hdr: %s has an unsupported FORMAT (only 32-bit_rle_rgbe)\n", path.c_str()); fclose(fp); return false; }
    if (!fgets(line, sizeof line, fp) || sscanf(line, "-Y %d +X %d", &h, &w) != 2 || w <= 0 || h <= 0) { fprintf(stderr, "hdr: %s: only '-Y H +X W' orientation is supported\n", path.c_str()); fclose(fp); return false; }
    rgb.assign((size_t)w * h * 3, 0.f);
    std::vector<unsigned char> scan((size_t)w * 4);
    for (int y = 0; y < h; y++) {
        unsigned char hd[4];
        if (fread(hd, 1, 4, fp) != 4) { fprintf(stderr, "hdr: %s truncated\n", path.c_str()); fclose(fp); return false; }
        if (w >= 8 && w < 32768 && hd[0] == 2 && hd[1] == 2 && ((hd[2] << 8) | hd[3]) == w) {
            for (int c = 0; c < 4; c++) {     // new RLE: each channel separately
                int x = 0;
                while (x < w) {
                    int cnt = fgetc(fp); if (cnt == EOF) { fprintf(stderr, "hdr: %s truncated\n", path.c_str()); fclose(fp); return false; }
                    if (cnt > 128) { cnt -= 128; int v = fgetc(fp); if (v == EOF || x + cnt > w) { fprintf(stderr, "hdr: %s corrupt run\n", path.c_str()); fclose(fp); return false; } for (int k = 0; k < cnt; k++) scan[4 * (x++) + c] = (unsigned char)v; }
                    else { if (cnt == 0 || x + cnt > w) { fprintf(stderr, "hdr: %s corrupt run\n", path.c_str()); fclose(fp); return false; } for (int k = 0; k < cnt; k++) { int v = fgetc(fp); if (v == EOF) { fclose(fp); return false; } scan[4 * (x++) + c] = (unsigned char)v; } }
                }
            }
        } else {                               // flat scanline (old-style RLE runs are not supported)
            memcpy(scan.data(), hd, 4);
            if (w > 1 && fread(scan.data() + 4, 1, (size_t)(w - 1) * 4, fp) != (size_t)(w - 1) * 4) { fprintf(stderr, "hdr: %s truncated\n", path.c_str()); fclose(fp); return false; }
        }
        for (int x = 0; x < w; x++) rgbeToFloat(&scan[4 * x], &rgb[((size_t)y * w + x) * 3]);
    }
    fclose(fp);
    return true;
}
