// scene/camera.cpp (written from scratch)
#include "scene/camera.h"
#include <cmath>

void identityC2W(float c2w[16]) { for (int j = 0; j < 16; j++) c2w[j] = (j % 5 == 0) ? 1.f : 0.f; }

void lookAtC2W(float3 pos, float3 tgt, float3 up, float c2w[16]) {
    identityC2W(c2w);
    float3 f = normalize(tgt - pos), r = normalize(cross(f, up)), u = cross(r, f);
    float3 az = -f; float cols[3][3] = { { r.x, r.y, r.z }, { u.x, u.y, u.z }, { az.x, az.y, az.z } };
    for (int row = 0; row < 3; row++) for (int col = 0; col < 3; col++) c2w[row * 4 + col] = cols[col][row];
    c2w[3] = pos.x; c2w[7] = pos.y; c2w[11] = pos.z;
}

float focalFromHfov(int width, float hfovDeg) { return (width * 0.5f) / tanf(hfovDeg * 0.5f * 3.14159265f / 180.f); }

Cam makeCamera(const float c2w[16], float fx, float fy, int w, int h) {
    Cam c;
    c.pos = mk3(c2w[3], c2w[7], c2w[11]);
    c.ax = mk3(c2w[0], c2w[4], c2w[8]); c.ay = mk3(c2w[1], c2w[5], c2w[9]); c.az = mk3(c2w[2], c2w[6], c2w[10]);
    c.fx = fx; c.fy = fy; c.px = w * 0.5f; c.py = h * 0.5f; c.w = w; c.h = h; c.lensR = 0.f; c.focusD = 1.f;
    return c;
}
