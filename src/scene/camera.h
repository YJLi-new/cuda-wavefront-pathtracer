// scene/camera.h - pinhole or thin-lens camera (OpenGL convention: looks down its -Z axis, +Y up; world z-up).
// Thin lens: lensR = aperture (lens) radius in world units, focusD = distance of the plane in focus along the view axis
// (camera-space depth). lensR = 0 is the pinhole camera and takes exactly the pinhole code path.
#pragma once
#include "core/vec.h"

struct Cam { float3 pos, ax, ay, az; float fx, fy, px, py; int w, h; float lensR, focusD; };

// identity camera-to-world matrix (row-major 4x4)
void identityC2W(float c2w[16]);
// row-major camera-to-world matrix looking from pos at tgt
void lookAtC2W(float3 pos, float3 tgt, float3 up, float c2w[16]);
// focal length in pixels from a horizontal field of view in degrees
float focalFromHfov(int width, float hfovDeg);
// pinhole camera from a row-major camera-to-world matrix and intrinsics (principal point at the image centre)
Cam makeCamera(const float c2w[16], float fx, float fy, int w, int h);
