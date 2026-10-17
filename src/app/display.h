// app/display.h - viewer display transform (app/display.cu)
#pragma once
#include <vector_types.h>

enum DisplayMode { kDisplayClip = 0, kDisplayReinhard = 1, kDisplayACES = 2 };

// accum: w*h radiance sums over spp samples (device); out: w*h RGBA8 (device). Runs on the default stream.
// mode kDisplayClip with exposure 0 gives the batch renderer's PNG values.
void launchDisplay(const float4* accum, uchar4* out, int n, int spp, float exposureEV, int mode);
