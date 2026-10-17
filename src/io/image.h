// io/image.h - minimal PNG (stored deflate) and PFM writers, and the display transform
#pragma once
#include <cstdint>
#include <string>
#include <vector>

bool writePNG(const std::string& path, int w, int h, const uint8_t* rgb);  // 8-bit RGB, top-to-bottom rows
bool writePFM(const std::string& path, int w, int h, const float* rgb);    // linear float RGB, top-to-bottom input rows
// linear radiance (3 floats per pixel) -> 8-bit sRGB: clip to [0,1], sRGB OETF, exposure 0
void tonemapSRGB8(const float* lin, size_t nPix, uint8_t* rgb);
