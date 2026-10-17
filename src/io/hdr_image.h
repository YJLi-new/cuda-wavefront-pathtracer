// io/hdr_image.h - Radiance RGBE (.hdr / .pic) reader, written from the format description in G. Ward,
// "Real Pixels" (Graphics Gems II, 1991): new-style run-length scanlines and flat scanlines, "-Y H +X W" orientation.
#pragma once
#include <string>
#include <vector>

// rgb: 3 floats per pixel, row 0 = top of the image. Returns false (with a message on stderr) on failure.
bool loadHDR(const std::string& path, int& w, int& h, std::vector<float>& rgb);
