// png.h - minimal PNG (stored/uncompressed deflate) and PFM writers (written from scratch)
#pragma once
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>

static uint32_t crcTable[256];
static void crcInit() { for (uint32_t n = 0; n < 256; n++) { uint32_t c = n; for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1; crcTable[n] = c; } }
static uint32_t crc32buf(const uint8_t* b, size_t n, uint32_t c = 0xFFFFFFFFu) { for (size_t i = 0; i < n; i++) c = crcTable[(c ^ b[i]) & 0xFF] ^ (c >> 8); return c; }
static void put32(std::vector<uint8_t>& v, uint32_t x) { v.push_back(x >> 24); v.push_back(x >> 16); v.push_back(x >> 8); v.push_back(x); }
static void chunk(FILE* fp, const char* type, const std::vector<uint8_t>& data) {
    std::vector<uint8_t> b; put32(b, (uint32_t)data.size());
    std::vector<uint8_t> td(type, type + 4); td.insert(td.end(), data.begin(), data.end());
    uint32_t c = crc32buf(td.data(), td.size()) ^ 0xFFFFFFFFu;
    fwrite(b.data(), 1, 4, fp); fwrite(td.data(), 1, td.size(), fp);
    std::vector<uint8_t> cb; put32(cb, c); fwrite(cb.data(), 1, 4, fp);
}
static bool writePNG(const std::string& path, int w, int h, const uint8_t* rgb) {
    static bool init = false; if (!init) { crcInit(); init = true; }
    FILE* fp = fopen(path.c_str(), "wb"); if (!fp) return false;
    const uint8_t sig[8] = { 137, 80, 78, 71, 13, 10, 26, 10 }; fwrite(sig, 1, 8, fp);
    std::vector<uint8_t> ihdr; put32(ihdr, w); put32(ihdr, h); ihdr.push_back(8); ihdr.push_back(2); ihdr.push_back(0); ihdr.push_back(0); ihdr.push_back(0);
    chunk(fp, "IHDR", ihdr);
    std::vector<uint8_t> raw; raw.reserve((size_t)h * (w * 3 + 1));
    for (int y = 0; y < h; y++) { raw.push_back(0); raw.insert(raw.end(), rgb + (size_t)y * w * 3, rgb + (size_t)(y + 1) * w * 3); }
    std::vector<uint8_t> z; z.push_back(0x78); z.push_back(0x01);
    size_t pos = 0; uint32_t a = 1, b = 0;
    for (size_t i = 0; i < raw.size(); i++) { a = (a + raw[i]) % 65521; b = (b + a) % 65521; }
    while (pos < raw.size()) {
        size_t n = raw.size() - pos; if (n > 65535) n = 65535;
        bool last = pos + n == raw.size();
        z.push_back(last ? 1 : 0); z.push_back(n & 0xFF); z.push_back(n >> 8); z.push_back((~n) & 0xFF); z.push_back(((~n) >> 8) & 0xFF);
        z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + n); pos += n;
    }
    put32(z, (b << 16) | a);
    chunk(fp, "IDAT", z);
    chunk(fp, "IEND", std::vector<uint8_t>());
    fclose(fp); return true;
}
// PFM: linear float RGB, little endian, bottom-to-top rows
static bool writePFM(const std::string& path, int w, int h, const float* rgb) {
    FILE* fp = fopen(path.c_str(), "wb"); if (!fp) return false;
    fprintf(fp, "PF\n%d %d\n-1.0\n", w, h);
    for (int y = h - 1; y >= 0; y--) fwrite(rgb + (size_t)y * w * 3, sizeof(float), (size_t)w * 3, fp);
    fclose(fp); return true;
}
