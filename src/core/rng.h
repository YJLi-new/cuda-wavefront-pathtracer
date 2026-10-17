// core/rng.h - stateless hash RNG: every random number is a pure function of (pixel, sample, depth, dimension, seed),
// so paths are reproducible and independent of launch order or batch size. Written from scratch.
#pragma once
#include "core/vec.h"

HD unsigned hash32(unsigned x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }

// uniform float in [0, 1) with 24 bits of resolution
HD float rng4(unsigned pix, unsigned smp, unsigned depth, unsigned dim, unsigned seed) {
    unsigned h = hash32(pix * 0x9E3779B1U ^ hash32(smp * 0x85EBCA77U ^ hash32((depth * 8u + dim) * 0xC2B2AE3DU ^ seed)));
    return (h >> 8) * (1.0f / 16777216.0f);
}
