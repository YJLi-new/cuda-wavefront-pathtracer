// vec.h - minimal float3 helpers usable on host and device (written from scratch)
#pragma once
#include <cuda_runtime.h>
#include <math.h>
#define HD __host__ __device__ __forceinline__
HD float3 mk3(float x, float y, float z) { return make_float3(x, y, z); }
HD float3 operator+(float3 a, float3 b) { return mk3(a.x + b.x, a.y + b.y, a.z + b.z); }
HD float3 operator-(float3 a, float3 b) { return mk3(a.x - b.x, a.y - b.y, a.z - b.z); }
HD float3 operator-(float3 a) { return mk3(-a.x, -a.y, -a.z); }
HD float3 operator*(float3 a, float3 b) { return mk3(a.x * b.x, a.y * b.y, a.z * b.z); }
HD float3 operator*(float3 a, float s) { return mk3(a.x * s, a.y * s, a.z * s); }
HD float3 operator*(float s, float3 a) { return mk3(a.x * s, a.y * s, a.z * s); }
HD float3 operator/(float3 a, float s) { float r = 1.0f / s; return mk3(a.x * r, a.y * r, a.z * r); }
HD float3& operator+=(float3& a, float3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
HD float3& operator*=(float3& a, float3 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
HD float3& operator*=(float3& a, float s) { a.x *= s; a.y *= s; a.z *= s; return a; }
HD float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
HD float3 cross(float3 a, float3 b) { return mk3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
HD float length(float3 a) { return sqrtf(dot(a, a)); }
HD float3 normalize(float3 a) { float l = length(a); return l > 0.f ? a * (1.0f / l) : a; }
HD float3 vmin(float3 a, float3 b) { return mk3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)); }
HD float3 vmax(float3 a, float3 b) { return mk3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }
HD float maxc(float3 a) { return fmaxf(a.x, fmaxf(a.y, a.z)); }
HD float3 xyz(float4 a) { return mk3(a.x, a.y, a.z); }
HD float4 mk4(float3 a, float w) { return make_float4(a.x, a.y, a.z, w); }
