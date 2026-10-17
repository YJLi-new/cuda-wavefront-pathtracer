// render/bsdf.h - material BSDFs (host + device; the CPU tests run exactly this code).
// Local shading frame: z = shading normal, and the caller guarantees wo.z > 0 (wo = toward the viewer).
// Lobes, written from the cited papers:
//   GGX microfacet distribution with height-correlated Smith masking-shadowing (Walter et al. 2007; Heitz 2014)
//   visible-normal sampling with spherical caps (Dupuy & Benyoub 2023)
//   rough dielectric reflection + refraction (Walter et al. 2007), exact unpolarized Fresnel
//   conductor Fresnel: Schlick with the F82 tint term (Kutz et al. 2021, as specified by OpenPBR)
//   Lambertian diffuse with cosine-weighted sampling
// Transmission uses the energy (symmetric) convention: the eta^2 radiance scaling of Walter's BTDF is not applied,
// so a lossless smooth interface transmits throughput 1 - F. For closed objects the factors cancel anyway.
#pragma once
#include "core/sampling.h"
#include "scene/material.h"

#define CWP_PI 3.14159265f
static constexpr float kDeltaAlpha = 1e-3f;   // GGX alpha below this is treated as a perfectly specular (delta) lobe

HD float sqrf(float x) { return x * x; }
HD float luminance(float3 c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }

// ---------------- GGX ----------------
HD float ggxD(float cosM, float a2) { if (cosM <= 0.f) return 0.f; float c2 = cosM * cosM; float d = c2 * (a2 - 1.f) + 1.f; return a2 / (CWP_PI * d * d); }
HD float ggxLambda(float cosW, float a2) { float c2 = cosW * cosW; if (c2 < 1e-12f) return 1e12f; return 0.5f * (sqrtf(1.f + a2 * (1.f - c2) / c2) - 1.f); }
HD float ggxG1(float cosW, float a2) { return 1.f / (1.f + ggxLambda(cosW, a2)); }
HD float ggxG2(float cosO, float cosI, float a2) { return 1.f / (1.f + ggxLambda(cosO, a2) + ggxLambda(cosI, a2)); }
// density of visible normals m seen from wo (wo.z > 0): G1(wo) max(0, wo.m) D(m) / wo.z
HD float ggxVndfPdf(float3 wo, float3 m, float a2) { float om = dot(wo, m); if (om <= 0.f || m.z <= 0.f) return 0.f; return ggxG1(wo.z, a2) * om * ggxD(m.z, a2) / wo.z; }
// sample a visible normal: spherical-cap method of Dupuy & Benyoub 2023 (isotropic alpha)
HD float3 ggxSampleVndf(float3 wo, float a, float u1, float u2) {
    float3 wh = normalize(mk3(a * wo.x, a * wo.y, wo.z));
    float ph = CWP_TWO_PI * u1;
    float z = (1.f - u2) * (1.f + wh.z) - wh.z;
    float st = sqrtf(fminf(fmaxf(1.f - z * z, 0.f), 1.f));
    float3 h = mk3(st * cosf(ph), st * sinf(ph), z) + wh;
    return normalize(mk3(a * h.x, a * h.y, fmaxf(h.z, 0.f)));
}

// ---------------- Fresnel ----------------
// unpolarized dielectric Fresnel; cosI > 0 on the incident side, eta = eta_transmitted / eta_incident
HD float fresnelDielectric(float cosI, float eta) {
    float s2t = (1.f - cosI * cosI) / (eta * eta);
    if (s2t >= 1.f) return 1.f;
    float cosT = sqrtf(1.f - s2t);
    float rs = (cosI - eta * cosT) / (cosI + eta * cosT), rp = (eta * cosI - cosT) / (eta * cosI + cosT);
    return 0.5f * (rs * rs + rp * rp);
}
// Schlick with the F82 tint correction: F(mu) = Fs(mu) - a mu (1-mu)^6, a chosen so that F(1/7) = tint * Fs(1/7)
HD float3 fresnelF82(float3 F0, float3 tint, float mu) {
    mu = fminf(fmaxf(mu, 0.f), 1.f);
    float m = 1.f - mu, m2 = m * m, m5 = m2 * m2 * m, m6 = m5 * m;
    const float mb = 1.f / 7.f, mbm5 = 0.46266436f /* (6/7)^5 */, mbm6 = 0.39656945f /* (6/7)^6 */;
    float3 one = mk3(1.f, 1.f, 1.f);
    float3 Fs = F0 + (one - F0) * m5;
    float3 Fsb = F0 + (one - F0) * mbm5;
    float3 a = Fsb * (one - tint) * (1.f / (mb * mbm6));
    float3 F = Fs - a * (mu * m6);
    return mk3(fminf(fmaxf(F.x, 0.f), 1.f), fminf(fmaxf(F.y, 0.f), 1.f), fminf(fmaxf(F.z, 0.f), 1.f));
}

// ---------------- lobes (local frame, wo.z > 0) ----------------
struct BsdfEval { float3 f; float pdf; };

// opaque dielectric base: glossy-diffuse = GGX specular (weight ws, IOR eta) + kd * Lambert, a non-physical mix.
//   energy = 0: kd = 1 - ws F(wo.z)            (plain Fresnel-weighted mix; can exceed albedo 1 by < 0.1%)
//   energy = 1: kd = 1 - X, E[X] = E_spec(wo)  (albedo scaling: X = ws F(wo.m) G2 / G1(wo) for one visible normal
//               m drawn with (ux1, ux2), an unbiased estimate of the specular lobe's directional albedo, so the
//               expected albedo is E_spec + (1 - E_spec) * base <= 1). Lobe selection uses F(wo.z) in both modes.
struct GlossyDiffuse {
    float3 base; float ws, eta, a, a2; bool delta; float Fo, pSpec, kd;
    HD GlossyDiffuse(const Material& m, float3 wo, int energy, float ux1, float ux2) {
        base = m.baseColor; ws = m.specularWeight; eta = m.ior; a = m.roughness * m.roughness; a2 = a * a; delta = a < kDeltaAlpha;
        Fo = ws > 0.f ? ws * fresnelDielectric(wo.z, eta) : 0.f;
        float pd = (1.f - Fo) * luminance(base); pSpec = Fo + pd > 0.f ? Fo / (Fo + pd) : 0.f;
        kd = 1.f - Fo;
        if (energy && ws > 0.f && !delta) {
            float3 mm = ggxSampleVndf(wo, a, ux1, ux2); float om = dot(wo, mm); float3 wr = mm * (2.f * om) - wo;
            float X = (om > 0.f && wr.z > 0.f) ? ws * fresnelDielectric(om, eta) * ggxG2(wo.z, wr.z, a2) / ggxG1(wo.z, a2) : 0.f;
            kd = 1.f - X;
        }
        kd = fmaxf(kd, 0.f);
    }
    HD BsdfEval eval(float3 wo, float3 wi) const {
        BsdfEval r; r.f = mk3(0.f, 0.f, 0.f); r.pdf = 0.f;
        if (wi.z <= 0.f) return r;
        r.f = base * (kd * CWP_INV_PI); r.pdf = (1.f - pSpec) * wi.z * CWP_INV_PI;
        if (!delta && Fo > 0.f) {
            float3 m = normalize(wo + wi); float om = dot(wo, m);
            if (om > 0.f) {
                float F = ws * fresnelDielectric(om, eta);
                float s = F * ggxD(m.z, a2) * ggxG2(wo.z, wi.z, a2) / (4.f * wo.z * wi.z);
                r.f = r.f + mk3(s, s, s); r.pdf += pSpec * ggxVndfPdf(wo, m, a2) / (4.f * om);
            }
        }
        return r;
    }
    // returns false if no direction; pdf = 0 marks a delta sample (weight already includes all factors)
    HD bool sample(float3 wo, float u, float u1, float u2, float3& wi, float3& w, float& pdf) const {
        if (u < pSpec) {
            if (delta) { wi = mk3(-wo.x, -wo.y, wo.z); float s = Fo / pSpec; w = mk3(s, s, s); pdf = 0.f; return true; }
            float3 m = ggxSampleVndf(wo, a, u1, u2); wi = m * (2.f * dot(wo, m)) - wo;
            if (wi.z <= 0.f) return false;
        } else {
            wi = sampleCosineHemisphere(mk3(0.f, 0.f, 1.f), u1, u2);
            if (wi.z <= 0.f) return false;
        }
        BsdfEval e = eval(wo, wi); if (e.pdf <= 0.f) return false;
        w = e.f * (wi.z / e.pdf); pdf = e.pdf; return true;
    }
};

// metallic base: rough conductor, F = ws * F82(base, specular_color)
struct Conductor {
    float3 F0, tint; float ws, a, a2; bool delta;
    HD Conductor(const Material& m) { F0 = m.baseColor; tint = m.specularColor; ws = m.specularWeight; a = m.roughness * m.roughness; a2 = a * a; delta = a < kDeltaAlpha; }
    HD BsdfEval eval(float3 wo, float3 wi) const {
        BsdfEval r; r.f = mk3(0.f, 0.f, 0.f); r.pdf = 0.f;
        if (delta || wi.z <= 0.f) return r;
        float3 m = normalize(wo + wi); float om = dot(wo, m); if (om <= 0.f) return r;
        r.f = fresnelF82(F0, tint, om) * (ws * ggxD(m.z, a2) * ggxG2(wo.z, wi.z, a2) / (4.f * wo.z * wi.z));
        r.pdf = ggxVndfPdf(wo, m, a2) / (4.f * om);
        return r;
    }
    HD bool sample(float3 wo, float u1, float u2, float3& wi, float3& w, float& pdf) const {
        if (delta) { wi = mk3(-wo.x, -wo.y, wo.z); w = fresnelF82(F0, tint, wo.z) * ws; pdf = 0.f; return true; }
        float3 m = ggxSampleVndf(wo, a, u1, u2); float om = dot(wo, m); wi = m * (2.f * om) - wo;
        if (wi.z <= 0.f || om <= 0.f) return false;
        // weight = F G2 / G1(wo)
        w = fresnelF82(F0, tint, om) * (ws * ggxG2(wo.z, wi.z, a2) / ggxG1(wo.z, a2));
        pdf = ggxVndfPdf(wo, m, a2) / (4.f * om); return true;
    }
};

// transmissive dielectric base: rough reflection + refraction; eta = eta(other side) / eta(wo side); refraction tinted
struct Dielectric {
    float3 tint; float eta, a, a2; bool delta;
    HD Dielectric(const Material& m, float etaRel) { tint = m.baseColor; eta = etaRel; a = m.roughness * m.roughness; a2 = a * a; delta = a < kDeltaAlpha; }
    HD BsdfEval eval(float3 wo, float3 wi) const {
        BsdfEval r; r.f = mk3(0.f, 0.f, 0.f); r.pdf = 0.f;
        if (delta) return r;
        if (wi.z > 0.f) {
            float3 m = normalize(wo + wi); float om = dot(wo, m); if (om <= 0.f) return r;
            float F = fresnelDielectric(om, eta);
            float s = F * ggxD(m.z, a2) * ggxG2(wo.z, wi.z, a2) / (4.f * wo.z * wi.z);
            r.f = mk3(s, s, s); r.pdf = F * ggxVndfPdf(wo, m, a2) / (4.f * om);
        } else if (wi.z < 0.f) {
            float3 m = normalize(wo + wi * eta); if (m.z < 0.f) m = -m;
            float om = dot(wo, m), im = dot(wi, m); if (om <= 0.f || im >= 0.f) return r;
            float F = fresnelDielectric(om, eta); if (F >= 1.f) return r;
            float den = om + eta * im, j = eta * eta * (-im) / (den * den);   // |d m / d wi|
            float s = (1.f - F) * ggxD(m.z, a2) * ggxG2(wo.z, -wi.z, a2) * om * j / (wo.z * -wi.z);
            r.f = tint * s; r.pdf = (1.f - F) * ggxVndfPdf(wo, m, a2) * j;
        }
        return r;
    }
    HD bool sample(float3 wo, float u, float u1, float u2, float3& wi, float3& w, float& pdf) const {
        float3 m = delta ? mk3(0.f, 0.f, 1.f) : ggxSampleVndf(wo, a, u1, u2);
        float om = dot(wo, m); if (om <= 0.f) return false;
        float F = fresnelDielectric(om, eta);
        if (u < F) {
            wi = m * (2.f * om) - wo; if (wi.z <= 0.f) return false;
            if (delta) { w = mk3(1.f, 1.f, 1.f); pdf = 0.f; return true; }
        } else {
            float cosT = sqrtf(fmaxf(1.f - (1.f - om * om) / (eta * eta), 0.f));
            wi = wo * (-1.f / eta) + m * (om / eta - cosT); if (wi.z >= 0.f) return false;
            wi = normalize(wi);
            if (delta) { w = tint; pdf = 0.f; return true; }
        }
        BsdfEval e = eval(wo, wi); if (e.pdf <= 0.f) return false;
        w = e.f * (fabsf(wi.z) / e.pdf); pdf = e.pdf; return true;
    }
};

// ---------------- full material: linear blend of the three bases ----------------
HD void baseWeights(const Material& m, float& wM, float& wT, float& wD) { wM = m.metallic; wT = (1.f - m.metallic) * m.transmission; wD = (1.f - m.metallic) * (1.f - m.transmission); }

// true if some lobe can be evaluated for light sampling (i.e. not purely specular)
HD bool bsdfHasNonDelta(const Material& m) {
    float wM, wT, wD; baseWeights(m, wM, wT, wD);
    bool rough = m.roughness * m.roughness >= kDeltaAlpha;
    return (wD > 0.f && (luminance(m.baseColor) > 0.f || rough)) || ((wM > 0.f || wT > 0.f) && rough);
}

// f and pdf (solid angle) of all non-delta lobes; etaRel is used by the transmissive base only;
// gdEnergy / ux1 / ux2: glossy-diffuse energy compensation switch and its two random numbers (see GlossyDiffuse)
HD BsdfEval bsdfEval(const Material& m, float3 wo, float3 wi, float etaRel, int gdEnergy, float ux1, float ux2) {
    float wM, wT, wD; baseWeights(m, wM, wT, wD);
    BsdfEval r; r.f = mk3(0.f, 0.f, 0.f); r.pdf = 0.f;
    if (wM > 0.f) { BsdfEval e = Conductor(m).eval(wo, wi); r.f = r.f + e.f * wM; r.pdf += wM * e.pdf; }
    if (wT > 0.f) { BsdfEval e = Dielectric(m, etaRel).eval(wo, wi); r.f = r.f + e.f * wT; r.pdf += wT * e.pdf; }
    if (wD > 0.f) { BsdfEval e = GlossyDiffuse(m, wo, gdEnergy, ux1, ux2).eval(wo, wi); r.f = r.f + e.f * wD; r.pdf += wD * e.pdf; }
    return r;
}

// Samples wi. weight = f |cos| / pdf of the whole material (one-sample MIS over the bases); pdf = 0 for delta samples.
HD bool bsdfSample(const Material& m, float3 wo, float etaRel, int gdEnergy, float ux1, float ux2, float u, float u1, float u2, float3& wi, float3& weight, float& pdf) {
    float wM, wT, wD; baseWeights(m, wM, wT, wD);
    bool ok;
    if (u < wM) { ok = Conductor(m).sample(wo, u1, u2, wi, weight, pdf); }
    else if (u < wM + wT) { float uu = (u - wM) / wT; ok = Dielectric(m, etaRel).sample(wo, fminf(uu, 0.99999994f), u1, u2, wi, weight, pdf); }
    else { float uu = wD > 0.f ? (u - wM - wT) / wD : 0.f; ok = GlossyDiffuse(m, wo, gdEnergy, ux1, ux2).sample(wo, fminf(uu, 0.99999994f), u1, u2, wi, weight, pdf); }
    if (!ok) return false;
    if (pdf <= 0.f) return true;                         // delta: other bases have no density there
    if (wM == 1.f || wT == 1.f || wD == 1.f) {           // single base: its own weight is already f|cos|/pdf
        return true;
    }
    BsdfEval e = bsdfEval(m, wo, wi, etaRel, gdEnergy, ux1, ux2); if (e.pdf <= 0.f) return false;
    weight = e.f * (fabsf(wi.z) / e.pdf); pdf = e.pdf; return true;
}

HD float misPower(float a, float b) { float a2 = a * a; return a2 / (a2 + b * b); }
