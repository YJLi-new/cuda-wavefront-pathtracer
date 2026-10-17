// bsdf_tests.cpp - CPU tests of the device BSDF code in src/render/bsdf.h (compiled here for the host).
//  (1) chi-square goodness of fit of every lobe's sampling routine against its pdf (in the spirit of Mitsuba's
//      chi2 tests): equal-solid-angle grid over the sphere plus one cell for "no sample", expected counts from
//      numerical integration of the pdf, cells with expected count < 5 pooled, p-value from the chi-square CDF.
//  (2) white furnace: Monte Carlo albedo E[weight] of a non-absorbing conductor (F0 = 1) and of a lossless
//      dielectric versus a deterministic quadrature over microfacet normals (exactly 1 for smooth surfaces);
//      glossy-diffuse with a white base must not exceed 1.
// Output: one JSON line per test, then a summary line. Exit code 0 if every test passes.
#include "render/bsdf.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static const double PI = 3.14159265358979323846;

// ---- regularized lower incomplete gamma P(a, x) (series / continued fraction) ----
static double gammaP(double a, double x) {
    if (x <= 0) return 0.0;
    double gln = lgamma(a);
    if (x < a + 1.0) {
        double ap = a, sum = 1.0 / a, del = sum;
        for (int n = 0; n < 100000; n++) { ap += 1; del *= x / ap; sum += del; if (fabs(del) < fabs(sum) * 1e-15) break; }
        return sum * exp(-x + a * log(x) - gln);
    }
    double b = x + 1.0 - a, c = 1.0 / 1e-300, d = 1.0 / b, h = d;
    for (int i = 1; i < 100000; i++) {
        double an = -i * (i - a); b += 2.0; d = an * d + b; if (fabs(d) < 1e-300) d = 1e-300; c = b + an / c; if (fabs(c) < 1e-300) c = 1e-300;
        d = 1.0 / d; double del = d * c; h *= del; if (fabs(del - 1.0) < 1e-15) break;
    }
    return 1.0 - exp(-x + a * log(x) - gln) * h;
}

struct Rng { std::mt19937_64 g; std::uniform_real_distribution<double> u{ 0.0, 1.0 }; explicit Rng(uint64_t s) : g(s) {} float next() { float f = (float)u(g); return f < 1.f ? f : 0.99999994f; } };

static float3 dirFromAngles(double thetaDeg) { double t = thetaDeg * PI / 180.0; return mk3((float)sin(t), 0.f, (float)cos(t)); }

// generic lobe interface for the tests
struct Lobe {
    std::string name; Material m; float eta;
    bool sample(float3 wo, Rng& r, float3& wi, float3& w, float& pdf) const { float u = r.next(), u1 = r.next(), u2 = r.next(), x1 = r.next(), x2 = r.next(); return bsdfSample(m, wo, eta, 1, x1, x2, u, u1, u2, wi, w, pdf); }
    float pdf(float3 wo, float3 wi) const { return bsdfEval(m, wo, wi, eta, 1, 0.5f, 0.5f).pdf; }   // the pdf does not depend on (x1, x2)
};

static Material mkMat(float3 base, float metallic, float rough, float ws, float transm, float ior) {
    Material m = defaultMaterial(); m.baseColor = base; m.metallic = metallic; m.roughness = rough; m.specularWeight = ws; m.transmission = transm; m.ior = ior; m.specularColor = mk3(1, 1, 1); return m;
}

static int nFail = 0, nTests = 0;

static void chi2Test(const Lobe& L, double thetaO, long nSamples, uint64_t seed) {
    const int NC = 20, NP = 40; const int SUB = getenv("CHI2_SUB") ? atoi(getenv("CHI2_SUB")) : 96;   // cos(theta) in [-1,1] x phi in [0, 2pi): equal solid angle cells
    float3 wo = dirFromAngles(thetaO);
    std::vector<double> obs(NC * NP + 1, 0.0), expc(NC * NP + 1, 0.0);
    Rng r(seed);
    for (long s = 0; s < nSamples; s++) {
        float3 wi, w; float pdf;
        if (!L.sample(wo, r, wi, w, pdf) || pdf <= 0.f) { obs[NC * NP] += 1; continue; }
        double c = std::max(-1.0, std::min(1.0, (double)wi.z)), ph = atan2((double)wi.y, (double)wi.x); if (ph < 0) ph += 2 * PI;
        int ic = std::min(NC - 1, (int)((c + 1.0) * 0.5 * NC)), ip = std::min(NP - 1, (int)(ph / (2 * PI) * NP));
        obs[ic * NP + ip] += 1;
    }
    double tot = 0;
    for (int ic = 0; ic < NC; ic++) for (int ip = 0; ip < NP; ip++) {
        double acc = 0, dc = 2.0 / NC / SUB, dp = 2 * PI / NP / SUB;
        for (int a = 0; a < SUB; a++) for (int b = 0; b < SUB; b++) {
            double c = -1.0 + (ic * SUB + a + 0.5) * dc, ph = (ip * SUB + b + 0.5) * dp, st = sqrt(std::max(0.0, 1 - c * c));
            acc += L.pdf(wo, mk3((float)(st * cos(ph)), (float)(st * sin(ph)), (float)c));
        }
        expc[ic * NP + ip] = acc * dc * dp * nSamples; tot += acc * dc * dp;
    }
    expc[NC * NP] = std::max(0.0, 1.0 - tot) * nSamples;
    // pool cells with expected count < 5
    double chi2 = 0, poolO = 0, poolE = 0; int cells = 0;
    for (size_t k = 0; k < obs.size(); k++) {
        if (expc[k] < 5.0) { poolO += obs[k]; poolE += expc[k]; continue; }
        chi2 += (obs[k] - expc[k]) * (obs[k] - expc[k]) / expc[k]; cells++;
    }
    if (poolE >= 5.0) { chi2 += (poolO - poolE) * (poolO - poolE) / poolE; cells++; }
    else if (poolO > 0 && poolE < 1e-3) { chi2 += 1e9; cells++; }     // samples where the pdf says there are none
    int dof = cells - 1; double p = 1.0 - gammaP(0.5 * dof, 0.5 * chi2);
    if (getenv("CHI2_DEBUG") && L.name == getenv("CHI2_DEBUG")) {   // print the worst cells
        std::vector<std::pair<double, int>> worst;
        for (size_t k = 0; k < obs.size(); k++) if (expc[k] >= 5.0) worst.push_back({ (obs[k] - expc[k]) * (obs[k] - expc[k]) / expc[k], (int)k });
        std::sort(worst.rbegin(), worst.rend());
        for (int j = 0; j < 12 && j < (int)worst.size(); j++) { int kk = worst[j].second; fprintf(stderr, "cell %d (cos %d phi %d): obs %.0f exp %.1f\n", kk, kk / NP, kk % NP, obs[kk], expc[kk]); }
    }
    bool pass = p > 0.01; nTests++; if (!pass) nFail++;
    printf("{\"test\":\"chi2\",\"lobe\":\"%s\",\"theta_o\":%.0f,\"samples\":%ld,\"cells\":%d,\"chi2\":%.2f,\"dof\":%d,\"p\":%.4f,\"pdf_mass\":%.5f,\"pass\":%s}\n",
           L.name.c_str(), thetaO, nSamples, cells, chi2, dof, p, tot, pass ? "true" : "false");
}

// deterministic albedo by quadrature over microfacet normals m ~ D(m) cos(theta_m) (smooth integrand):
//   albedo = int D(m) (wo.m) / wo.z * [ F G2(wo,wr) 1(wr.z>0) + (1-F) G2(wo,wt) 1(wt.z<0) ] dm   (energy convention)
// conductor: F = 1, no transmission. Glossy-diffuse spec lobe: F = ws * Fdiel(wo.m), plus the white diffuse lobe:
//   kind 2 (energy compensation): 1 - E_spec, so the total is exactly 1; kind 3 (plain mix): 1 - Fdiel(wo.z).
static double quadAlbedo(int kind, float rough, float eta, float3 wo) {
    double a = (double)rough * rough, a2 = a * a; const int NU = 1024, NV = 256; double sum = 0;
    // u1 = 1 - s^2 removes the integrable (1 - u1)^(-1/2) endpoint singularity of the weight below (du1 = 2 s ds)
    for (int i = 0; i < NU; i++) for (int j = 0; j < NV; j++) {
        double sq = (i + 0.5) / NU, u1 = 1.0 - sq * sq, u2 = (j + 0.5) / NV, jac = 2.0 * sq;
        double t2 = a2 * u1 / (1.0 - u1), ct = 1.0 / sqrt(1.0 + t2), st = sqrt(std::max(0.0, 1 - ct * ct)), ph = 2 * PI * u2;   // GGX D cos sampling
        float3 m = mk3((float)(st * cos(ph)), (float)(st * sin(ph)), (float)ct);
        double om = dot(wo, m); if (om <= 0) continue;
        double g = om / (wo.z * ct);   // D(m) (wo.m)/wo.z divided by the density D(m) cos(theta_m)
        float3 wr = m * (float)(2 * om) - wo; double val = 0;
        if (kind == 0) { if (wr.z > 0) val = ggxG2(wo.z, wr.z, (float)a2); }
        else if (kind == 1) {
            double F = fresnelDielectric((float)om, eta);
            if (wr.z > 0) val += F * ggxG2(wo.z, wr.z, (float)a2);
            if (F < 1) { double cosT = sqrt(std::max(0.0, 1 - (1 - om * om) / (eta * eta))); float3 wt = wo * (-1.f / eta) + m * (float)(om / eta - cosT); if (wt.z < 0) val += (1 - F) * ggxG2(wo.z, -normalize(wt).z, (float)a2); }
        } else { if (wr.z > 0) val = fresnelDielectric((float)om, eta) * ggxG2(wo.z, wr.z, (float)a2); }
        sum += g * val * jac;
    }
    double r = sum / ((double)NU * NV);
    if (kind == 2) r = 1.0;                                  // E_spec + (1 - E_spec)
    if (kind == 3) r += 1.0 - fresnelDielectric(wo.z, eta);   // white Lambert lobe weighted by 1 - F(wo.z)
    return r;
}

static void furnaceTest(const char* name, int kind, float rough, float eta, double thetaO, long n, uint64_t seed) {
    Material m = kind == 0 ? mkMat(mk3(1, 1, 1), 1.f, rough, 1.f, 0.f, 1.5f) : kind == 1 ? mkMat(mk3(1, 1, 1), 0.f, rough, 1.f, 1.f, 1.5f) : mkMat(mk3(1, 1, 1), 0.f, rough, 1.f, 0.f, eta);
    float3 wo = dirFromAngles(thetaO); Rng r(seed); double acc = 0, acc2 = 0;
    for (long s = 0; s < n; s++) {
        float3 wi, w; float pdf; double x = 0;
        float u = r.next(), u1 = r.next(), u2 = r.next(), x1 = r.next(), x2 = r.next();
        if (bsdfSample(m, wo, eta, kind == 3 ? 0 : 1, x1, x2, u, u1, u2, wi, w, pdf)) x = luminance(w);
        acc += x; acc2 += x * x;
    }
    double mc = acc / n, se = sqrt(std::max(0.0, acc2 / n - mc * mc) / n);
    bool delta = (double)rough * rough < kDeltaAlpha;
    double ref = delta ? 1.0 : quadAlbedo(kind, rough, eta, wo);
    double relErr = (mc - ref) / ref;
    bool pass = fabs(relErr) <= 0.01;
    if (kind >= 2) pass = pass && mc <= 1.0 + 3 * se;      // glossy-diffuse must not exceed 1
    bool counted = kind != 3;                              // the plain mix is reported, not gated
    if (counted) { nTests++; if (!pass) nFail++; }
    printf("{\"test\":\"%s\",\"lobe\":\"%s\",\"roughness\":%.2f,\"eta\":%.4f,\"theta_o\":%.0f,\"mc_albedo\":%.5f,\"mc_se\":%.5f,\"reference\":%.5f,\"rel_err\":%.5f,\"pass\":%s}\n",
           counted ? "furnace" : "furnace_info", name, rough, eta, thetaO, mc, se, ref, relErr, pass ? "true" : "false");
}

int main() {
    const long NS = 2000000; uint64_t seed = 1000;
    // (1) chi-square tests: 16 configurations
    std::vector<std::pair<Lobe, std::vector<double>>> cfg;
    cfg.push_back({ { "diffuse", mkMat(mk3(0.8f, 0.8f, 0.8f), 0.f, 0.5f, 0.f, 0.f, 1.5f), 1.5f }, { 30 } });
    for (float r : { 0.3f, 0.7f }) {
        char n[64];
        snprintf(n, sizeof n, "conductor_r%.1f", r); cfg.push_back({ { n, mkMat(mk3(0.95f, 0.64f, 0.54f), 1.f, r, 1.f, 0.f, 1.5f), 1.5f }, { 30, 75 } });
        snprintf(n, sizeof n, "glossy_diffuse_r%.1f", r); cfg.push_back({ { n, mkMat(mk3(0.2f, 0.4f, 0.8f), 0.f, r, 1.f, 0.f, 1.5f), 1.5f }, { 30, 75 } });
        snprintf(n, sizeof n, "dielectric_out_r%.1f", r); cfg.push_back({ { n, mkMat(mk3(1, 1, 1), 0.f, r, 1.f, 1.f, 1.5f), 1.5f }, { 30, 75 } });
    }
    cfg.push_back({ { "dielectric_in_r0.5", mkMat(mk3(1, 1, 1), 0.f, 0.5f, 1.f, 1.f, 1.5f), 1.f / 1.5f }, { 30, 60 } });
    cfg.push_back({ { "mixture_m0.3_t0.4_r0.5", mkMat(mk3(0.7f, 0.5f, 0.3f), 0.3f, 0.5f, 1.f, 0.4f, 1.5f), 1.5f }, { 45 } });
    for (auto& c : cfg) for (double th : c.second) chi2Test(c.first, th, NS, seed++);
    // (2) white furnace
    for (float r : { 0.0f, 0.1f, 0.3f, 0.5f, 0.7f, 1.0f }) for (double th : { 1.0, 45.0, 80.0 }) {
        furnaceTest("conductor_F0_1", 0, r, 1.5f, th, 1000000, seed++);
        furnaceTest("dielectric_out", 1, r, 1.5f, th, 1000000, seed++);
        furnaceTest("dielectric_in", 1, r, 1.f / 1.5f, th, 1000000, seed++);
        furnaceTest("glossy_diffuse_white", 2, r, 1.5f, th, 1000000, seed++);
        furnaceTest("glossy_diffuse_white_plain_mix", 3, r, 1.5f, th, 1000000, seed++);
    }
    printf("{\"summary\":true,\"tests\":%d,\"failed\":%d}\n", nTests, nFail);
    return nFail ? 1 : 0;
}
