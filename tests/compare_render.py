#!/usr/bin/env python3
# compare_render.py - compares a pt render with a reference render of the same scene (linear PFMs).
#   python compare_render.py ours.pfm reference.pfm [--coverage coverage.pfm] [--json out.json]
# Readouts: PSNR of the full image in display space (clip [0,1] + sRGB OETF, exposure 0, peak 1), and mean luminance
# ours / reference - 1 over the full image and, with a coverage map (tests/cycles_gltf_reference.py), over the object
# (pixels with coverage >= 0.999, i.e. no partially covered edge pixels) and over the background (coverage <= 0.001).
import argparse, json, math
import numpy as np

def read_pfm(p):
    with open(p, "rb") as f:
        assert f.readline().strip() == b"PF"; w, h = map(int, f.readline().split()); s = float(f.readline())
        return np.frombuffer(f.read(), dtype="<f4" if s < 0 else ">f4").reshape(h, w, 3)[::-1].astype(np.float64)

def srgb(x):
    x = np.clip(x, 0, 1); return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)

def lum(x): return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]

def region(A, B, m):
    la, lb = lum(A[m]).mean(), lum(B[m]).mean()
    return dict(pixels=int(m.sum()), mean_lum_ours=float(la), mean_lum_ref=float(lb), rel_diff=float(la / lb - 1),
                rel_diff_rgb=[float(x) for x in A[m].mean(0) / B[m].mean(0) - 1])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ours"); ap.add_argument("ref"); ap.add_argument("--coverage"); ap.add_argument("--json")
    a = ap.parse_args()
    A, B = read_pfm(a.ours), read_pfm(a.ref)
    mse = float(np.mean((srgb(A) - srgb(B)) ** 2))
    out = dict(ours=a.ours, ref=a.ref, psnr_srgb=10 * math.log10(1 / mse) if mse > 0 else float("inf"), full=region(A, B, np.ones(A.shape[:2], bool)))
    if a.coverage:
        cov = lum(read_pfm(a.coverage))
        out["object"] = region(A, B, cov >= 0.999); out["background"] = region(A, B, cov <= 0.001)
        mo = cov >= 0.999; mseo = float(np.mean((srgb(A[mo]) - srgb(B[mo])) ** 2)); out["psnr_srgb_object"] = 10 * math.log10(1 / mseo) if mseo > 0 else float("inf")
    for k in ("full", "object", "background"):
        if k in out: r = out[k]; print("%-10s px %7d  ours %.5f  ref %.5f  rel %+.4f  rgb %s" % (k, r["pixels"], r["mean_lum_ours"], r["mean_lum_ref"], r["rel_diff"], " ".join("%+.4f" % x for x in r["rel_diff_rgb"])))
    print("PSNR (sRGB) full %.2f dB%s" % (out["psnr_srgb"], "  object %.2f dB" % out["psnr_srgb_object"] if a.coverage else ""))
    if a.json: json.dump(out, open(a.json, "w"), indent=1)
