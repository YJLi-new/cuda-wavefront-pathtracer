#!/usr/bin/env python3
# mis_consistency.py - checks that NEE-only, BSDF-only and MIS renders converge to the same image.
#   python mis_consistency.py DIR [--modes mis,nee,bsdf] [--grid 4x4] [--json out.json]
# DIR holds <mode>_s<seed>.pfm for independent seeds. For every region of a grid over the image the mean luminance
# of each seed is computed; per mode the regional mean and its standard error (std over seeds / sqrt(n)) follow,
# and every pair of modes is compared with z = |mean_a - mean_b| / sqrt(se_a^2 + se_b^2). Pass: all z <= 3.
import argparse, glob, json, os, re
import numpy as np

def read_pfm(p):
    with open(p, "rb") as f:
        assert f.readline().strip() == b"PF"; w, h = map(int, f.readline().split()); s = float(f.readline())
        return np.frombuffer(f.read(), dtype="<f4" if s < 0 else ">f4").reshape(h, w, 3)[::-1].astype(np.float64)

ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--modes", default="mis,nee,bsdf"); ap.add_argument("--grid", default="4x4"); ap.add_argument("--json")
a = ap.parse_args()
gx, gy = map(int, a.grid.split("x")); modes = a.modes.split(",")
stats = {}
for md in modes:
    files = sorted(glob.glob(os.path.join(a.dir, md + "_s*.pfm")))
    vals = []
    for f in files:
        im = read_pfm(f); L = 0.2126 * im[..., 0] + 0.7152 * im[..., 1] + 0.0722 * im[..., 2]; h, w = L.shape
        vals.append([L[j * h // gy:(j + 1) * h // gy, i * w // gx:(i + 1) * w // gx].mean() for j in range(gy) for i in range(gx)])
    v = np.array(vals); stats[md] = dict(n=len(files), mean=v.mean(0), se=v.std(0, ddof=1) / np.sqrt(len(files)))
pairs = []; zmax = 0.0
for i in range(len(modes)):
    for j in range(i + 1, len(modes)):
        A, B = stats[modes[i]], stats[modes[j]]
        z = np.abs(A["mean"] - B["mean"]) / np.sqrt(A["se"] ** 2 + B["se"] ** 2)
        pairs.append(dict(a=modes[i], b=modes[j], z_max=float(z.max()), z_mean=float(z.mean()), max_rel_diff=float(np.max(np.abs(A["mean"] / B["mean"] - 1)))))
        zmax = max(zmax, float(z.max()))
rel_se = {m: float(np.median(stats[m]["se"] / stats[m]["mean"])) for m in modes}
out = dict(grid=a.grid, seeds={m: stats[m]["n"] for m in modes}, pairs=pairs, z_max=zmax, regions=gx * gy, median_rel_se=rel_se, pass_=zmax <= 3.0,
           region_means={m: stats[m]["mean"].tolist() for m in modes})
for p in pairs: print("%s vs %s: z_max %.2f  z_mean %.2f  max rel diff %.4f" % (p["a"], p["b"], p["z_max"], p["z_mean"], p["max_rel_diff"]))
print("median relative SE per region:", {m: round(v, 5) for m, v in rel_se.items()}, " PASS" if zmax <= 3 else " FAIL")
if a.json: json.dump(out, open(a.json, "w"), indent=1)
