#!/usr/bin/env python3
# env_sampling_test.py - environment-map importance sampling vs uniform sphere sampling on the same scene.
#   python env_sampling_test.py DIR [--scene SCENE_DIR] [--grid 4x4] [--json out.json]
# DIR holds importance_s<seed>.pfm and uniform_s<seed>.pfm (independent seeds, same spp). Unbiasedness: the image mean
# luminance and the mean of every grid region agree between the two strategies, z = |m_a - m_b| / sqrt(se_a^2 + se_b^2)
# with se = std over seeds / sqrt(n); pass: all z <= 3. Efficiency: per-pixel luminance variance over seeds, averaged
# over the image (and over the material spheres when --scene gives the tests/material_spheres.py scene), as the ratio
# uniform / importance at equal spp, plus the median of the per-pixel ratios.
import argparse, glob, json, os, sys
import numpy as np

def read_pfm(p):
    with open(p, "rb") as f:
        assert f.readline().strip() == b"PF"; w, h = map(int, f.readline().split()); s = float(f.readline())
        return np.frombuffer(f.read(), dtype="<f4" if s < 0 else ">f4").reshape(h, w, 3)[::-1].astype(np.float64)

ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--scene"); ap.add_argument("--grid", default="4x4"); ap.add_argument("--json")
a = ap.parse_args()
gx, gy = map(int, a.grid.split("x")); modes = ["importance", "uniform"]
L = {}
for md in modes:
    ims = [read_pfm(f) for f in sorted(glob.glob(os.path.join(a.dir, md + "_s*.pfm")))]
    L[md] = np.array([0.2126 * im[..., 0] + 0.7152 * im[..., 1] + 0.0722 * im[..., 2] for im in ims])   # seeds x h x w
n, h, w = L["importance"].shape
def stats(x):   # x: seeds x k -> mean, se
    return x.mean(0), x.std(0, ddof=1) / np.sqrt(x.shape[0])
def regions(Ls): return np.stack([Ls[:, j * h // gy:(j + 1) * h // gy, i * w // gx:(i + 1) * w // gx].mean((1, 2)) for j in range(gy) for i in range(gx)], 1)
res = {}
for name, f in (("image", lambda Ls: Ls.mean((1, 2))[:, None]), ("regions", regions)):
    (ma, sa), (mb, sb) = stats(f(L["importance"])), stats(f(L["uniform"]))
    z = np.abs(ma - mb) / np.sqrt(sa ** 2 + sb ** 2)
    res[name] = dict(z=z.tolist(), z_max=float(z.max()), mean_importance=ma.tolist(), mean_uniform=mb.tolist(), se_importance=sa.tolist(), se_uniform=sb.tolist(),
                     max_rel_diff=float(np.max(np.abs(mb / ma - 1))))
vi, vu = L["importance"].var(0, ddof=1), L["uniform"].var(0, ddof=1)
vr = dict(image_mean_var_ratio=float(vu.mean() / vi.mean()), image_median_pixel_ratio=float(np.median(vu[vi > 0] / vi[vi > 0])))
if a.scene:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from compare_spheres import masks
    ms, bg, names = masks(a.scene, w, h); m = np.logical_or.reduce(ms)
    vr["spheres_mean_var_ratio"] = float(vu[m].mean() / vi[m].mean()); vr["spheres_pixels"] = int(m.sum())
    vr["per_sphere_mean_var_ratio"] = {nm: float(vu[mk].mean() / vi[mk].mean()) for nm, mk in zip(names, ms)}
zmax = max(res["image"]["z_max"], res["regions"]["z_max"])
out = dict(seeds=n, grid=a.grid, unbiasedness=res, z_max=zmax, pass_=zmax <= 3.0, variance_reduction=vr)
print("seeds %d  image mean: importance %.6f +- %.6f  uniform %.6f +- %.6f  z %.2f" % (n, res["image"]["mean_importance"][0], res["image"]["se_importance"][0], res["image"]["mean_uniform"][0], res["image"]["se_uniform"][0], res["image"]["z"][0]))
print("regions %s: z_max %.2f  max rel diff %.4f   -> %s" % (a.grid, res["regions"]["z_max"], res["regions"]["max_rel_diff"], "PASS" if zmax <= 3 else "FAIL"))
print("variance uniform / importance at equal spp:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in vr.items() if k != "per_sphere_mean_var_ratio"}))
if "per_sphere_mean_var_ratio" in vr: print("per sphere:", {k: round(v, 3) for k, v in vr["per_sphere_mean_var_ratio"].items()})
if a.json: json.dump(out, open(a.json, "w"), indent=1)
