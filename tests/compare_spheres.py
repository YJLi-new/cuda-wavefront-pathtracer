#!/usr/bin/env python3
# compare_spheres.py - per-sphere mean radiance of two renders of the material-sphere scene (material_spheres.py).
#   python compare_spheres.py SCENE_DIR ours.pfm reference.pfm [--json out.json]
# Each sphere's mask = pixels whose four corner rays all hit the sphere (no partially covered edge pixels).
# Background = pixels whose centre ray misses every sphere enlarged by 30%. Values are linear radiance.
import argparse, json, math, os
import numpy as np

def read_pfm(p):
    with open(p, "rb") as f:
        assert f.readline().strip() == b"PF"; w, h = map(int, f.readline().split()); s = float(f.readline())
        return np.frombuffer(f.read(), dtype="<f4" if s < 0 else ">f4").reshape(h, w, 3)[::-1].astype(np.float64)

def srgb(x):
    x = np.clip(x, 0, 1); return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)

def lum(x): return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]

def masks(sd, w, h):
    cam = json.load(open(os.path.join(sd, "camera.json"))); sp = json.load(open(os.path.join(sd, "spheres.json")))
    c2w = np.array(cam["c2w_blender_zup"]).reshape(4, 4); fx = cam["fx"] * w / cam["width"]; o = c2w[:3, 3]   # fx scales with the render width
    def hits(px, py, rad):
        d = np.stack([(px - w / 2) / fx, -(py - h / 2) / fx, -np.ones_like(px)], -1) @ c2w[:3, :3].T
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        out = []
        for c in sp["centers"]:
            oc = o - np.array(c); b = d @ oc; cc = oc @ oc - rad * rad; out.append(b * b - cc > 0)
        return out
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    corner = [hits(xs + dx, ys + dy, sp["radius"]) for dx in (0, 1) for dy in (0, 1)]
    m = [np.logical_and.reduce([corner[c][k] for c in range(4)]) for k in range(len(sp["centers"]))]
    big = hits(xs + 0.5, ys + 0.5, sp["radius"] * 1.3); bg = ~np.logical_or.reduce(big)
    return m, bg, sp["names"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("scene"); ap.add_argument("ours"); ap.add_argument("ref"); ap.add_argument("--json")
    a = ap.parse_args()
    A, B = read_pfm(a.ours), read_pfm(a.ref); h, w = A.shape[:2]
    ms, bg, names = masks(a.scene, w, h)
    rows = []
    for name, m in zip(names, ms):
        la, lb = lum(A[m]).mean(), lum(B[m]).mean()
        ca, cb = A[m].mean(0), B[m].mean(0)
        rows.append(dict(sphere=name, pixels=int(m.sum()), mean_lum_ours=la, mean_lum_ref=lb, rel_diff=la / lb - 1,
                         rel_diff_rgb=[float(x) for x in ca / cb - 1]))
    mse_all = float(np.mean((srgb(A) - srgb(B)) ** 2))
    out = dict(spheres=rows, background=dict(pixels=int(bg.sum()), rel_diff=float(lum(A[bg]).mean() / lum(B[bg]).mean() - 1)),
               psnr_srgb_full=10 * math.log10(1 / mse_all), ours=a.ours, ref=a.ref)
    for r in rows: print("%-15s px %6d  ours %.5f  ref %.5f  rel %+.4f  rgb %s" % (r["sphere"], r["pixels"], r["mean_lum_ours"], r["mean_lum_ref"], r["rel_diff"], " ".join("%+.4f" % x for x in r["rel_diff_rgb"])))
    print("background rel %+.4f   full-image PSNR (sRGB) %.2f dB" % (out["background"]["rel_diff"], out["psnr_srgb_full"]))
    if a.json: json.dump(out, open(a.json, "w"), indent=1)
