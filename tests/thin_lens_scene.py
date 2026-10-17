#!/usr/bin/env python3
# thin_lens_scene.py - focus test scene: three slanted-edge cards (left half dark, right half bright, edge tilted 5 deg
# from vertical) facing the camera at three depths, lit by a constant white sky.
#   python thin_lens_scene.py OUTDIR [--width 960 --height 320 --hfov 60 --depths 4,6,9]
# Camera at the origin looking along +y (z up). Card k is centred at image column (2k + 1) W / 6 on the middle row and
# spans 27% of the image width. Outputs OUTDIR/{cards.obj, cards.mtl, camera.json, lighting.json, cards.json}.
import argparse, json, math, os

ap = argparse.ArgumentParser(); ap.add_argument("outdir")
ap.add_argument("--width", type=int, default=960); ap.add_argument("--height", type=int, default=320)
ap.add_argument("--hfov", type=float, default=60.0); ap.add_argument("--depths", default="4,6,9")
ap.add_argument("--tilt", type=float, default=5.0); ap.add_argument("--dark", type=float, default=0.05); ap.add_argument("--bright", type=float, default=0.85)
a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
W, H = a.width, a.height; fx = W * 0.5 / math.tan(math.radians(a.hfov) * 0.5)
depths = [float(x) for x in a.depths.split(",")]; ta = math.tan(math.radians(a.tilt))
V, F, cards = [], [], []
for k, z in enumerate(depths):
    cxp = (2 * k + 1) * W / 6.0; S = 0.27 * W
    X = (cxp - W / 2) / fx * z; hs = 0.5 * S / fx * z            # card centre (world x) and half size at depth z
    xb, xt = X - hs * ta, X + hs * ta                              # edge x at the bottom / top of the card
    quads = {"dark": [(X - hs, -hs), (xb, -hs), (xt, hs), (X - hs, hs)], "bright": [(xb, -hs), (X + hs, -hs), (X + hs, hs), (xt, hs)]}
    for mat, q in quads.items():
        b = len(V); V += [(x, z, zz) for x, zz in q]
        F.append((mat, (b, b + 1, b + 2))); F.append((mat, (b, b + 2, b + 3)))   # normal -y: faces the camera
    cards.append(dict(depth=z, center_px=[cxp, H / 2.0], size_px=S, tilt_deg=a.tilt))
with open(os.path.join(a.outdir, "cards.mtl"), "w") as f:
    f.write("newmtl dark\nKd %g %g %g\nnewmtl bright\nKd %g %g %g\n" % ((a.dark,) * 3 + (a.bright,) * 3))
with open(os.path.join(a.outdir, "cards.obj"), "w") as f:
    f.write("mtllib cards.mtl\n")
    for v in V: f.write("v %.7f %.7f %.7f\n" % v)
    cur = None
    for mat, t in F:
        if mat != cur: f.write("usemtl %s\n" % mat); cur = mat
        f.write("f %d %d %d\n" % (t[0] + 1, t[1] + 1, t[2] + 1))
c2w = [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1]   # columns: x = +x, y (up) = +z, z = -y (looks along +y)
json.dump({"width": W, "height": H, "fx": fx, "fy": fx, "hfov_deg": a.hfov, "c2w_blender_zup": c2w}, open(os.path.join(a.outdir, "camera.json"), "w"), indent=1)
json.dump({"sun_irradiance_W_m2": 0.0, "sky_radiance_rgb_linear": [1, 1, 1]}, open(os.path.join(a.outdir, "lighting.json"), "w"), indent=1)
json.dump({"fx": fx, "width": W, "height": H, "cards": cards}, open(os.path.join(a.outdir, "cards.json"), "w"), indent=1)
print("wrote", a.outdir, len(V), "verts", len(F), "tris, fx %.3f" % fx)
