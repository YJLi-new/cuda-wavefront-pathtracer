#!/usr/bin/env python3
# material_spheres.py - writes the material-sphere test scene used by the Cycles comparison and the MIS test:
# five UV spheres (diffuse, rough conductor, rough glass, glossy-diffuse, emissive) in a row, lit by an HDRI.
#   python material_spheres.py OUTDIR HDRI.hdr [--width 640 --height 256]
# Outputs OUTDIR/{spheres.obj, spheres.mtl, materials.json, camera.json, lighting.json, spheres.json}
import argparse, json, math, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("outdir"); ap.add_argument("hdri")
ap.add_argument("--width", type=int, default=640); ap.add_argument("--height", type=int, default=256)
ap.add_argument("--nu", type=int, default=192); ap.add_argument("--nv", type=int, default=96)
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)

MATS = [  # OpenPBR-style parameters (see README "Materials"); Blender mapping in cycles_reference.py
    dict(name="diffuse", base_color=[0.8, 0.8, 0.8], specular_weight=0.0, roughness=0.5),
    dict(name="conductor", base_color=[0.95, 0.64, 0.54], metallic=1.0, roughness=0.3, specular_color=[1, 1, 1]),
    dict(name="rough_glass", base_color=[1, 1, 1], transmission=1.0, roughness=0.3, ior=1.5),
    dict(name="glossy_diffuse", base_color=[0.2, 0.4, 0.8], roughness=0.3, ior=1.5, specular_weight=1.0),
    dict(name="emissive", base_color=[0.5, 0.5, 0.5], specular_weight=0.0, roughness=0.5, emission_color=[1.0, 0.6, 0.3], emission_strength=3.0),
]
R = 1.0; CENTERS = [[x, 0.0, 0.0] for x in (-5.0, -2.5, 0.0, 2.5, 5.0)]

v, vn, faces = [], [], []
for k, c in enumerate(CENTERS):
    base = len(v)
    for j in range(a.nv + 1):
        th = math.pi * j / a.nv
        for i in range(a.nu):
            ph = 2 * math.pi * i / a.nu
            d = (math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th))
            v.append([c[0] + R * d[0], c[1] + R * d[1], c[2] + R * d[2]]); vn.append(d)
    fk = []
    for j in range(a.nv):
        for i in range(a.nu):
            i1 = (i + 1) % a.nu
            p00, p01, p10, p11 = base + j * a.nu + i, base + j * a.nu + i1, base + (j + 1) * a.nu + i, base + (j + 1) * a.nu + i1
            # outward winding (counter-clockwise seen from outside)
            if j > 0: fk.append((p00, p10, p01))
            if j < a.nv - 1: fk.append((p01, p10, p11))
    faces.append(fk)

with open(os.path.join(a.outdir, "spheres.mtl"), "w") as f:
    for m in MATS: f.write("newmtl %s\nKd %g %g %g\n" % (m["name"], *m["base_color"]))
with open(os.path.join(a.outdir, "spheres.obj"), "w") as f:
    f.write("mtllib spheres.mtl\n")
    for p in v: f.write("v %.7f %.7f %.7f\n" % tuple(p))
    for n in vn: f.write("vn %.7f %.7f %.7f\n" % tuple(n))
    for k, fk in enumerate(faces):
        f.write("usemtl %s\n" % MATS[k]["name"])
        for t in fk: f.write("f %d//%d %d//%d %d//%d\n" % (t[0] + 1, t[0] + 1, t[1] + 1, t[1] + 1, t[2] + 1, t[2] + 1))
json.dump({"materials": MATS}, open(os.path.join(a.outdir, "materials.json"), "w"), indent=1)

# camera: looks at the origin from -y, z-up world, OpenGL camera axes, row-major c2w
pos = np.array([0.0, -12.0, 1.5]); tgt = np.array([0.0, 0.0, 0.0]); up = np.array([0.0, 0.0, 1.0])
fw = tgt - pos; fw /= np.linalg.norm(fw); r = np.cross(fw, up); r /= np.linalg.norm(r); u = np.cross(r, fw)
c2w = np.eye(4); c2w[:3, 0] = r; c2w[:3, 1] = u; c2w[:3, 2] = -fw; c2w[:3, 3] = pos
hfov = 60.0; fx = a.width * 0.5 / math.tan(math.radians(hfov) * 0.5)
json.dump({"width": a.width, "height": a.height, "fx": fx, "fy": fx, "hfov_deg": hfov, "c2w_blender_zup": c2w.reshape(-1).tolist()},
          open(os.path.join(a.outdir, "camera.json"), "w"), indent=1)
json.dump({"sun_irradiance_W_m2": 0.0, "sky_radiance_rgb_linear": [0, 0, 0], "env_map": os.path.abspath(a.hdri), "env_strength": 1.0, "env_rotation_deg": 0.0},
          open(os.path.join(a.outdir, "lighting.json"), "w"), indent=1)
json.dump({"radius": R, "centers": CENTERS, "names": [m["name"] for m in MATS]}, open(os.path.join(a.outdir, "spheres.json"), "w"), indent=1)
print("wrote", a.outdir, len(v), "verts", sum(len(fk) for fk in faces), "tris")
