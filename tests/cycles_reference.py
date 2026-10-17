# cycles_reference.py - renders a pt scene (OBJ + materials.json + camera.json + lighting.json) with Blender Cycles,
# mapping every parameter item by item, and writes linear radiance as PFM.
#   blender -b -P cycles_reference.py -- SCENE_DIR OUT_PREFIX [--spp 1024] [--bounces 8] [--device OPTIX] [--envcheck 1] [--time 1]
# --time 1: persistent data on, the scene is rendered twice and both renders are timed with a wall clock around
#           bpy.ops.render.render (the second render, with the BVH and kernels reused, is the timing we report).
# Mapping: Principled BSDF (multiscatter off: distribution GGX), base_color -> Base Color, metallic -> Metallic,
# roughness -> Roughness, ior -> IOR, specular_weight w -> Specular IOR Level 0.5*w (0 disables, 0.5 = plain IOR),
# specular_color -> Specular Tint (F82 tint for metals), transmission -> Transmission Weight,
# emission_color/strength -> Emission Color/Strength. World: Environment Texture (Closest interpolation) x strength.
# Russian roulette off (min_light_bounces = bounces), no clamping, 1 px box filter, no denoising/adaptive sampling.
import bpy, sys, json, math, os, time
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
SD, OUT = argv[0], argv[1]
opt = dict(zip(argv[2::2], argv[3::2]))
SPP = int(opt.get("--spp", 1024)); BOUNCES = int(opt.get("--bounces", 8)); DEV = opt.get("--device", "OPTIX")
cam_js = json.load(open(os.path.join(SD, "camera.json"))); light_js = json.load(open(os.path.join(SD, "lighting.json")))
mats_js = json.load(open(os.path.join(SD, "materials.json")))["materials"]
W, H = cam_js["width"], cam_js["height"]

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene

# ---- mesh (same vertices, faces and vertex normals as pt reads) ----
V, N, F, FM, names = [], [], [], [], []
cur = 0
for line in open(os.path.join(SD, "spheres.obj")):
    t = line.split()
    if not t: continue
    if t[0] == "v": V.append([float(x) for x in t[1:4]])
    elif t[0] == "vn": N.append([float(x) for x in t[1:4]])
    elif t[0] == "usemtl": names.append(t[1]); cur = len(names) - 1
    elif t[0] == "f": F.append([int(x.split("/")[0]) - 1 for x in t[1:4]]); FM.append(cur)
me = bpy.data.meshes.new("scene"); me.from_pydata(V, [], F); me.update()
me.polygons.foreach_set("material_index", np.array(FM, dtype=np.int32))
me.shade_smooth()
me.normals_split_custom_set_from_vertices([tuple(n) for n in N])
for m in mats_js:
    mat = bpy.data.materials.new(m["name"]); mat.use_nodes = True; nt = mat.node_tree
    p = nt.nodes["Principled BSDF"]; p.distribution = "GGX"
    g = lambda k, d: m.get(k, d)
    p.inputs["Base Color"].default_value = (*g("base_color", [0.8] * 3), 1.0)
    p.inputs["Metallic"].default_value = g("metallic", 0.0)
    p.inputs["Roughness"].default_value = g("roughness", 0.5)
    p.inputs["IOR"].default_value = g("ior", 1.5)
    p.inputs["Specular IOR Level"].default_value = 0.5 * g("specular_weight", 1.0)
    p.inputs["Specular Tint"].default_value = (*g("specular_color", [1.0] * 3), 1.0)
    p.inputs["Transmission Weight"].default_value = g("transmission", 0.0)
    p.inputs["Emission Color"].default_value = (*g("emission_color", [1.0] * 3), 1.0)
    p.inputs["Emission Strength"].default_value = g("emission_strength", 0.0)
    p.inputs["Coat Weight"].default_value = 0.0; p.inputs["Sheen Weight"].default_value = 0.0; p.inputs["Subsurface Weight"].default_value = 0.0
    me.materials.append(mat)
ob = bpy.data.objects.new("scene", me); sc.collection.objects.link(ob)

# ---- world ----
w = bpy.data.worlds.new("env"); w.use_nodes = True; nt = w.node_tree
env = nt.nodes.new("ShaderNodeTexEnvironment"); img = bpy.data.images.load(light_js["env_map"]); env.image = img
env.interpolation = "Closest"; env.projection = "EQUIRECTANGULAR"
bg = nt.nodes["Background"]; bg.inputs["Strength"].default_value = light_js.get("env_strength", 1.0)
nt.links.new(env.outputs["Color"], bg.inputs["Color"])
assert abs(light_js.get("env_rotation_deg", 0.0)) < 1e-9
sc.world = w

# ---- camera ----
cd = bpy.data.cameras.new("cam"); cd.sensor_fit = "HORIZONTAL"; cd.sensor_width = 36.0; cd.lens = cam_js["fx"] * 36.0 / W; cd.clip_start = 0.01
co = bpy.data.objects.new("cam", cd); sc.collection.objects.link(co); sc.camera = co
from mathutils import Matrix
co.matrix_world = Matrix(np.array(cam_js["c2w_blender_zup"]).reshape(4, 4).tolist())

# ---- render settings ----
sc.render.engine = "CYCLES"; sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
cy = sc.cycles
cy.samples = SPP; cy.use_adaptive_sampling = False; cy.use_denoising = False; cy.seed = 1000
cy.max_bounces = BOUNCES; cy.diffuse_bounces = BOUNCES; cy.glossy_bounces = BOUNCES; cy.transmission_bounces = BOUNCES
cy.transparent_max_bounces = BOUNCES; cy.volume_bounces = 0
cy.min_light_bounces = BOUNCES; cy.min_transparent_bounces = BOUNCES   # no Russian roulette within the limits
cy.sample_clamp_direct = 0.0; cy.sample_clamp_indirect = 0.0; cy.blur_glossy = 0.0
cy.caustics_reflective = True; cy.caustics_refractive = True; cy.light_sampling_threshold = 0.0
cy.pixel_filter_type = "BOX"; cy.filter_width = 1.0
sc.view_settings.view_transform = "Standard"; sc.view_settings.exposure = 0.0; sc.view_settings.look = "None"
sc.render.use_compositing = False; sc.render.use_sequencer = False
prefs = bpy.context.preferences.addons["cycles"].preferences; prefs.compute_device_type = DEV; prefs.get_devices()
for d in prefs.devices: d.use = d.type == DEV
cy.device = "GPU"
print("devices", [d.name for d in prefs.devices if d.use])
sc.render.image_settings.file_format = "OPEN_EXR"; sc.render.image_settings.color_depth = "32"
sc.render.filepath = OUT + ".exr"
if opt.get("--time", "0") == "1":
    sc.render.use_persistent_data = True; ts = []
    for _ in range(2):
        t0 = time.perf_counter(); bpy.ops.render.render(write_still=False); ts.append(time.perf_counter() - t0)
    bpy.data.images["Render Result"].save_render(sc.render.filepath)
    print("cycles_time_s render1 %.4f render2 %.4f" % (ts[0], ts[1]))
else:
    bpy.ops.render.render(write_still=True)

def load_rgb(path):
    im = bpy.data.images.load(path); a = np.array(im.pixels[:], dtype=np.float32).reshape(im.size[1], im.size[0], 4)[::-1, :, :3]
    return np.ascontiguousarray(a)
rad = load_rgb(OUT + ".exr")
with open(OUT + ".pfm", "wb") as f:
    f.write(b"PF\n%d %d\n-1.0\n" % (W, H)); f.write(rad[::-1].astype("<f4").tobytes())
print("wrote", OUT + ".pfm", "mean", rad.reshape(-1, 3).mean(0))

# ---- environment-map orientation check: Cycles background pixels vs candidate equirect conventions ----
if opt.get("--envcheck", "0") == "1":
    hdr = np.array(img.pixels[:], dtype=np.float32).reshape(img.size[1], img.size[0], 4)[::-1, :, :3]   # row 0 = top of file
    print("hdri decoded by Blender: %dx%d mean %s texel[0,0] %s texel[100,200] %s" % (img.size[0], img.size[1], hdr.reshape(-1, 3).mean(0), hdr[0, 0], hdr[100, 200]))
    sp = json.load(open(os.path.join(SD, "spheres.json")))
    c2w = np.array(cam_js["c2w_blender_zup"]).reshape(4, 4); fx = cam_js["fx"]
    ys, xs = np.mgrid[0:H, 0:W]
    d = np.stack([(xs + 0.5 - W / 2) / fx, -(ys + 0.5 - H / 2) / fx, -np.ones_like(xs, dtype=np.float64)], -1) @ c2w[:3, :3].T
    d /= np.linalg.norm(d, axis=-1, keepdims=True); o = c2w[:3, 3]
    hit = np.zeros((H, W), bool)
    for c in sp["centers"]:
        oc = o - np.array(c); b = d @ oc; cc = oc @ oc - (sp["radius"] * 1.3) ** 2; hit |= (b * b - cc) > 0
    bgm = ~hit; th = np.arccos(np.clip(d[..., 2], -1, 1)); ph = np.arctan2(d[..., 1], d[..., 0])
    hh, ww = hdr.shape[:2]; row = np.clip((th / math.pi * hh).astype(int), 0, hh - 1)
    res = []
    for s in (1, -1):
        for u0 in (0.0, 0.25, 0.5, 0.75):
            u = (u0 + s * ph / (2 * math.pi)) % 1.0; col = np.clip((u * ww).astype(int), 0, ww - 1)
            e = np.log1p(hdr[row, col][bgm]) - np.log1p(rad[bgm]); res.append((float(np.mean(e * e)), s, u0, float(hdr[row, col][bgm].mean() / rad[bgm].mean())))
    res.sort()
    for r in res[:3]: print("envcheck mse_log %.6f sign %d u0 %.2f mean_ratio %.5f" % r)
