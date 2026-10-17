# cycles_gltf_reference.py - Blender Cycles reference for a glTF model under an HDRI (run with Blender's Python / bpy).
#   python cycles_gltf_reference.py -- MODEL.gltf OUTDIR --hdri ENV.hdr [--spp 1024] [--bounces 8] [--width 512]
#          [--height 512] [--hfov 40] [--az -60] [--el 20] [--device OPTIX]
# Imports the model with Blender's glTF importer, frames it automatically (bounding sphere of all mesh vertices, camera
# at azimuth/elevation looking at its centre) and writes OUTDIR/{camera.json, lighting.json} for pt, then renders
#   OUTDIR/cycles.pfm    linear radiance, settings as tests/cycles_reference.py: every Principled BSDF set to GGX
#                        (multiscatter off), world = Environment Texture (Closest) strength 1, RR off, no clamping,
#                        1 px box filter, no denoising / adaptive sampling
#   OUTDIR/coverage.pfm  per-pixel object coverage (all materials replaced by white emission, black world, 64 spp)
import bpy, sys, json, math, os
import numpy as np
import addon_utils

argv = sys.argv[sys.argv.index("--") + 1:]
MODEL, OD = argv[0], argv[1]
opt = dict(zip(argv[2::2], argv[3::2]))
SPP = int(opt.get("--spp", 1024)); BOUNCES = int(opt.get("--bounces", 8)); DEV = opt.get("--device", "OPTIX")
W, H = int(opt.get("--width", 512)), int(opt.get("--height", 512))
HFOV, AZ, EL = float(opt.get("--hfov", 40)), math.radians(float(opt.get("--az", -60))), math.radians(float(opt.get("--el", 20)))
os.makedirs(OD, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
addon_utils.enable("io_scene_gltf2", default_set=True)
sc = bpy.context.scene
bpy.ops.import_scene.gltf(filepath=MODEL)
meshes = [o for o in sc.objects if o.type == "MESH"]
nbsdf = 0
for mat in bpy.data.materials:
    if not mat.use_nodes: continue
    for n in mat.node_tree.nodes:
        if n.type == "BSDF_PRINCIPLED":
            n.distribution = "GGX"; nbsdf += 1
            print("material %s: specular IOR level %.3f IOR %.3f transmission %.3f coat %.3f sheen %.3f emission strength %.3f normal linked %s" % (
                mat.name, n.inputs["Specular IOR Level"].default_value, n.inputs["IOR"].default_value, n.inputs["Transmission Weight"].default_value,
                n.inputs["Coat Weight"].default_value, n.inputs["Sheen Weight"].default_value, n.inputs["Emission Strength"].default_value, n.inputs["Normal"].is_linked))
print("imported", MODEL, "mesh objects", len(meshes), "principled BSDFs", nbsdf)
# diagnostics (not used by the gate): --tex_interp Closest (all image textures), --no_normal 1 (normal maps unlinked),
# --spec_level X (Specular IOR Level of every Principled BSDF; 0 = no dielectric specular layer)
for mat in bpy.data.materials:
    if not mat.use_nodes: continue
    for n in mat.node_tree.nodes:
        if n.type == "TEX_IMAGE" and "--tex_interp" in opt: n.interpolation = opt["--tex_interp"]
        if n.type == "BSDF_PRINCIPLED" and "--spec_level" in opt: n.inputs["Specular IOR Level"].default_value = float(opt["--spec_level"])
        if n.type == "BSDF_PRINCIPLED" and opt.get("--no_normal", "0") == "1":
            for l in list(n.inputs["Normal"].links): mat.node_tree.links.remove(l)

# ---- automatic framing ----
pts = np.concatenate([np.array([tuple(o.matrix_world @ v.co) for v in o.data.vertices]) for o in meshes])
lo, hi = pts.min(0), pts.max(0); c = 0.5 * (lo + hi); r = float(np.linalg.norm(pts - c, axis=1).max())
dist = r / math.sin(math.radians(HFOV) * 0.5) * 1.02
pos = c + dist * np.array([math.cos(EL) * math.cos(AZ), math.cos(EL) * math.sin(AZ), math.sin(EL)])
fw = c - pos; fw /= np.linalg.norm(fw); rt = np.cross(fw, [0.0, 0.0, 1.0]); rt /= np.linalg.norm(rt); up = np.cross(rt, fw)
c2w = np.eye(4); c2w[:3, 0] = rt; c2w[:3, 1] = up; c2w[:3, 2] = -fw; c2w[:3, 3] = pos
fx = W * 0.5 / math.tan(math.radians(HFOV) * 0.5)
json.dump({"width": W, "height": H, "fx": fx, "fy": fx, "hfov_deg": HFOV, "c2w_blender_zup": c2w.reshape(-1).tolist(),
           "bbox_lo": lo.tolist(), "bbox_hi": hi.tolist(), "radius": r}, open(os.path.join(OD, "camera.json"), "w"), indent=1)
json.dump({"sun_irradiance_W_m2": 0.0, "sky_radiance_rgb_linear": [0, 0, 0], "env_map": os.path.abspath(opt["--hdri"]), "env_strength": 1.0, "env_rotation_deg": 0.0},
          open(os.path.join(OD, "lighting.json"), "w"), indent=1)

# ---- world ----
w = bpy.data.worlds.new("env"); w.use_nodes = True; nt = w.node_tree
env = nt.nodes.new("ShaderNodeTexEnvironment"); env.image = bpy.data.images.load(opt["--hdri"])
env.interpolation = "Closest"; env.projection = "EQUIRECTANGULAR"
bg = nt.nodes["Background"]; bg.inputs["Strength"].default_value = 1.0
nt.links.new(env.outputs["Color"], bg.inputs["Color"])
sc.world = w

# ---- camera ----
cd = bpy.data.cameras.new("cam"); cd.sensor_fit = "HORIZONTAL"; cd.sensor_width = 36.0; cd.lens = fx * 36.0 / W; cd.clip_start = 0.001
co = bpy.data.objects.new("cam", cd); sc.collection.objects.link(co); sc.camera = co
from mathutils import Matrix
co.matrix_world = Matrix(c2w.tolist())

# ---- render settings (as tests/cycles_reference.py) ----
sc.render.engine = "CYCLES"; sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
cy = sc.cycles
cy.samples = SPP; cy.use_adaptive_sampling = False; cy.use_denoising = False; cy.seed = 1000
cy.max_bounces = BOUNCES; cy.diffuse_bounces = BOUNCES; cy.glossy_bounces = BOUNCES; cy.transmission_bounces = BOUNCES
cy.transparent_max_bounces = BOUNCES; cy.volume_bounces = 0
cy.min_light_bounces = BOUNCES; cy.min_transparent_bounces = BOUNCES
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

def render_pfm(name):
    sc.render.filepath = os.path.join(OD, name + ".exr")
    bpy.ops.render.render(write_still=True)
    im = bpy.data.images.load(sc.render.filepath)
    a = np.array(im.pixels[:], dtype=np.float32).reshape(im.size[1], im.size[0], 4)[::-1, :, :3]   # row 0 = top
    with open(os.path.join(OD, name + ".pfm"), "wb") as f:
        f.write(b"PF\n%d %d\n-1.0\n" % (W, H)); f.write(np.ascontiguousarray(a[::-1]).astype("<f4").tobytes())
    os.remove(sc.render.filepath)
    print("wrote", name + ".pfm", "mean", a.reshape(-1, 3).mean(0))

render_pfm("cycles")

# ---- coverage: white emission everywhere, black world, camera rays only ----
em = bpy.data.materials.new("cov"); em.use_nodes = True; ent = em.node_tree
for n in list(ent.nodes): ent.nodes.remove(n)
e = ent.nodes.new("ShaderNodeEmission"); e.inputs["Strength"].default_value = 1.0; e.inputs["Color"].default_value = (1, 1, 1, 1)
out = ent.nodes.new("ShaderNodeOutputMaterial"); ent.links.new(e.outputs["Emission"], out.inputs["Surface"])
for o in meshes:
    for s in o.material_slots: s.material = em
    if not o.material_slots: o.data.materials.append(em)
bg.inputs["Strength"].default_value = 0.0
cy.samples = 64; cy.max_bounces = 0; cy.min_light_bounces = 0
render_pfm("coverage")
