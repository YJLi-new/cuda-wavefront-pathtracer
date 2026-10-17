# cycles_scene_reference.py - Blender Cycles reference for a pt scene file (--scene: meshes + instances) under an HDRI.
#   python cycles_scene_reference.py -- SCENE.json CAMERA.json LIGHTING.json OUT_PREFIX [--spp 1024] [--bounces 8]
#          [--device OPTIX] [--time 1]
# Meshes: OBJ files are rebuilt from their vertices / faces (vertex normals when there is one per vertex, as pt reads
# them); with a "materials" JSON every entry maps to a Principled BSDF item by item as in tests/cycles_reference.py,
# otherwise each MTL Kd colour becomes a Diffuse BSDF (pt: Lambert). glTF files go through Blender's glTF importer
# (defaults, every Principled BSDF set to GGX), as in tests/cycles_gltf_reference.py. Each pt instance becomes one
# Blender object per mesh part that shares the mesh data (Cycles instancing), with the instance matrix T * R * S of
# scene/instances.cpp (frame 0). World, camera and render settings as tests/cycles_reference.py: Environment Texture
# (Closest) x strength, RR off, no clamping, 1 px box filter, no denoising / adaptive sampling. Writes OUT_PREFIX.pfm.
# --time 1: persistent data on, two renders, both timed with a wall clock around bpy.ops.render.render (report the second).
import bpy, sys, json, math, os, time
import numpy as np
import addon_utils
from mathutils import Matrix

argv = sys.argv[sys.argv.index("--") + 1:]
SCENE, CAM, LIGHT, OUT = argv[0], argv[1], argv[2], argv[3]
opt = dict(zip(argv[4::2], argv[5::2]))
SPP = int(opt.get("--spp", 1024)); BOUNCES = int(opt.get("--bounces", 8)); DEV = opt.get("--device", "OPTIX")
scene_js = json.load(open(SCENE)); cam_js = json.load(open(CAM)); light_js = json.load(open(LIGHT))
W, H = cam_js["width"], cam_js["height"]
SD = os.path.dirname(os.path.abspath(SCENE))
rel = lambda p: p if (not p or os.path.isabs(p)) else os.path.join(SD, p)

bpy.ops.wm.read_factory_settings(use_empty=True)
addon_utils.enable("io_scene_gltf2", default_set=True)
sc = bpy.context.scene

def principled(m):
    mat = bpy.data.materials.new(m.get("name", "mat")); mat.use_nodes = True; p = mat.node_tree.nodes["Principled BSDF"]; p.distribution = "GGX"
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
    return mat

def lambert(name, kd):
    mat = bpy.data.materials.new(name); mat.use_nodes = True; nt = mat.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    d = nt.nodes.new("ShaderNodeBsdfDiffuse"); d.inputs["Color"].default_value = (*kd, 1.0); d.inputs["Roughness"].default_value = 0.0
    o = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(d.outputs["BSDF"], o.inputs["Surface"])
    return mat

def load_obj(path, mats_path):
    V, N, F, FM, names, kd, mtl = [], [], [], [], [], {}, None
    cur = 0
    for line in open(path):
        t = line.split()
        if not t: continue
        if t[0] == "v": V.append([float(x) for x in t[1:4]])
        elif t[0] == "vn": N.append([float(x) for x in t[1:4]])
        elif t[0] == "mtllib": mtl = os.path.join(os.path.dirname(path), t[1])
        elif t[0] == "usemtl":
            if t[1] not in names: names.append(t[1])
            cur = names.index(t[1])
        elif t[0] == "f":
            ix = [int(x.split("/")[0]) for x in t[1:]]; ix = [i - 1 if i > 0 else len(V) + i for i in ix]
            for k in range(1, len(ix) - 1): F.append([ix[0], ix[k], ix[k + 1]]); FM.append(cur)
    if mtl and os.path.exists(mtl):
        nm = None
        for line in open(mtl):
            t = line.split()
            if t and t[0] == "newmtl": nm = t[1]
            elif t and t[0] == "Kd" and nm: kd[nm] = [float(x) for x in t[1:4]]
    me = bpy.data.meshes.new(os.path.basename(path)); me.from_pydata(V, [], F); me.update()
    me.polygons.foreach_set("material_index", np.array(FM, dtype=np.int32))
    if len(N) == len(V):
        me.shade_smooth(); me.normals_split_custom_set_from_vertices([tuple(n) for n in N])
    if not names: names = ["default"]
    mj = json.load(open(mats_path))["materials"] if mats_path else None
    for i, nm in enumerate(names):
        if mj is not None:
            e = next((m for m in mj if m.get("name") == nm), mj[i] if i < len(mj) else {})
            me.materials.append(principled(e))
        else:
            me.materials.append(lambert(nm, kd.get(nm, [0.8, 0.8, 0.8])))
    return [me]

def load_gltf(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    parts = []
    for o in new:
        if o.type != "MESH": continue
        me = o.data.copy(); me.transform(o.matrix_world); parts.append(me)
        for s in o.material_slots:
            if s.material and s.material.use_nodes:
                for n in s.material.node_tree.nodes:
                    if n.type == "BSDF_PRINCIPLED": n.distribution = "GGX"
    for o in new: bpy.data.objects.remove(o, do_unlink=True)
    return parts

meshes = []
for m in scene_js["meshes"]:
    f = rel(m["file"])
    meshes.append(load_gltf(f) if f.lower().endswith((".gltf", ".glb")) else load_obj(f, rel(m.get("materials", ""))))
    print("mesh", os.path.basename(f), "parts", len(meshes[-1]), "triangles", sum(len(p.polygons) for p in meshes[-1]))

def inst_matrix(I):
    if "transform" in I:
        v = I["transform"]; M = np.eye(4); M[:3, :4] = np.array(v[:12]).reshape(3, 4); return M
    t = np.array(I.get("translate", [0, 0, 0]), float); s = I.get("scale", 1.0); s = np.array([s] * 3 if np.isscalar(s) else s, float)
    a = np.array(I.get("axis", [0, 0, 1]), float); l = np.linalg.norm(a); a = a / l if l > 0 else np.array([0, 0, 1.0])
    g = math.radians(I.get("angle_deg", 0.0)); c, sn = math.cos(g), math.sin(g); C = 1 - c; x, y, z = a
    R = np.array([[c + x * x * C, x * y * C - z * sn, x * z * C + y * sn], [y * x * C + z * sn, c + y * y * C, y * z * C - x * sn], [z * x * C - y * sn, z * y * C + x * sn, c + z * z * C]])
    M = np.eye(4); M[:3, :3] = R * s[None, :]; M[:3, 3] = t; return M

for i, I in enumerate(scene_js["instances"]):
    M = Matrix(inst_matrix(I).tolist())
    for k, me in enumerate(meshes[I["mesh"]]):
        ob = bpy.data.objects.new("inst%d_%d" % (i, k), me); sc.collection.objects.link(ob); ob.matrix_world = M
print("instances", len(scene_js["instances"]), "objects", len([o for o in sc.objects if o.type == "MESH"]))

# ---- world (environment only) ----
assert light_js.get("sun_irradiance_W_m2", 0.0) == 0.0 and abs(light_js.get("env_rotation_deg", 0.0)) < 1e-9
w = bpy.data.worlds.new("env"); w.use_nodes = True; nt = w.node_tree
env = nt.nodes.new("ShaderNodeTexEnvironment"); env.image = bpy.data.images.load(light_js["env_map"])
env.interpolation = "Closest"; env.projection = "EQUIRECTANGULAR"
bg = nt.nodes["Background"]; bg.inputs["Strength"].default_value = light_js.get("env_strength", 1.0)
nt.links.new(env.outputs["Color"], bg.inputs["Color"])
sc.world = w

# ---- camera ----
cd = bpy.data.cameras.new("cam"); cd.sensor_fit = "HORIZONTAL"; cd.sensor_width = 36.0; cd.lens = cam_js["fx"] * 36.0 / W; cd.clip_start = 0.001
co = bpy.data.objects.new("cam", cd); sc.collection.objects.link(co); sc.camera = co
co.matrix_world = Matrix(np.array(cam_js["c2w_blender_zup"]).reshape(4, 4).tolist())

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
sc.render.filepath = OUT + ".exr"
if opt.get("--time", "0") == "1":
    sc.render.use_persistent_data = True; ts = []
    for _ in range(2):
        t0 = time.perf_counter(); bpy.ops.render.render(write_still=False); ts.append(time.perf_counter() - t0)
    bpy.data.images["Render Result"].save_render(sc.render.filepath)
    print("cycles_time_s render1 %.4f render2 %.4f" % (ts[0], ts[1]))
else:
    bpy.ops.render.render(write_still=True)
im = bpy.data.images.load(sc.render.filepath)
a = np.array(im.pixels[:], dtype=np.float32).reshape(im.size[1], im.size[0], 4)[::-1, :, :3]
with open(OUT + ".pfm", "wb") as f:
    f.write(b"PF\n%d %d\n-1.0\n" % (W, H)); f.write(np.ascontiguousarray(a[::-1]).astype("<f4").tobytes())
os.remove(sc.render.filepath)
print("wrote", OUT + ".pfm", "mean", a.reshape(-1, 3).mean(0))
