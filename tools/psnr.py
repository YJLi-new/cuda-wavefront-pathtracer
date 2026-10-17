# psnr.py - PSNR between two images in tonemapped sRGB (clip [0,1] + sRGB OETF, exposure 0).
# usage: python psnr.py A B [--json]   (A/B: .png (display-referred 8-bit) or .pfm (linear float) or .exr (linear, needs imageio/OpenEXR))
import sys, json, math, numpy as np
def srgb(x):
    x = np.clip(x, 0, 1); return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
def load(p):
    if p.lower().endswith('.png') or p.lower().endswith('.jpg'):
        from PIL import Image; return np.asarray(Image.open(p).convert('RGB'), dtype=np.float64) / 255.0, 'display'
    if p.lower().endswith('.pfm'):
        with open(p, 'rb') as f:
            assert f.readline().strip() == b'PF'; w, h = map(int, f.readline().split()); s = float(f.readline())
            d = np.frombuffer(f.read(), dtype='<f4' if s < 0 else '>f4').reshape(h, w, 3)[::-1].astype(np.float64)
        return d, 'linear'
    if p.lower().endswith('.exr'):
        try:
            import imageio.v3 as iio; d = np.asarray(iio.imread(p), dtype=np.float64)[..., :3]; return d, 'linear'
        except Exception:
            import OpenEXR, Imath; f = OpenEXR.InputFile(p); dw = f.header()['dataWindow']; w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
            ch = [np.frombuffer(f.channel(c, Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape(h, w) for c in 'RGB']
            return np.stack(ch, -1).astype(np.float64), 'linear'
    raise SystemExit('unknown format ' + p)
a, ka = load(sys.argv[1]); b, kb = load(sys.argv[2])
if ka == 'linear': a = srgb(a)
if kb == 'linear': b = srgb(b)
if a.shape != b.shape: raise SystemExit('shape mismatch %s vs %s' % (a.shape, b.shape))
mse = float(np.mean((a - b) ** 2)); psnr = 10 * math.log10(1 / mse) if mse > 0 else 99.0
out = dict(a=sys.argv[1], b=sys.argv[2], psnr_db=round(psnr, 2), mse=mse, mean_a=[round(float(x), 4) for x in a.reshape(-1, 3).mean(0)], mean_b=[round(float(x), 4) for x in b.reshape(-1, 3).mean(0)], tonemap='clip[0,1]+sRGB OETF, exposure 0')
print(json.dumps(out))
