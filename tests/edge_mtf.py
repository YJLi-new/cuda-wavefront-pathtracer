#!/usr/bin/env python3
# edge_mtf.py - slanted-edge sharpness readout for the thin-lens focus scene (tests/thin_lens_scene.py).
#   python edge_mtf.py SCENE_DIR pinhole.pfm lens.pfm --aperture R --focus D [--json out.json]
# Per card and image: edge position per row = centroid of the row's derivative, a line fit over the rows, then the
# edge-spread function (ESF) of all ROI pixels binned at 0.1 px of signed distance to the line (ISO 12233-style
# super-resolution), its 10-90% rise width and MTF50 (Hann-windowed FFT of the line-spread function).
# Thin-lens prediction for the lens image: circle of confusion D_coc = 2 R fx |1/D - 1/z| px for a card at depth z;
# predicted ESF = pinhole ESF of the same card convolved with the edge response of a uniform disk of diameter D_coc.
import argparse, json, math
import numpy as np

def read_pfm(p):
    with open(p, "rb") as f:
        assert f.readline().strip() == b"PF"; w, h = map(int, f.readline().split()); s = float(f.readline())
        return np.frombuffer(f.read(), dtype="<f4" if s < 0 else ">f4").reshape(h, w, 3)[::-1].astype(np.float64)

BIN = 0.1

def width1090(s, e):
    m = np.maximum.accumulate(e)   # monotone envelope for the crossing search
    return float(np.interp(0.9, m, s) - np.interp(0.1, m, s))

def mtf50(e):
    lsf = np.gradient(e) * np.hanning(len(e)); M = np.abs(np.fft.rfft(lsf)); M /= M[0]
    f = np.fft.rfftfreq(len(e), d=BIN); k = int(np.argmax(M < 0.5))
    return float(f[k - 1] + (M[k - 1] - 0.5) / (M[k - 1] - M[k]) * (f[k] - f[k - 1])) if k > 0 else float("nan")

def esf(L, card, hw):
    cx, cy = card["center_px"]; S = card["size_px"]
    y0, y1 = int(cy - 0.3 * S), int(cy + 0.3 * S); x0, x1 = int(cx - hw), int(cx + hw)
    R = L[y0:y1, x0:x1]
    lo, hi = np.median(R[:, :4]), np.median(R[:, -4:]); P = (R - lo) / (hi - lo)
    xs = np.arange(x0, x1) + 0.5; ys = np.arange(y0, y1) + 0.5
    d = np.diff(P, axis=1); xm = 0.5 * (xs[1:] + xs[:-1])
    xe = (d * xm).sum(1) / d.sum(1)
    b, a = np.polyfit(ys, xe, 1)                          # x_edge = a + b y
    sd = (xs[None, :] - (a + b * ys[:, None])) / math.sqrt(1 + b * b)
    lim = hw - 6; edges = np.arange(-lim, lim + 1e-9, BIN); idx = np.digitize(sd.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    e = np.bincount(idx[ok], weights=P.ravel()[ok], minlength=len(edges) - 1) / np.maximum(np.bincount(idx[ok], minlength=len(edges) - 1), 1)
    s = 0.5 * (edges[1:] + edges[:-1])
    return s, e, dict(tilt_deg=float(math.degrees(math.atan(b))), rows=int(y1 - y0), level_dark=float(lo), level_bright=float(hi))

def disk_kernel(D):
    r = 0.5 * D; t = np.arange(-math.ceil(r / BIN), math.ceil(r / BIN) + 1) * BIN
    k = np.sqrt(np.clip(r * r - t * t, 0, None)); return k / k.sum()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("scene"); ap.add_argument("pinhole"); ap.add_argument("lens")
    ap.add_argument("--aperture", type=float, required=True); ap.add_argument("--focus", type=float, required=True); ap.add_argument("--json")
    a = ap.parse_args()
    sc = json.load(open(a.scene + "/cards.json")); fx = sc["fx"]
    lum = lambda im: 0.2126 * im[..., 0] + 0.7152 * im[..., 1] + 0.0722 * im[..., 2]
    Lp, Ll = lum(read_pfm(a.pinhole)), lum(read_pfm(a.lens))
    cocs = [2 * a.aperture * fx * abs(1 / a.focus - 1 / c["depth"]) for c in sc["cards"]]
    hw = int(math.ceil(0.5 * max(cocs) + 20))
    rows = []
    for c, D in zip(sc["cards"], cocs):
        sp, ep, ip = esf(Lp, c, hw); sl, el, il = esf(Ll, c, hw)
        r = dict(depth=c["depth"], coc_px=D, in_focus=abs(c["depth"] - a.focus) < 1e-6,
                 pinhole=dict(w1090_px=width1090(sp, ep), mtf50_cyc_px=mtf50(ep), **ip), lens=dict(w1090_px=width1090(sl, el), mtf50_cyc_px=mtf50(el), **il))
        if D > 0:
            k = disk_kernel(D); pad = len(k)
            epad = np.concatenate([np.zeros(pad), ep, np.ones(pad)]); pred = np.convolve(epad, k, mode="same")[pad:-pad]
            r["predicted_w1090_px"] = width1090(sp, pred); r["w1090_ratio_meas_pred"] = r["lens"]["w1090_px"] / r["predicted_w1090_px"]
        rows.append(r)
        print("card z=%.1f%s  CoC %.2f px | pinhole w10-90 %.3f px MTF50 %.4f | lens w10-90 %.3f px MTF50 %.4f%s" % (
            c["depth"], " (focal plane)" if r["in_focus"] else "", D, r["pinhole"]["w1090_px"], r["pinhole"]["mtf50_cyc_px"], r["lens"]["w1090_px"], r["lens"]["mtf50_cyc_px"],
            "  predicted w10-90 %.3f px  meas/pred %.4f" % (r["predicted_w1090_px"], r["w1090_ratio_meas_pred"]) if D > 0 else ""))
    if a.json: json.dump(dict(aperture=a.aperture, focus=a.focus, fx=fx, roi_half_width=hw, cards=rows), open(a.json, "w"), indent=1)
