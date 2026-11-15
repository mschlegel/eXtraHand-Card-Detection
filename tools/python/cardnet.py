#!/usr/bin/env python3
"""
cardnet.py - reference loader / classifier for the eXtraHand playing-card dataset.

Mirrors the Kotlin library (cardrecognition/.../core). Spec: docs/FEATURE_SPEC.md, docs/DATASET_FORMAT.md.
Requires: numpy, pillow.

  python cardnet.py classify card.jpg                 # uses the public GitHub dataset
  python cardnet.py classify card.jpg --dataset ../../dataset   # local checkout
  python cardnet.py stats
  python cardnet.py rebuild --dataset ../../dataset   # recompute every feature from the JPEGs (prints check)
  python cardnet.py evaluate --dataset ../../dataset  # leave-one-out accuracy on the stored images

As a library:
  from cardnet import Dataset, extract, locate
  ds = Dataset.load()                # or Dataset.load("path/or/url")
  print(ds.classify(Image.open("x.jpg")))
"""
import argparse
import base64
import hashlib
import json
import os
import sys
import urllib.request

import numpy as np
from PIL import Image

DEFAULT_URL = "https://raw.githubusercontent.com/mschlegel/eXtraHand-Card-Detection/main/dataset"
EXTRACTOR_ID = "luma-grid-v1"

CANON_W, CANON_H = 126, 176
CORNER_W, CORNER_H = 30, 52
CORNER_G = (16, 24)
CARD_G = (24, 32)
RED_G = (8, 12)
W_CORNER, W_CARD = 1.0, 0.6
RED_OFFSET = 0.06
W_RED = 3.0 / np.sqrt(RED_G[0] * RED_G[1])
DIM = 16 * 24 + 24 * 32 + 8 * 12  # 1248


# ---------------------------------------------------------------- features
def _resample(a, x0, y0, x1, y1, dw, dh):
    """Box-filter resample of a[y0:y1, x0:x1] to (dh, dw). Integer bounds exactly as the Kotlin Grid.resample."""
    sub = a[y0:y1, x0:x1].astype(np.float64)
    sh, sw = sub.shape
    S = np.zeros((sh + 1, sw + 1))
    S[1:, 1:] = sub.cumsum(0).cumsum(1)
    i = np.arange(dw); j = np.arange(dh)
    xa = (i * sw) // dw; xb = np.maximum(xa + 1, ((i + 1) * sw) // dw)
    ya = (j * sh) // dh; yb = np.maximum(ya + 1, ((j + 1) * sh) // dh)
    tot = S[yb][:, xb] - S[ya][:, xb] - S[yb][:, xa] + S[ya][:, xa]
    return tot / ((yb - ya)[:, None] * (xb - xa)[None, :])


def _corner(a, g):
    gx, gy = g
    tl = _resample(a, 0, 0, CORNER_W, CORNER_H, gx, gy)
    br = _resample(a, CANON_W - CORNER_W, CANON_H - CORNER_H, CANON_W, CANON_H, gx, gy)[::-1, ::-1]
    return (tl + br) / 2


def _norm(v, w):
    z = v - v.mean()
    n = np.linalg.norm(z)
    return np.zeros_like(z) if n < 1e-6 else z / n * w


def extract(img: Image.Image) -> np.ndarray:
    """Feature vector (float32, L2-normalized, length 1248) for a tightly cropped card image."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    if rgb.shape[1] > rgb.shape[0]:  # landscape -> rotate 90 deg clockwise
        rgb = np.rot90(rgb, k=-1)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    red = np.maximum(0, r - np.maximum(g, b) - RED_OFFSET)
    h, w = luma.shape
    L = _resample(luma, 0, 0, w, h, CANON_W, CANON_H)
    R = _resample(red, 0, 0, w, h, CANON_W, CANON_H)
    corner = _corner(L, CORNER_G)
    whole = _resample(L, 0, 0, CANON_W, CANON_H, *CARD_G)
    whole = (whole + whole[::-1, ::-1]) / 2
    redc = _corner(R, RED_G)
    v = np.concatenate([_norm(corner.ravel(), W_CORNER), _norm(whole.ravel(), W_CARD), redc.ravel() * W_RED])
    n = np.linalg.norm(v)
    return (v / n if n > 1e-12 else v).astype(np.float32)


def _otsu(v):
    hist = np.bincount(np.clip(v.astype(int), 0, 255).ravel(), minlength=256).astype(np.float64)
    total, s = hist.sum(), (np.arange(256) * hist).sum()
    wb = sb = best = 0.0; t = 127
    for i in range(256):
        wb += hist[i]
        if wb == 0: continue
        wf = total - wb
        if wf == 0: break
        sb += i * hist[i]
        mb, mf = sb / wb, (s - sb) / wf
        between = wb * wf * (mb - mf) ** 2
        if between > best: best, t = between, i
    return t


def locate(img: Image.Image) -> Image.Image:
    """Tighten a rough crop to the bright card face (CardLocator). Returns the input if unsure."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float64)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    h, w = luma.shape
    wh = 96; ww = max(8, int(w / (h / wh)))
    small = _resample(luma, 0, 0, w, h, ww, wh)
    bright = small > _otsu(small)
    cols = np.where(bright.mean(0) >= 0.55)[0]; rows = np.where(bright.mean(1) >= 0.55)[0]
    if len(cols) == 0 or len(rows) == 0: return img
    l, r, t, b = cols[0] / ww, (cols[-1] + 1) / ww, rows[0] / wh, (rows[-1] + 1) / wh
    aspect = ((r - l) * w) / ((b - t) * h)
    pa = 1 / aspect if aspect > 1 else aspect
    ca = 63 / 88
    if (r - l) * (b - t) < 0.30 or not (ca * 0.8 <= pa <= ca * 1.2): return img
    return img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))


def encode_vector(v):
    m = np.abs(v).max()
    q = np.clip(np.rint(v * (127 / m if m > 0 else 0)), -127, 127).astype(np.int8)
    return base64.b64encode(q.tobytes()).decode()


def decode_vector(s, dim=DIM):
    q = np.frombuffer(base64.b64decode(s), dtype=np.int8).astype(np.float32)
    assert q.size == dim, f"vector length {q.size} != {dim}"
    return q / (np.linalg.norm(q) or 1)


# ---------------------------------------------------------------- dataset
def _read(base, rel):
    if base.startswith("http"):
        with urllib.request.urlopen(base.rstrip("/") + "/" + rel, timeout=30) as r:
            return r.read()
    with open(os.path.join(base, rel), "rb") as f:
        return f.read()


class Dataset:
    def __init__(self, manifest, samples, base):
        self.manifest, self.samples, self.base = manifest, samples, base
        self.labels = np.array([s["label"] for s in samples])
        self.ids = [s["id"] for s in samples]
        self.X = np.stack([decode_vector(s["f"]) for s in samples]) if samples else np.zeros((0, DIM), np.float32)

    @classmethod
    def load(cls, base=DEFAULT_URL):
        """Load manifest + all shards from a URL or local folder (the dataset root)."""
        try:
            m = json.loads(_read(base, "manifest.json"))
        except FileNotFoundError:
            return cls({"shards": [], "deleted": [], "extractor": EXTRACTOR_ID}, [], base)
        assert m["extractor"] == EXTRACTOR_ID, f"dataset extractor {m['extractor']} != {EXTRACTOR_ID}"
        deleted, seen, samples = set(m.get("deleted", [])), set(), []
        for ref in m["shards"]:
            raw = _read(base, ref["file"])
            if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
                print(f"warning: sha mismatch for {ref['file']}", file=sys.stderr)
            for s in json.loads(raw)["samples"]:
                if s["id"] not in deleted and s["id"] not in seen:
                    seen.add(s["id"]); samples.append(s)
        return cls(m, samples, base)

    def classify_vector(self, f, k=7, temperature=0.03, top_n=3, exclude=None):
        if len(self.samples) == 0: return []
        sims = self.X @ f
        if exclude is not None: sims[exclude] = -np.inf
        idx = np.argsort(-sims)[:k]
        top = sims[idx[0]]
        scores, best = {}, {}
        for i in idx:
            lab = self.labels[i]
            scores[lab] = scores.get(lab, 0) + np.exp((sims[i] - top) / temperature)
            best[lab] = max(best.get(lab, -1), sims[i])
        tot = sum(scores.values())
        out = sorted(((lab, v / tot, best[lab]) for lab, v in scores.items()), key=lambda t: -t[1])
        return [{"label": l, "probability": float(p), "similarity": float(s)} for l, p, s in out[:top_n]]

    def classify(self, img, do_locate=True, **kw):
        return self.classify_vector(extract(locate(img) if do_locate else img), **kw)


# ---------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["classify", "stats", "rebuild", "evaluate"])
    ap.add_argument("image", nargs="?")
    ap.add_argument("--dataset", default=DEFAULT_URL, help="dataset root: URL or local folder")
    a = ap.parse_args()
    ds = Dataset.load(a.dataset)
    if a.cmd == "stats":
        m = ds.manifest
        print(f"version {m.get('datasetVersion')} | {len(ds.samples)} samples | {len(set(ds.labels))} labels | updated {m.get('updatedAt')}")
        for lab, c in sorted(m.get("labelCounts", {}).items()): print(f"  {lab:>3} {c}")
    elif a.cmd == "classify":
        for c in ds.classify(Image.open(a.image)): print(c)
    elif a.cmd == "rebuild":
        sims = []
        for s in ds.samples:
            if not s.get("image"): continue
            f = extract(Image.open(os.path.join(a.dataset, s["image"])))
            sims.append(float(f @ decode_vector(s["f"])))
        if sims: print(f"{len(sims)} images, cosine(python, stored) min={min(sims):.4f} mean={np.mean(sims):.4f}")
    elif a.cmd == "evaluate":
        ok = sum(ds.classify_vector(ds.X[i], exclude=i, top_n=1)[0]["label"] == ds.labels[i] for i in range(len(ds.samples)))
        print(f"leave-one-out accuracy: {ok}/{len(ds.samples)} = {ok / max(1, len(ds.samples)):.1%}")


if __name__ == "__main__":
    main()
