"""
DOTA-val detector-vs-VLM counting (closed-loop, non-circular, powered).
Pre-registration: experiments/dota_val_preregistration.json (frozen before val run).

For a pre-registered random sample of DOTA-v1 VAL images (held out from the
detector's DOTA-train training), and the 4 paper-relevant M2 classes, we compute
per (image x class):
  GT (strict + resolvable@10px), detector count (yolo11l-obb), VLM count (qwen2.5vl).
Detector-vs-VLM is the paired comparison of interest (both scored on the same GT).
"""
import argparse
import base64
import io
import json
import math
import random
import re
import sys
from pathlib import Path

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent
IMG_DIR = Path("E:/FP-Tag/DOTAv1/images/val")
LBL_DIR = Path("E:/FP-Tag/DOTAv1/labels/val")
OUT = ROOT / "experiments/dota_val_counting.json"

DOTA = {0: "plane", 1: "ship", 2: "storage tank", 3: "baseball diamond", 4: "tennis court",
        5: "basketball court", 6: "ground track field", 7: "harbor", 8: "bridge",
        9: "large vehicle", 10: "small vehicle", 11: "helicopter", 12: "roundabout",
        13: "soccer ball field", 14: "swimming pool"}
NAME2IDX = {v: k for k, v in DOTA.items()}
PRIMARY = ["plane", "ship", "small vehicle", "large vehicle"]
NOUN = {"plane": "airplanes", "ship": "ships",
        "small vehicle": "small vehicles (cars)",
        "large vehicle": "large vehicles (trucks or buses)"}
S_FLOOR = 10  # px resolvability floor


def gt_counts(stem, W, H):
    """return {class: (strict, resolvable)} from YOLO-OBB label file."""
    p = LBL_DIR / f"{stem}.txt"
    strict = {c: 0 for c in PRIMARY}
    resolv = {c: 0 for c in PRIMARY}
    if not p.exists():
        return strict, resolv
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) < 9:
            continue
        ci = int(parts[0])
        cname = DOTA.get(ci)
        if cname not in PRIMARY:
            continue
        xs = [float(parts[i]) * W for i in (1, 3, 5, 7)]
        ys = [float(parts[i]) * H for i in (2, 4, 6, 8)]
        # OBB longer side = max of the 4 edge lengths
        pts = list(zip(xs, ys))
        edges = [math.dist(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        longer = max(edges)
        strict[cname] += 1
        if longer >= S_FLOOR:
            resolv[cname] += 1
    return strict, resolv


def detect_counts(model, ip, imgsz, conf):
    res = model.predict(str(ip), imgsz=imgsz, conf=conf, verbose=False)[0]
    ids = res.obb.cls.tolist() if res.obb is not None else []
    out = {c: 0 for c in PRIMARY}
    for i in ids:
        n = DOTA.get(int(i))
        if n in out:
            out[n] += 1
    return out


def vlm_count(ip, cname, b64=None):
    if b64 is None:
        b64 = base64.b64encode(open(ip, "rb").read()).decode()
    prompt = f"How many {NOUN[cname]} are in this image? Answer with a single integer only."
    try:
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": "qwen2.5vl:7b", "prompt": prompt, "images": [b64], "stream": False,
            "options": {"temperature": 0, "num_predict": 32, "num_gpu": 99}}, timeout=300)
        m = re.search(r"\b(\d+)\b", r.json().get("response", ""))
        return int(m.group(1)) if m else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO("yolo11l-obb.pt")

    imgs = sorted(p.stem for p in IMG_DIR.glob("*.jpg"))
    random.seed(42)
    sample = sorted(random.sample(imgs, min(args.n, len(imgs))))

    rows = []
    done = set()
    if OUT.exists():
        prev = json.load(open(OUT, encoding="utf-8"))
        rows = prev.get("rows", [])
        done = {r["image"] for r in rows}
        print(f"[checkpoint] {len(done)} images done")

    for k, stem in enumerate(sample):
        if stem in done:
            continue
        ip = IMG_DIR / f"{stem}.jpg"
        if not ip.exists():
            continue
        W, H = Image.open(ip).size
        strict, resolv = gt_counts(stem, W, H)
        det = detect_counts(model, ip, args.imgsz, args.conf)
        b64 = base64.b64encode(open(ip, "rb").read()).decode()
        for c in PRIMARY:
            vc = vlm_count(ip, c, b64)
            rows.append({"image": stem, "cls": c,
                         "gt": strict[c], "gt_resolv": resolv[c],
                         "det": det[c], "vlm": vc})
        if k % 10 == 0:
            json.dump({"rows": rows, "config": {"imgsz": args.imgsz, "conf": args.conf, "n": args.n}},
                      open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
            print(f"  {k+1}/{len(sample)} images  ({len(rows)} cells)")

    json.dump({"rows": rows, "config": {"imgsz": args.imgsz, "conf": args.conf, "n": args.n}},
              open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"DONE {len(rows)} cells -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
