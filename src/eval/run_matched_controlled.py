"""
Matched-pair controlled experiment on DOTA-v1 val.

Design decisions:
  - D presence arm is BALANCED 50/50 (210 present + 210 sampled absent cells)
    so the 73.8% empty-cell majority prior cannot inflate the D/M gap.
  - Within-positive descriptive anchor (D2, ship orientation) exists on the
    same cells as M1, so the D-vs-M1 pairing is not against a constant-GT arm.
  - M1 is restricted to SHIP (the only class whose max-OBB x GSD ground truth
    is physically plausible); dual tolerance scoring (+-20% / +-50%) plus a
    class-median constant baseline as the answer-space floor.
  - GSD manipulation: the SAME images and questions re-asked at 0.5x and
    0.25x downsampling (effective GSD x2 / x4) -- isolates GSD with image
    domain, object density, and question wording held fixed.
  - Trivial-baseline floors reported for every arm.

Arms (per backbone):
  D1   presence, balanced 420 cells, native scale
  D2   largest-ship orientation (horizontal/vertical), unambiguous cells only
  M1   largest-ship length in meters, valid-GSD ship cells
  M2s  counting on the 210 present cells at scale 0.5 / 0.25
       (native-scale counts already exist in dota_val_counting.json)
  D1s  presence on 100 present + 100 absent cells at scale 0.5 / 0.25

Run:
  uv run python src/eval/run_matched_controlled.py --validate   # GT sanity only
  uv run python src/eval/run_matched_controlled.py --pilot      # small pilot, qwen2.5vl
  uv run python src/eval/run_matched_controlled.py --backbone qwen25vl  # one full backbone
Output: experiments/matched_controlled.json (checkpointed)
"""
import argparse
import base64
import io
import json
import random
import re
import sys
import os
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
# DOTA-v1 val root. Defaults to data/DOTAv1 under the repo; override with
# the DOTA_ROOT environment variable to point at an external copy.
DOTA_ROOT = Path(os.environ.get("DOTA_ROOT", ROOT / "data" / "DOTAv1"))
IMG_DIR = DOTA_ROOT / "images" / "val"
COUNTS = ROOT / "experiments/dota_val_counting.json"
GSDMAP = ROOT / "experiments/dota_val_gsd_map.json"
OUT = ROOT / "experiments/matched_controlled.json"

NOUN = {"plane": "airplanes", "ship": "ships",
        "small vehicle": "small vehicles (cars)",
        "large vehicle": "large vehicles (trucks or buses)"}
BACKBONES = {"qwen25vl": ("qwen2.5vl:7b", "generate"),
             "internvl35": ("blaifa/InternVL3_5:8b", "generate"),
             "qwen3vl": ("qwen3-vl:8b", "chat")}
SEED = 42
M1_MAX_GT_M = 400.0
D2_H_MAX, D2_V_MIN = 30.0, 60.0


# ---------------- question generation ----------------

def build_questions():
    rows = json.load(open(COUNTS, encoding="utf-8"))["rows"]
    gsd = json.load(open(GSDMAP, encoding="utf-8"))["images"]
    rng = random.Random(SEED)

    present = [r for r in rows if r["gt"] > 0]
    absent = [r for r in rows if r["gt"] == 0]
    absent_bal = rng.sample(absent, len(present))

    qs = []

    def add(arm, r, scale, prompt, gt, **meta):
        qs.append({"qid": f"{arm}|{r['image']}|{r['cls']}|{scale}",
                   "arm": arm, "image": r["image"], "cls": r["cls"],
                   "scale": scale, "prompt": prompt, "gt": gt, **meta})

    # D1: balanced presence, native
    for r in present + absent_bal:
        add("D1", r, 1.0,
            f"Is there at least one {NOUN[r['cls']].rstrip('s') if False else NOUN[r['cls']]} "
            f"in this image? Answer with only 'yes' or 'no'.",
            "yes" if r["gt"] > 0 else "no")

    # ship cells with valid GSD -> D2 + M1
    ships = [r for r in present if r["cls"] == "ship"
             and gsd.get(r["image"], {}).get("flag") == "ok"]
    for r in ships:
        m = gsd[r["image"]]
        inst = m["instances_px"]["ship"]
        if not inst:
            continue
        largest = max(inst, key=lambda i: i["px"])
        gt_m = largest["px"] * m["gsd"]
        ang = largest["angle_deg"]
        if ang <= D2_H_MAX or ang >= D2_V_MIN:
            add("D2", r, 1.0,
                "Consider the largest ship in this image. Is its long axis "
                "oriented more horizontally or more vertically in the image? "
                "Answer with only 'horizontal' or 'vertical'.",
                "horizontal" if ang <= D2_H_MAX else "vertical",
                angle_deg=ang)
        if gt_m <= M1_MAX_GT_M:
            add("M1", r, 1.0,
                "What is the approximate length in meters of the largest ship "
                "in this image? Answer with a single number only.",
                round(gt_m, 1), gsd_mpp=m["gsd"], largest_px=largest["px"])

    # GSD manipulation arms
    d1s_present = rng.sample(present, 100)
    d1s_absent = rng.sample(absent, 100)
    for scale in (0.5, 0.25):
        for r in present:
            add("M2s", r, scale,
                f"How many {NOUN[r['cls']]} are in this image? "
                f"Answer with a single integer only.", r["gt"])
        for r in d1s_present + d1s_absent:
            add("D1s", r, scale,
                f"Is there at least one {NOUN[r['cls']]} in this image? "
                f"Answer with only 'yes' or 'no'.",
                "yes" if r["gt"] > 0 else "no")
    return qs


# ---------------- GT validation ----------------

def validate(qs):
    import statistics
    by_arm = {}
    for q in qs:
        by_arm.setdefault(q["arm"] + (f"@{q['scale']}" if q["scale"] != 1.0 else ""),
                          []).append(q)
    print("arm            n     notes")
    for arm, items in sorted(by_arm.items()):
        note = ""
        if arm == "D1":
            yes = sum(1 for q in items if q["gt"] == "yes")
            note = f"yes={yes} no={len(items) - yes} (balanced)"
        if arm == "D2":
            h = sum(1 for q in items if q["gt"] == "horizontal")
            note = f"horizontal={h} vertical={len(items) - h}"
        if arm == "M1":
            g = sorted(q["gt"] for q in items)
            note = (f"gt_m median={statistics.median(g):.0f} "
                    f"range=[{g[0]:.0f},{g[-1]:.0f}] "
                    f"const-median baseline @20%tol="
                    f"{100 * sum(1 for x in g if abs(x - statistics.median(g)) <= 0.2 * x) / len(g):.0f}%")
        print(f"{arm:12s} {len(items):5d}  {note}")


# ---------------- inference ----------------

def encode_image(stem, scale, cache={}):
    key = (stem, scale)
    if key in cache:
        return cache[key]
    p = IMG_DIR / f"{stem}.jpg"
    if scale == 1.0:
        b = p.read_bytes()
    else:
        im = Image.open(p)
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                       Image.LANCZOS)
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=92)
        b = buf.getvalue()
    enc = base64.b64encode(b).decode()
    cache.clear()  # keep memory flat; sequential access per image anyway
    cache[key] = enc
    return enc


def ask(model, mode, prompt, img_b64):
    try:
        if mode == "generate":
            r = requests.post("http://localhost:11434/api/generate", json={
                "model": model, "prompt": prompt, "images": [img_b64], "stream": False,
                "options": {"temperature": 0, "num_predict": 48, "num_gpu": 99}},
                timeout=300)
            return r.json().get("response", "")
        r = requests.post("http://localhost:11434/api/chat", json={
            "model": model,
            "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 1024, "num_gpu": 99}},
            timeout=300)
        msg = r.json().get("message", {})
        return msg.get("content", "") or str(msg.get("thinking", ""))
    except Exception:
        return None


def parse(arm, text):
    if text is None:
        return None
    t = text.lower()
    if arm in ("D1", "D1s"):
        if re.search(r"\byes\b", t):
            return "yes"
        if re.search(r"\bno\b", t):
            return "no"
        return None
    if arm == "D2":
        if "horizontal" in t:
            return "horizontal"
        if "vertical" in t:
            return "vertical"
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", t)
    return float(m.group(1)) if m else None


def run(qs, backbone, limit=None):
    model, mode = BACKBONES[backbone]
    data = json.load(open(OUT, encoding="utf-8")) if OUT.exists() else {"questions": {}}
    store = data["questions"]
    for q in qs:
        store.setdefault(q["qid"], q)
    field = f"ans_{backbone}"
    todo = [q for q in qs if field not in store[q["qid"]]]
    if limit:
        todo = todo[:limit]
    # group by (image, scale) so each encode serves several questions
    todo.sort(key=lambda q: (q["image"], q["scale"], q["arm"]))
    print(f"[{backbone}] {len(todo)} questions to run")
    for i, q in enumerate(todo):
        img = encode_image(q["image"], q["scale"])
        raw = ask(model, mode, q["prompt"], img)
        store[q["qid"]][field] = parse(q["arm"], raw)
        store[q["qid"]][f"raw_{backbone}"] = (raw or "")[:120]
        if (i + 1) % 20 == 0:
            json.dump(data, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
            print(f"  {i + 1}/{len(todo)}", flush=True)
    json.dump(data, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"[{backbone}] done -> {OUT.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--backbone", choices=list(BACKBONES))
    args = ap.parse_args()

    qs = build_questions()
    if args.validate:
        validate(qs)
        return
    if args.pilot:
        rng = random.Random(0)
        sample = []
        for arm in ("D1", "D2", "M1", "M2s", "D1s"):
            arm_qs = [q for q in qs if q["arm"] == arm]
            sample += rng.sample(arm_qs, min(8, len(arm_qs)))
        run(sample, "qwen25vl")
        return
    if args.backbone:
        run(qs, args.backbone)
        return
    for b in ("qwen25vl", "internvl35", "qwen3vl"):
        run(qs, b)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
