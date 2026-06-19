"""
Design C -- Detector-based counting for resolvable M2 (the honest 'real tool').

Hypothesis (the one path to an honest performance win): for M2 counting questions
where the object IS resolvable (GSD <= d/s, i.e. model-error not resolution-limit)
and the class is DOTA-detectable, an in-domain oriented detector (YOLO11-OBB
trained on DOTAv1) counts instances better than the VLM's holistic guess.

Scope is honest and narrow: DOTA-detectable classes (plane, ship, small/large
vehicle, tennis-court, baseball-diamond, roundabout, ...). xBD buildings are NOT
a DOTA class and are mostly resolution-limited -> excluded (those route to abstain).
Attribute-filtered counts ("white cars", "heading southeast") are reported
separately: the detector gives a class count and cannot filter by colour/heading.

Metric: detector count vs numeric GT (exact and within-1), MAE, and the SAME
for the VLM baseline answer -> a fair tool-vs-VLM comparison (McNemar on exact).

Usage:
    python src/eval/run_m2_detector_count.py --model yolo11l-obb.pt --imgsz 1280
"""
import argparse
import io
import json
import re
import sys
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent
BENCH = json.load(open(ROOT / "data/thinkgeo/ThinkGeoBench.json", encoding="utf-8"))
LABELS = {str(r["task_id"]): r for _, r in
          json.load(open(ROOT / "annotation/thinkgeo_taxonomy_labels.json", encoding="utf-8")).items()}

# query keyword -> DOTAv1 class name(s) used by yolo11-obb
CLASS_MAP = [
    (r"\bairplane|aircraft|\bplane|jet\b", ["plane"]),
    (r"\bship|boat|vessel|carrier", ["ship"]),
    (r"\bsmall[- ]vehicle|\bcar\b|\bcars\b|sedan", ["small vehicle"]),
    (r"\blarge[- ]vehicle|\btruck|\bbus\b|lorry", ["large vehicle"]),
    (r"\bvehicle", ["small vehicle", "large vehicle"]),
    (r"tennis", ["tennis court"]),
    (r"baseball", ["baseball diamond"]),
    (r"basketball", ["basketball court"]),
    (r"roundabout", ["roundabout"]),
    (r"harbor|harbour", ["harbor"]),
    (r"\bbridge", ["bridge"]),
    (r"storage tank|tank\b", ["storage tank"]),
    (r"helicopter", ["helicopter"]),
    (r"swimming pool|\bpool", ["swimming pool"]),
]

ATTR_RE = re.compile(
    r"white|black|red|blue|green|yellow|colou?r|direction|heading|facing|north|south|"
    r"east|west|moving|same direction|overlap|touching|occupied|parked facing|"
    r"proximity|within\s+\d|radius|near|lower road|top left|top right|quadrant|empty",
    re.I)


def target_classes(q):
    ql = q.lower()
    for pat, cls in CLASS_MAP:
        if re.search(pat, ql):
            return cls
    return None


def numeric_gt(gt):
    if isinstance(gt, dict):
        for grp in gt.get("whitelist") or []:
            for t in grp:
                if re.fullmatch(r"\d+", str(t).strip()):
                    return int(str(t).strip())
    return None


def img_path(t):
    for f in t.get("files", []):
        if f.get("type") == "image" and f.get("path"):
            p = ROOT / "data/thinkgeo" / f["path"]
            if p.exists():
                return p
    return None


def ollama_count(ip, query, model="qwen2.5vl:7b"):
    """Fresh VLM count baseline (uniform across all targets) via Ollama."""
    import base64, requests
    b64 = base64.b64encode(open(ip, "rb").read()).decode()
    prompt = ("Look at the image and answer with a single integer only.\n"
              f"Question: {query}\nAnswer (number only):")
    try:
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": model, "prompt": prompt, "images": [b64], "stream": False,
            "options": {"temperature": 0, "num_predict": 64, "num_gpu": 99}}, timeout=300)
        m = re.search(r"\b(\d+)\b", r.json().get("response", ""))
        return int(m.group(1)) if m else None
    except Exception:
        return None


def collect_targets():
    out = []
    for tid, t in BENCH.items():
        r = LABELS.get(str(tid))
        if not r or "M2" not in (r.get("annotation", {}).get("types") or []):
            continue
        q = (t.get("dialogs", [{}])[0].get("content", "") or "")
        cls = target_classes(q)
        gtn = numeric_gt(t.get("gt_answer"))
        if cls is None or gtn is None:
            continue
        out.append({"task_id": str(tid), "classes": cls, "gt": gtn,
                    "attr": bool(ATTR_RE.search(q)), "query": q})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11l-obb.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="only clean pure-count vehicle/plane tasks")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    names = model.names  # idx -> classname
    name2idx = {v: k for k, v in names.items()}

    targets = collect_targets()
    if args.smoke:
        targets = [x for x in targets if not x["attr"]][:8]
    if args.limit:
        targets = targets[:args.limit]
    print(f"targets: {len(targets)}  (model={args.model}, imgsz={args.imgsz}, conf={args.conf})")

    rows = []
    for x in targets:
        t = BENCH[x["task_id"]]
        ip = img_path(t)
        if ip is None:
            continue
        want = [name2idx[c] for c in x["classes"] if c in name2idx]
        res = model.predict(str(ip), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        cls_ids = res.obb.cls.tolist() if res.obb is not None else []
        det = sum(1 for c in cls_ids if int(c) in want)
        vc = ollama_count(ip, x["query"])
        gt = x["gt"]
        row = {**{k: x[k] for k in ("task_id", "classes", "gt", "attr")},
               "det_count": det, "vlm_count": vc,
               "det_exact": det == gt, "det_within1": abs(det - gt) <= 1,
               "vlm_exact": vc == gt if vc is not None else None,
               "det_ae": abs(det - gt), "vlm_ae": abs(vc - gt) if vc is not None else None}
        rows.append(row)
        print(f"  [{x['task_id']}] {'ATTR' if x['attr'] else 'pure'} {','.join(x['classes']):24s} "
              f"GT={gt:2d} det={det:2d} vlm={vc} "
              f"{'DET_OK' if row['det_exact'] else ''}{' VLM_OK' if row['vlm_exact'] else ''}")

    # summary on pure-count subset (the honest clean claim)
    pure = [r for r in rows if not r["attr"]]
    def acc(rs, k): return sum(1 for r in rs if r[k]) / len(rs) * 100 if rs else 0
    def mae(rs, k):
        v = [r[k] for r in rs if r[k] is not None]; return sum(v) / len(v) if v else None
    print("\n===== SUMMARY (pure-count, the clean detector claim) =====")
    print(f"  n={len(pure)}")
    print(f"  exact:    detector={acc(pure,'det_exact'):.1f}%   vlm={acc([r for r in pure if r['vlm_exact'] is not None],'vlm_exact'):.1f}%")
    print(f"  within1:  detector={acc(pure,'det_within1'):.1f}%")
    print(f"  MAE:      detector={mae(pure,'det_ae'):.2f}   vlm={mae(pure,'vlm_ae')}")
    attr = [r for r in rows if r["attr"]]
    if attr:
        print(f"\n  [attribute-filtered, detector cannot filter] n={len(attr)} "
              f"detector_exact={acc(attr,'det_exact'):.1f}%")

    out = ROOT / "experiments/m2_detector_count.json"
    json.dump({"model": args.model, "imgsz": args.imgsz, "conf": args.conf, "rows": rows},
              open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
