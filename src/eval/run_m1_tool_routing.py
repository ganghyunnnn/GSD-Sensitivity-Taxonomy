"""
Design B -- Taxonomy-routed tool augmentation for M1 (spatial-metric) tasks.

Hypothesis: M1 failures stem from the VLM's inability to convert pixels to meters,
not from perception per se. So we split the task:
  * VLM  = perception  -> output bounding boxes for the relevant objects
  * TOOL = computation -> deterministic geometry (distance / radius / area)
                          x GSD, then threshold/compare. (externalizes the math)

This is the literal tool routing mechanism. ThinkGeo already
exposes TextToBbox/Calculator; here the backbone's own grounding plays TextToBbox
and a deterministic function plays Calculator.

Scoring is subtype-aware (and honest about the exact-number-whitelist problem):
  * proximity (yes/no, count)  -> whitelist accuracy (robust to localization noise)
  * distance / area (numeric)  -> relative error |pred-gt|/gt (exact-match whitelist
                                  is hostile to any grounding-based estimate, so we
                                  report how close the computed metric value is)

Usage:
    python src/eval/run_m1_tool_routing.py --dry-run --limit 10 --model qwen2.5vl:7b
    python src/eval/run_m1_tool_routing.py --model qwen2.5vl:7b   # full 161
"""
import argparse
import base64
import io
import json
import math
import re
import sys
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from eval.evaluator import evaluate_answer  # noqa: E402

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except AttributeError:
    pass

BENCH = json.load(open(ROOT / "data/thinkgeo/ThinkGeoBench.json", encoding="utf-8"))
ABL = json.load(open(ROOT / "annotation/gsd_ablation.json", encoding="utf-8"))
M1_IDS = [str(r["task_id"]) for r in ABL["qwen3-vl_8b"]["baseline"]]
OLLAMA = "http://localhost:11434/api/generate"
OLLAMA_CHAT = "http://localhost:11434/api/chat"


# ── helpers ──────────────────────────────────────────────────────────────────
def img_path(t):
    for f in t.get("files", []):
        if f.get("type") == "image" and f.get("path"):
            p = ROOT / "data/thinkgeo" / f["path"]
            if p.exists():
                return p
    return None


def _safe_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def extract_gsd(q, default=0.3):
    for pat in (r'gsd\s*[=:]?\s*(\d+(?:\.\d+)?)', r'(\d+(?:\.\d+)?)\s*m/px'):
        m = re.search(pat, q, re.I)
        if m:
            v = _safe_float(m.group(1))
            if v is not None:
                return v
    return default


def extract_threshold_m(q):
    """radius / 'within X m' threshold in meters, if any."""
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:m|meter|metre)s?\b', q, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r'within\s+(\d+(?:\.\d+)?)', q, re.I)
    return float(m.group(1)) if m else None


def subtype(q):
    ql = q.lower()
    if re.search(r"\bwithin\b|radius|around .*\b(m|meter)|nearest|closest|how close|premis", ql):
        return "proximity"
    if re.search(r"area|square|coverage|extent|\bsize\b", ql):
        return "area"
    if re.search(r"distance|how far|how wide|how long|\blength|\bwidth", ql):
        return "distance"
    return "other"


def gt_number(gt):
    """first numeric whitelist target, if any."""
    if isinstance(gt, dict):
        wl = gt.get("whitelist") or []
        for group in wl:
            for term in group:
                m = re.search(r'-?\d+(?:\.\d+)?', str(term))
                if m:
                    return float(m.group())
    return None


def num_from_text(text):
    """Extract the answer's numeric value from free text, robustly.
    Prefer a number adjacent to a unit (meter/m/square meter); skip the GSD value
    (numbers tied to 'm/px' or preceded by 'gsd'). Fallback: last bare number."""
    if not text:
        return None
    t = str(text)
    # mask out GSD mentions so we never pick the scale value as the answer
    t = re.sub(r'gsd\s*[=:]?\s*[\d.]+', ' ', t, flags=re.I)
    t = re.sub(r'[\d.]+\s*m\s*/\s*px', ' ', t, flags=re.I)
    # number directly before a metric unit
    m = re.search(r'(-?\d+(?:\.\d+)?)\s*(?:square\s+meters?|sq\s*m|m2|meters?|metres?|m\b)', t, re.I)
    if m:
        return float(m.group(1))
    nums = re.findall(r'-?\d+(?:\.\d+)?', t)
    return float(nums[-1]) if nums else None


def parse_boxes(text):
    """Robustly extract [x1,y1,x2,y2] integer quads from possibly-malformed JSON."""
    quads = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', text)
    return [tuple(map(int, q)) for q in quads]


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


MAX_SIDE = 1536  # cap longest side before grounding: large RS tiles (>14MP) stall
                 # thinking-mode backbones. Bboxes are scaled back to original px.


def ground_boxes(model, ip, query):
    """Return (boxes_in_ORIGINAL_pixel_coords, raw_text). Downsizes the image for
    speed, then rescales returned boxes back so geometry uses the original GSD."""
    im = Image.open(ip).convert("RGB")
    W, H = im.size
    scale = min(1.0, MAX_SIDE / max(W, H))
    if scale < 1.0:
        im = im.resize((max(1, round(W * scale)), max(1, round(H * scale))))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    prompt = (
        "You are an object localizer. For the question below, detect every object "
        "instance needed to answer it.\n"
        f"Question: {query}\n"
        "Output ONLY a JSON list, each item {\"label\":str,\"bbox_2d\":[x1,y1,x2,y2]} "
        "in absolute pixel coordinates of THIS image. No other text."
    )
    # Chat API: thinking backbones (qwen3-vl) put the answer in message.content after
    # message.thinking; non-thinking ones (qwen2.5vl) return content directly. num_predict
    # must be large enough for thinking + the JSON answer to complete.
    r = requests.post(OLLAMA_CHAT, json={
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 1536, "num_gpu": 99}}, timeout=600)
    msg = r.json().get("message", {})
    raw = msg.get("content", "") or ""
    # fallback: if content empty (token budget exhausted mid-think), salvage coords
    # the model emitted inside its reasoning trace
    if not parse_boxes(raw):
        raw = (raw + "\n" + str(msg.get("thinking", "")))
    boxes = [tuple(round(c / scale) for c in b) for b in parse_boxes(raw)]
    return boxes, raw


# ── geometry / tool answer ───────────────────────────────────────────────────
def tool_answer(st, boxes, gsd, thr_m):
    """Return (answer_str, pred_value_or_None)."""
    if len(boxes) == 0:
        return "unknown", None
    cs = [center(b) for b in boxes]

    if st == "area":
        area_px = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
        val = area_px * gsd * gsd
        return f"{val:.1f} square meters", val

    if st == "distance":
        if len(cs) < 2:
            return "unknown", None
        # max pairwise distance (the two named objects)
        best = max(math.dist(cs[i], cs[j]) for i in range(len(cs)) for j in range(i + 1, len(cs)))
        val = best * gsd
        return f"{val:.1f} meters", val

    if st == "proximity":
        # min pairwise distance in meters; compare to threshold -> yes/no
        if len(cs) < 2:
            # single class detected -> nothing else within radius
            return "no", None
        dmin = min(math.dist(cs[i], cs[j]) for i in range(len(cs)) for j in range(i + 1, len(cs)))
        dmin_m = dmin * gsd
        if thr_m is not None:
            return ("yes" if dmin_m <= thr_m else "no"), dmin_m
        return f"{dmin_m:.1f} meters", dmin_m

    # other -> report nearest-pair distance if >=2 boxes
    if len(cs) >= 2:
        dmin = min(math.dist(cs[i], cs[j]) for i in range(len(cs)) for j in range(i + 1, len(cs)))
        val = dmin * gsd
        return f"{val:.1f} meters", val
    return "unknown", None


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5vl:7b")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # baseline VLM answers (from the GSD ablation, same backbone) for rel-err comparison
    abl_key = {"qwen2.5vl:7b": "qwen2.5vl_7b", "qwen3-vl:8b": "qwen3-vl_8b",
               "blaifa/InternVL3_5:8b": "blaifa_InternVL3_5_8b"}.get(args.model)
    base_ans = {}
    if abl_key and abl_key in ABL:
        base_ans = {str(r["task_id"]): r["answer"] for r in ABL[abl_key]["baseline"]}

    # dry run: pick a mix of subtypes (proximity + distance emphasis; area is bbox-limited)
    ids = M1_IDS
    if args.limit:
        by = {"distance": [], "proximity": [], "area": [], "other": []}
        for tid in ids:
            t = BENCH.get(tid)
            if t:
                by[subtype(t["dialogs"][0]["content"])].append(tid)
        n = args.limit
        ids = (by["proximity"][:max(1, 2 * n // 5)] +
               by["distance"][:max(1, 2 * n // 5)] +
               by["area"][:max(1, n // 5)])[:n]

    out_path = ROOT / f"experiments/m1_tool_{'dryrun_' if args.dry_run else ''}{args.model.replace('/','_').replace(':','_')}.json"
    rows = []
    done = set()
    if not args.dry_run and out_path.exists():
        prev = json.load(open(out_path, encoding="utf-8"))
        rows = prev.get("rows", [])
        done = {r["task_id"] for r in rows}
        print(f"[checkpoint] resuming: {len(done)} tasks already done")

    for i, tid in enumerate(ids):
        if tid in done:
            continue
        t = BENCH[tid]
        q = t["dialogs"][0]["content"]
        gt = t.get("gt_answer")
        ip = img_path(t)
        st = subtype(q)
        gsd = extract_gsd(q)
        thr = extract_threshold_m(q)
        boxes, raw = ground_boxes(args.model, ip, q)
        ans, pred = tool_answer(st, boxes, gsd, thr)
        ev = evaluate_answer(ans, gt)
        gtn = gt_number(gt)
        relerr = (abs(pred - gtn) / gtn) if (pred is not None and gtn) else None
        # baseline rel-err on the same numeric target (fair comparison)
        bnum = num_from_text(base_ans.get(tid, ""))
        base_relerr = (abs(bnum - gtn) / gtn) if (bnum is not None and gtn) else None
        base_correct = evaluate_answer(base_ans.get(tid, ""), gt)["correct"] if tid in base_ans else None
        row = {"task_id": tid, "subtype": st, "gsd": gsd, "thr_m": thr,
               "n_boxes": len(boxes), "answer": ans, "pred": pred,
               "gt_num": gtn, "rel_err": relerr, "correct": ev["correct"],
               "base_pred": bnum, "base_rel_err": base_relerr, "base_correct": base_correct}
        rows.append(row)
        if args.dry_run:
            re_s = f'{relerr*100:.0f}%' if relerr is not None else '-'
            bre_s = f'{base_relerr*100:.0f}%' if base_relerr is not None else '-'
            print(f"\n[{tid}] {st:9s} gsd={gsd} thr={thr}")
            print(f"  Q: {q[:88]}")
            print(f"  boxes={len(boxes)}  ans={ans!r}  pred={pred}  GTnum={gtn}")
            print(f"  TOOL relerr={re_s} correct={ev['correct']}  |  "
                  f"BASE relerr={bre_s} correct={base_correct} (base_pred={bnum})")
        # incremental checkpoint every 10 tasks (full run only)
        if not args.dry_run and (i % 10 == 0):
            json.dump({"model": args.model, "dry_run": False, "rows": rows},
                      open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # summary
    def med_of(xs):
        xs = sorted(x for x in xs if x is not None)
        return xs[len(xs) // 2] if xs else None

    print("\n===== SUMMARY (TOOL vs BASELINE) =====")
    for st in ["proximity", "distance", "area", "other"]:
        sub = [r for r in rows if r["subtype"] == st]
        if not sub:
            continue
        acc = sum(r["correct"] for r in sub) / len(sub) * 100
        bacc_vals = [r["base_correct"] for r in sub if r["base_correct"] is not None]
        bacc = sum(bacc_vals) / len(bacc_vals) * 100 if bacc_vals else None
        tmed = med_of([r["rel_err"] for r in sub])
        bmed = med_of([r["base_rel_err"] for r in sub])
        found = sum(1 for r in sub if r["n_boxes"])
        print(f"  {st:9s} n={len(sub):2d} grounded={found}/{len(sub)}")
        print(f"      acc:        tool={acc:5.1f}%   base={f'{bacc:5.1f}%' if bacc is not None else '  -  '}")
        print(f"      med_relerr: tool={f'{tmed*100:4.0f}%' if tmed is not None else '  - '}   "
              f"base={f'{bmed*100:4.0f}%' if bmed is not None else '  - '}")

    json.dump({"model": args.model, "dry_run": args.dry_run, "rows": rows},
              open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
