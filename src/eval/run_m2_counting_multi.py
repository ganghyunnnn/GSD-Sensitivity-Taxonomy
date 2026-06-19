"""
M2 Counting-Specific Prompt Evaluation — multi-backbone version.

Builds taxonomy-routed prompting result for a given backbone:
  D  = backbone_<key>.json baseline (same prompt reuse)
  M1 = gsd_ablation few_shot condition, eval-split subset (best ablation reuse)
  M2 = counting-specific prompt, NEW inference

Usage:
    python src/eval/run_m2_counting_multi.py --model qwen2.5vl:7b \
        --ablation-key qwen2.5vl_7b --backbone backbone_qwen25vl_7b.json \
        --out routed_eval_qwen25vl_7b.json
"""
import argparse
import base64
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval.evaluator import evaluate_answer
from models.ollama_vlm import OllamaVLM

BENCH_PATH    = ROOT / "data/thinkgeo/ThinkGeoBench.json"
LABELS_PATH   = ROOT / "annotation/thinkgeo_taxonomy_labels.json"
ABLATION_PATH = ROOT / "annotation/gsd_ablation.json"
DATA_ROOT     = ROOT / "data/thinkgeo"

MAX_IMAGE_PX = 1024   # match the qwen3-vl routed M2 run (run_m2_counting_eval.py) for fair comparison

COUNTING_PROMPT = (
    "Look at the image carefully and count precisely.\n"
    "Instructions:\n"
    "  - Identify each distinct instance of the target object.\n"
    "  - Do not double-count partially visible objects unless clearly identifiable.\n"
    "  - State your final answer as a single number or short phrase.\n"
    "Question: {query}\n"
    "Answer:"
)


def load_image_b64(task: dict) -> str | None:
    from PIL import Image
    for f in task.get("files", []):
        if f.get("type") == "image" and f.get("path"):
            p = DATA_ROOT / f["path"]
            if not p.exists():
                continue
            img = Image.open(p).convert("RGB")
            w, h = img.size
            if max(w, h) > MAX_IMAGE_PX:
                scale = MAX_IMAGE_PX / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode()
    return None


def main(model: str, ablation_key: str, backbone_file: str, out_file: str):
    backbone_path = ROOT / "annotation" / backbone_file
    out_path = ROOT / "annotation" / out_file

    with open(BENCH_PATH, encoding="utf-8") as f:
        bench = json.load(f)
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)
    with open(backbone_path, encoding="utf-8") as f:
        baseline = json.load(f)
    with open(ABLATION_PATH, encoding="utf-8") as f:
        ablation = json.load(f)

    eval_results = baseline["direct"]["results"]

    # ── D: baseline reuse ──────────────────────────────────────────────
    d_results = [
        {"task_id": r["task_id"], "gt_type": "D", "gt_types": r.get("gt_types", ["D"]),
         "routed_to": "D", "gsd_used": None,
         "correct": r["correct"], "answer": str(r.get("answer", ""))[:200]}
        for r in eval_results if r["gt_type"] == "D"
    ]
    print(f"D (baseline reuse): {len(d_results)} tasks")

    # ── M1: ablation few_shot reuse ────────────────────────────────────
    m1_eval_ids = {r["task_id"] for r in eval_results if r["gt_type"] == "M1"}
    m1_abl = ablation.get(ablation_key, {}).get("few_shot", [])
    m1_abl_map = {r["task_id"]: r for r in m1_abl if r["task_id"] in m1_eval_ids}
    m1_base_map = {r["task_id"]: r for r in eval_results if r["gt_type"] == "M1"}

    import re
    m1_results = []
    for tid in m1_eval_ids:
        gt_types = labels[tid]["annotation"]["types"] if tid in labels else ["M1"]
        if tid in m1_abl_map:
            ar = m1_abl_map[tid]
            task = bench.get(tid, {})
            query = task.get("dialogs", [{}])[0].get("content", "")
            gsd_m = re.search(r'gsd\s*[=:]\s*([\d.]+)', query, re.IGNORECASE)
            gsd = gsd_m.group(1) if gsd_m else "0.3"
            m1_results.append({
                "task_id": tid, "gt_type": "M1", "gt_types": gt_types,
                "routed_to": "M1", "gsd_used": gsd,
                "correct": ar["correct"], "answer": str(ar.get("answer", ""))[:200]
            })
        elif tid in m1_base_map:
            r = m1_base_map[tid]
            m1_results.append({
                "task_id": tid, "gt_type": "M1", "gt_types": gt_types,
                "routed_to": "M1", "gsd_used": None,
                "correct": r["correct"], "answer": str(r.get("answer", ""))[:200]
            })
    print(f"M1 (ablation few_shot reuse): {len(m1_results)} tasks "
          f"(few_shot={len(m1_abl_map)}, fallback={len(m1_results)-len(m1_abl_map)})")

    # ── M2: new counting inference ─────────────────────────────────────
    m2_eval_tasks = [
        (r["task_id"], bench[r["task_id"]])
        for r in eval_results if r["gt_type"] == "M2" and r["task_id"] in bench
    ]
    print(f"M2 (new inference): {len(m2_eval_tasks)} tasks on {model}")

    m2_done = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            saved = json.load(f)
        for r in saved.get("results", []):
            if r["gt_type"] == "M2":
                m2_done[r["task_id"]] = r
        print(f"  Checkpoint: {len(m2_done)} M2 done")

    vlm = OllamaVLM(model)
    m2_results = list(m2_done.values())
    done_ids = set(m2_done.keys())

    for i, (tid, task) in enumerate(m2_eval_tasks):
        if tid in done_ids:
            continue
        query = task.get("dialogs", [{}])[0].get("content", "")
        gt_answer = task.get("gt_answer")
        img_b64 = load_image_b64(task)
        gt_types = labels[tid]["annotation"]["types"] if tid in labels else ["M2"]
        prompt = COUNTING_PROMPT.format(query=query)

        t0 = time.monotonic()
        try:
            if img_b64:
                resp = vlm.generate_b64(prompt, img_b64, max_tokens=64)
            else:
                resp = vlm.generate(prompt, image_path=None, max_tokens=64)
            answer = resp.text
            correct = evaluate_answer(answer, gt_answer)["correct"]
        except Exception as e:
            answer = f"ERROR: {e}"
            correct = False
        elapsed = round(time.monotonic() - t0, 1)

        m2_results.append({
            "task_id": tid, "gt_type": "M2", "gt_types": gt_types,
            "routed_to": "M2", "gsd_used": None,
            "correct": correct, "answer": str(answer)[:200]
        })
        done_ids.add(tid)

        if (i + 1) % 10 == 0 or (i + 1) == len(m2_eval_tasks):
            _save(out_path, model, d_results, m1_results, m2_results)
            acc = sum(r["correct"] for r in m2_results) / len(m2_results)
            print(f"  M2 [{i+1}/{len(m2_eval_tasks)}] {elapsed}s acc_m2={acc:.1%}")

    _save(out_path, model, d_results, m1_results, m2_results)
    _summary(model, d_results, m1_results, m2_results, out_path)


def _save(out_path, model, d, m1, m2):
    allr = d + m1 + m2
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in allr:
        stats[r["gt_type"]]["total"] += 1
        if r["correct"]:
            stats[r["gt_type"]]["correct"] += 1
    per_type = {}
    for t in ["D", "M1", "M2"]:
        s = stats[t]
        acc = round(s["correct"] / s["total"], 4) if s["total"] else 0.0
        per_type[t] = {"correct": s["correct"], "total": s["total"],
                       "accuracy": acc, "failure_rate": round(1 - acc, 4)}
    total_c = sum(r["correct"] for r in allr)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model, "method": "taxonomy_routed",
            "routing": {"D": "direct (baseline reuse)",
                        "M1": "gsd_few_shot (ablation reuse)",
                        "M2": "counting_specific (new inference)"},
            "accuracy": round(total_c / len(allr), 4) if allr else 0,
            "correct": total_c, "total": len(allr),
            "per_type": per_type, "results": allr,
        }, f, indent=2, ensure_ascii=False)


def _summary(model, d, m1, m2, out_path):
    print("\n" + "=" * 50)
    print(f"Routed prompting: {model}")
    for t, rs in [("D", d), ("M1", m1), ("M2", m2)]:
        if rs:
            acc = sum(r["correct"] for r in rs) / len(rs)
            print(f"  {t}: N={len(rs)} fail={(1-acc)*100:.1f}%")
    m = m1 + m2
    if m and d:
        mf = 1 - sum(r["correct"] for r in m) / len(m)
        df = 1 - sum(r["correct"] for r in d) / len(d)
        print(f"  Gap(M-D): +{(mf-df)*100:.1f}pp")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ablation-key", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.model, args.ablation_key, args.backbone, args.out)
