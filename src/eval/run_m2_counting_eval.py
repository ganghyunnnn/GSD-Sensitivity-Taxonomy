"""
M2 Counting-Specific Prompt Evaluation (Experiment A, M2 component)

D 결과 = backbone_qwen3vl_8b (동일 프롬프트 재사용)
M1 결과 = gsd_ablation few_shot 조건 eval split (best ablation condition 재사용)
M2 결과 = counting-specific 프롬프트로 새로 inference (83개)

Usage:
    python src/eval/run_m2_counting_eval.py
"""
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
BASELINE_PATH = ROOT / "annotation/backbone_qwen3vl_8b.json"
ABLATION_PATH = ROOT / "annotation/gsd_ablation.json"
DATA_ROOT     = ROOT / "data/thinkgeo"
OUT_PATH      = ROOT / "annotation/routed_eval_qwen3vl_8b.json"

MAX_IMAGE_PX = 1024

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


def primary_type(types: list) -> str:
    for t in ["M1", "M2", "D"]:
        if t in types:
            return t
    return types[0]


def main():
    with open(BENCH_PATH, encoding="utf-8") as f:
        bench = json.load(f)
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)
    with open(BASELINE_PATH, encoding="utf-8") as f:
        baseline = json.load(f)
    with open(ABLATION_PATH, encoding="utf-8") as f:
        ablation = json.load(f)

    eval_results = baseline["direct"]["results"]
    eval_ids = {r["task_id"] for r in eval_results}

    # ── D 결과: baseline 그대로 ──────────────────────────────────────────────
    d_results = [
        {"task_id": r["task_id"], "gt_type": "D", "gt_types": r.get("gt_types", ["D"]),
         "routed_to": "D", "gsd_used": None,
         "correct": r["correct"], "answer": r.get("answer", "")[:200]}
        for r in eval_results if r["gt_type"] == "D"
    ]
    print(f"D (baseline reuse): {len(d_results)} tasks")

    # ── M1 결과: ablation few_shot eval split ────────────────────────────────
    m1_eval_ids = {r["task_id"] for r in eval_results if r["gt_type"] == "M1"}
    m1_abl = ablation.get("qwen3-vl_8b", {}).get("few_shot", [])
    m1_abl_map = {r["task_id"]: r for r in m1_abl if r["task_id"] in m1_eval_ids}

    # few_shot에 없는 M1은 baseline으로 fallback
    m1_base_map = {r["task_id"]: r for r in eval_results if r["gt_type"] == "M1"}

    m1_results = []
    for tid in m1_eval_ids:
        if tid in m1_abl_map:
            ar = m1_abl_map[tid]
            task = bench.get(tid, {})
            query = task.get("dialogs", [{}])[0].get("content", "")
            import re
            gsd_m = re.search(r'gsd\s*[=:]\s*([\d.]+)', query, re.IGNORECASE)
            gsd = gsd_m.group(1) if gsd_m else "0.3"
            gt_types = labels[tid]["annotation"]["types"] if tid in labels else ["M1"]
            m1_results.append({
                "task_id": tid, "gt_type": "M1", "gt_types": gt_types,
                "routed_to": "M1", "gsd_used": gsd,
                "correct": ar["correct"], "answer": ar.get("answer", "")[:200]
            })
        elif tid in m1_base_map:
            r = m1_base_map[tid]
            gt_types = labels[tid]["annotation"]["types"] if tid in labels else ["M1"]
            m1_results.append({
                "task_id": tid, "gt_type": "M1", "gt_types": gt_types,
                "routed_to": "M1", "gsd_used": None,
                "correct": r["correct"], "answer": r.get("answer", "")[:200]
            })
    print(f"M1 (ablation few_shot reuse): {len(m1_results)} tasks "
          f"(few_shot={len(m1_abl_map)}, fallback={len(m1_results)-len(m1_abl_map)})")

    # ── M2 결과: counting 프롬프트로 새로 inference ──────────────────────────
    m2_eval_tasks = [
        (r["task_id"], bench[r["task_id"]])
        for r in eval_results if r["gt_type"] == "M2" and r["task_id"] in bench
    ]
    print(f"M2 (new inference): {len(m2_eval_tasks)} tasks")

    # 체크포인트
    m2_done = {}
    if OUT_PATH.exists():
        with open(OUT_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        for r in saved.get("results", []):
            if r["gt_type"] == "M2":
                m2_done[r["task_id"]] = r
        print(f"  Checkpoint: {len(m2_done)} M2 tasks already done")

    vlm = OllamaVLM("qwen3-vl:8b")
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
            elapsed = round(time.monotonic() - t0, 1)
        except Exception as e:
            answer = f"ERROR: {e}"
            correct = False
            elapsed = 0.0

        m2_results.append({
            "task_id": tid, "gt_type": "M2", "gt_types": gt_types,
            "routed_to": "M2", "gsd_used": None,
            "correct": correct, "answer": answer[:200]
        })
        done_ids.add(tid)

        if (i + 1) % 10 == 0 or (i + 1) == len(m2_eval_tasks):
            _save(d_results, m1_results, m2_results)
            acc_m2 = sum(r["correct"] for r in m2_results) / len(m2_results) if m2_results else 0
            print(f"  M2 [{i+1}/{len(m2_eval_tasks)}] {elapsed}s | acc_m2={acc_m2:.1%}")

    _save(d_results, m1_results, m2_results)
    _print_summary(d_results, m1_results, m2_results)


def _save(d, m1, m2):
    all_results = d + m1 + m2
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in all_results:
        t = r["gt_type"]
        stats[t]["total"] += 1
        if r["correct"]:
            stats[t]["correct"] += 1

    per_type = {}
    for t in ["D", "M1", "M2"]:
        s = stats[t]
        acc = round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0.0
        per_type[t] = {
            "correct": s["correct"], "total": s["total"],
            "accuracy": acc, "failure_rate": round(1 - acc, 4)
        }

    total_c = sum(r["correct"] for r in all_results)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model": "qwen3-vl:8b",
            "method": "taxonomy_routed",
            "routing": {
                "D": "direct_prompt (baseline reuse)",
                "M1": "gsd_few_shot (ablation reuse)",
                "M2": "counting_specific_prompt (new inference)"
            },
            "accuracy": round(total_c / len(all_results), 4) if all_results else 0,
            "correct": total_c,
            "total": len(all_results),
            "per_type": per_type,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)


def _print_summary(d, m1, m2):
    all_r = d + m1 + m2
    print("\n" + "=" * 50)
    print("Taxonomy-Routed Prompting (qwen3-vl:8b)")
    for t, results in [("D", d), ("M1", m1), ("M2", m2)]:
        if results:
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"  {t}: N={len(results)}, acc={acc:.1%}, fail={1-acc:.1%}")
    m_results = m1 + m2
    if m_results:
        m_acc = sum(r["correct"] for r in m_results) / len(m_results)
        d_acc = sum(r["correct"] for r in d) / len(d) if d else 0
        gap = (1 - m_acc) - (1 - d_acc)
        print(f"  M overall: acc={m_acc:.1%}, fail={1-m_acc:.1%}")
        print(f"  Gap (M-D): +{gap*100:.1f}pp")
    print(f"  Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
