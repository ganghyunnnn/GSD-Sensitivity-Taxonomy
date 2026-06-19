"""
Experiment A: Taxonomy-Routed Prompting
Route-specific prompts per taxonomy label: D→direct, M1→GSD-CoT, M2→counting.

Usage:
    python src/eval/run_routed_eval.py
    python src/eval/run_routed_eval.py --model qwen3-vl:8b
"""
import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

# Force UTF-8 stdout to avoid cp949 encoding errors on Windows
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval.evaluator import evaluate_answer
from models.ollama_vlm import OllamaVLM

BENCH_PATH   = ROOT / "data/thinkgeo/ThinkGeoBench.json"
LABELS_PATH  = ROOT / "annotation/thinkgeo_taxonomy_labels.json"
EVAL_REF     = ROOT / "annotation/backbone_qwen3vl_8b.json"
DATA_ROOT    = ROOT / "data/thinkgeo"

DEFAULT_GSD  = "0.3"


# ── 프롬프트 정의 ──────────────────────────────────────────────────────────────

def _prompt_d(query: str, gsd: str = None) -> str:
    """D (Descriptive): 기본 direct prompting."""
    return (
        "Look at the image carefully and answer the following question concisely.\n"
        f"Question: {query}\n"
        "Answer:"
    )


def _prompt_m1_cot(query: str, gsd: str) -> str:
    """M1 (Spatial Metric): CoT with GSD unit-conversion steps."""
    return (
        "Look at the image carefully and answer the following question.\n"
        f"The image GSD is {gsd} m/px.\n"
        "To answer spatial metric questions:\n"
        "  Step 1: Identify the relevant objects or distances in the image.\n"
        "  Step 2: Estimate the pixel distance or area.\n"
        f"  Step 3: Multiply by {gsd} to convert pixels to meters.\n"
        "  Step 4: State your final answer with units.\n"
        f"Question: {query}\n"
        "Answer:"
    )


def _prompt_m2_counting(query: str, gsd: str = None) -> str:
    """M2 (Cardinality): Counting-specific prompt."""
    return (
        "Look at the image carefully and count precisely.\n"
        "Instructions:\n"
        "  - Identify each distinct instance of the target object.\n"
        "  - Do not double-count partially visible objects unless they are clearly identifiable.\n"
        "  - State your final count as a single number.\n"
        f"Question: {query}\n"
        "Answer:"
    )


ROUTE_PROMPTS = {
    "D":  _prompt_d,
    "M1": _prompt_m1_cot,
    "M2": _prompt_m2_counting,
}

import re

def _extract_gsd(query: str) -> str:
    m = re.search(r'gsd\s*[=:]\s*([\d.]+)', query, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r'([\d.]+)\s*m/px', query, re.IGNORECASE)
    if m:
        return m.group(1)
    return DEFAULT_GSD


def primary_type(types: list[str]) -> str:
    for t in ["M1", "M2", "D"]:
        if t in types:
            return t
    return types[0]


MAX_IMAGE_PX = 1024  # Resize long side to this to speed up VLM inference


def get_image_b64(task: dict) -> str | None:
    """이미지를 로드, 최대 1024px로 리사이즈 후 base64 반환."""
    from PIL import Image
    import base64, io
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


def compute_per_type(results: list[dict]) -> dict:
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        gt = r["gt_type"]
        stats[gt]["total"] += 1
        if r["correct"]:
            stats[gt]["correct"] += 1
    out = {}
    for t in ["D", "M1", "M2"]:
        s = stats[t]
        acc = round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0.0
        fail = round(1 - acc, 4)
        out[t] = {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": acc,
            "failure_rate": fail,
        }
    return out


def run_routed_eval(model_name: str, out_path: Path):
    with open(BENCH_PATH, encoding="utf-8") as f:
        bench = json.load(f)
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)
    with open(EVAL_REF, encoding="utf-8") as f:
        ref = json.load(f)

    eval_ids = {r["task_id"] for r in ref["direct"]["results"]}

    eval_tasks = []
    for tid in eval_ids:
        task = bench.get(tid)
        if task and task.get("gt_answer") and tid in labels:
            eval_tasks.append((tid, task))
    eval_tasks.sort(key=lambda x: int(x[0]))
    print(f"평가 태스크: {len(eval_tasks)}개")

    # 체크포인트 로드
    checkpoint = []
    done_ids = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            saved = json.load(f)
            checkpoint = saved.get("results", [])
            done_ids = {r["task_id"] for r in checkpoint}
        print(f"체크포인트: {len(done_ids)}개 이미 완료")

    vlm = OllamaVLM(model_name=model_name)

    for i, (tid, task) in enumerate(eval_tasks):
        if tid in done_ids:
            continue

        query = task.get("dialogs", [{}])[0].get("content", "")
        gt_answer = task.get("gt_answer")
        img_b64 = get_image_b64(task)  # resized to MAX_IMAGE_PX
        types = labels[tid]["annotation"]["types"]
        gt_type = primary_type(types)
        gsd = _extract_gsd(query)

        # 라우팅 적용
        prompt_fn = ROUTE_PROMPTS.get(gt_type, _prompt_d)
        if gt_type == "M1":
            prompt = prompt_fn(query, gsd)
        else:
            prompt = prompt_fn(query)

        try:
            if img_b64:
                resp = vlm.generate_b64(prompt, img_b64, max_tokens=256)
            else:
                resp = vlm.generate(prompt, image_path=None, max_tokens=256)
            answer = resp.text
            correct = evaluate_answer(answer, gt_answer)["correct"]
        except Exception as e:
            answer = f"ERROR: {e}"
            correct = False

        checkpoint.append({
            "task_id": tid,
            "gt_type": gt_type,
            "gt_types": types,
            "routed_to": gt_type,
            "gsd_used": gsd if gt_type == "M1" else None,
            "correct": correct,
            "answer": answer[:300],
        })
        done_ids.add(tid)

        if len(checkpoint) % 20 == 0 or len(checkpoint) == len(eval_tasks):
            per_type = compute_per_type(checkpoint)
            total_c = sum(1 for r in checkpoint if r["correct"])
            _save(out_path, model_name, checkpoint, per_type, total_c)
            print(f"  [{len(checkpoint)}/{len(eval_tasks)}] "
                  f"acc={total_c/len(checkpoint):.1%} | "
                  f"D={per_type['D']['accuracy']:.1%} "
                  f"M1={per_type['M1']['accuracy']:.1%} "
                  f"M2={per_type['M2']['accuracy']:.1%}")

    per_type = compute_per_type(checkpoint)
    total_c = sum(1 for r in checkpoint if r["correct"])
    _save(out_path, model_name, checkpoint, per_type, total_c)

    print("\n" + "=" * 50)
    print(f"Routed prompting 결과 ({model_name})")
    print(f"전체: {total_c}/{len(checkpoint)} = {total_c/len(checkpoint):.1%}")
    for t in ["D", "M1", "M2"]:
        s = per_type[t]
        print(f"  {t}: N={s['total']}, acc={s['accuracy']:.1%}, fail={s['failure_rate']:.1%}")
    print(f"저장: {out_path}")


def _save(out_path, model_name, results, per_type, total_c):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model": model_name,
        "method": "taxonomy_routed",
        "routing": {
            "D": "direct_prompt",
            "M1": "gsd_cot_prompt",
            "M2": "counting_specific_prompt",
        },
        "accuracy": round(total_c / len(results), 4) if results else 0,
        "correct": total_c,
        "total": len(results),
        "per_type": per_type,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-vl:8b")
    parser.add_argument("--out", default="annotation/routed_eval_qwen3vl_8b.json")
    args = parser.parse_args()
    out_path = ROOT / args.out
    run_routed_eval(args.model, out_path)


if __name__ == "__main__":
    main()
