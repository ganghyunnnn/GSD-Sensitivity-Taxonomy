"""
VLM backbone direct prompting 평가 스크립트.
새 3-type taxonomy 레이블 기준으로 per_type 통계를 생성한다.

사용법:
    cd <repo root>
    python src/eval/run_backbone_eval.py --model blaifa/InternVL3_5:8b --out annotation/backbone_internvl35_8b.json
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.ollama_vlm import OllamaVLM
from eval.evaluator import evaluate_answer

ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = ROOT / "data/thinkgeo"
BENCH_PATH = ROOT / "data/thinkgeo/ThinkGeoBench.json"
LABELS_PATH = ROOT / "annotation/thinkgeo_taxonomy_labels.json"
# eval split IDs: qwen3vl backbone 기준 189개
EVAL_SPLIT_REF = ROOT / "annotation/backbone_qwen3vl_8b.json"


DIRECT_PROMPT = (
    "Look at the image carefully and answer the following question concisely.\n"
    "Question: {query}\n"
    "Answer:"
)


def primary_type(types: list[str]) -> str:
    for t in ["M1", "M2", "D"]:
        if t in types:
            return t
    return types[0]


def get_query(task: dict) -> str:
    dialogs = task.get("dialogs", [])
    if dialogs:
        return dialogs[0].get("content", "")
    return ""


def get_image_paths(task: dict) -> list[Path]:
    paths = []
    for f in task.get("files", []):
        if f.get("type") == "image" and f.get("path"):
            p = DATA_ROOT / f["path"]
            if p.exists():
                paths.append(p)
    return paths


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
        out[t] = {"correct": s["correct"], "total": s["total"], "accuracy": acc}
    return out


def run_eval(model_name: str, out_path: str, limit: int = None):
    with open(BENCH_PATH, encoding="utf-8") as f:
        bench = json.load(f)
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)
    with open(EVAL_SPLIT_REF, encoding="utf-8") as f:
        ref = json.load(f)
    eval_ids = {r["task_id"] for r in ref["direct"]["results"]}

    # eval split 189개만
    eval_tasks = []
    for tid in eval_ids:
        task = bench.get(tid)
        if task and task.get("gt_answer") and tid in labels:
            eval_tasks.append((tid, task))
    eval_tasks.sort(key=lambda x: int(x[0]))
    print(f"평가 대상 태스크: {len(eval_tasks)}개")
    if limit:
        eval_tasks = eval_tasks[:limit]
        print(f"  (제한: {limit}개)")

    vlm = OllamaVLM(model_name=model_name)
    results = []
    errors = 0

    for i, (tid, task) in enumerate(eval_tasks):
        query = get_query(task)
        image_paths = get_image_paths(task)
        gt_answer = task.get("gt_answer")
        types = labels[tid]["annotation"]["types"]
        gt_type = primary_type(types)

        # 이미지가 여러 장이면 첫 번째만 사용 (direct prompting)
        img_path = image_paths[0] if image_paths else None

        try:
            prompt = DIRECT_PROMPT.format(query=query)
            resp = vlm.generate(prompt, image_path=img_path, max_tokens=512)
            answer = resp.text
            eval_result = evaluate_answer(answer, gt_answer)
            correct = eval_result["correct"]
        except Exception as e:
            answer = f"ERROR: {e}"
            correct = False
            errors += 1

        results.append({
            "task_id": tid,
            "gt_type": gt_type,
            "gt_types": types,
            "correct": correct,
            "answer": answer,
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(eval_tasks):
            done = sum(1 for r in results if r["correct"])
            print(f"  [{i+1}/{len(eval_tasks)}] correct={done}, errors={errors}")

    # 집계
    per_type = compute_per_type(results)
    total = len(results)
    correct_total = sum(1 for r in results if r["correct"])

    output = {
        "direct": {
            "agent": f"Direct ({model_name})",
            "accuracy": round(correct_total / total, 4) if total > 0 else 0,
            "correct": correct_total,
            "total": total,
            "per_type": per_type,
            "results": results,
        }
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n결과 저장: {out}")
    print(f"전체: {correct_total}/{total} = {correct_total/total:.1%}")
    for t in ["D", "M1", "M2"]:
        s = per_type[t]
        fr = round(1 - s["accuracy"], 4)
        print(f"  {t}: N={s['total']}, fail={fr:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="blaifa/InternVL3_5:8b")
    parser.add_argument("--out", default="annotation/backbone_internvl35_8b.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_path = ROOT / args.out
    run_eval(args.model, str(out_path), limit=args.limit)
