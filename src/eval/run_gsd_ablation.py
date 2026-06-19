"""
Experiment B: GSD Injection Ablation Study
M1 태스크 40개에 대해 4가지 프롬프트 조건 × 3개 VLM 실험.

조건:
  baseline  : GSD 없이 직접 프롬프팅 (기존 결과 재확인)
  gsd_value : GSD 값만 주입 (기존 파일럿)
  cot       : 단위 변환 절차 명시 (Chain-of-Thought)
  few_shot  : 풀이 예시 포함
  cot_fs    : CoT + few-shot 결합

사용법:
    python src/eval/run_gsd_ablation.py
    python src/eval/run_gsd_ablation.py --models qwen2.5vl:7b qwen3-vl:8b
"""
import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval.evaluator import evaluate_answer
from models.ollama_vlm import OllamaVLM

BENCH_PATH   = ROOT / "data/thinkgeo/ThinkGeoBench.json"
LABELS_PATH  = ROOT / "annotation/thinkgeo_taxonomy_labels.json"
DATA_ROOT    = ROOT / "data/thinkgeo"
REF_PATH     = ROOT / "annotation/backbone_qwen25vl_7b.json"
OUT_PATH     = ROOT / "annotation/gsd_ablation.json"

VLM_MODELS = ["qwen2.5vl:7b", "qwen3-vl:8b", "blaifa/InternVL3_5:8b"]

# ── 프롬프트 템플릿 ────────────────────────────────────────────────────────────

def _base(query: str, gsd: str | None = None) -> str:
    return (
        "Look at the image carefully and answer the following question concisely.\n"
        f"Question: {query}\n"
        "Answer:"
    )

def _gsd_value(query: str, gsd: str) -> str:
    return (
        "Look at the image carefully and answer the following question concisely.\n"
        f"Note: the image GSD is {gsd} m/px. "
        f"To convert pixel distances to real-world meters, multiply pixel count by {gsd}.\n"
        f"Question: {query}\n"
        "Answer:"
    )

def _cot(query: str, gsd: str) -> str:
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

def _few_shot(query: str, gsd: str) -> str:
    return (
        "Look at the image carefully and answer the following question concisely.\n"
        f"The image GSD is {gsd} m/px.\n\n"
        "Example: If two buildings are 60 pixels apart and GSD = 0.3 m/px, "
        "then distance = 60 × 0.3 = 18 m.\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

def _cot_fs(query: str, gsd: str) -> str:
    return (
        "Look at the image carefully and answer the following question.\n"
        f"The image GSD is {gsd} m/px.\n\n"
        "Example: If two buildings are 60 pixels apart and GSD = 0.3 m/px, "
        "then distance = 60 × 0.3 = 18 m.\n\n"
        "To answer spatial metric questions:\n"
        "  Step 1: Identify the relevant objects or distances in the image.\n"
        "  Step 2: Estimate the pixel distance or area.\n"
        f"  Step 3: Multiply by {gsd} to convert pixels to meters.\n"
        "  Step 4: State your final answer with units.\n"
        f"Question: {query}\n"
        "Answer:"
    )

PROMPT_BUILDERS = {
    "baseline":  lambda q, gsd: _base(q),
    "gsd_value": _gsd_value,
    "cot":       _cot,
    "few_shot":  _few_shot,
    "cot_fs":    _cot_fs,
}

DEFAULT_GSD = "0.3"   # ThinkGeo 대표 GSD; 태스크별 GSD가 있으면 교체


def _extract_gsd(query: str) -> str:
    """쿼리에서 GSD 값 추출. 없으면 DEFAULT_GSD 반환."""
    import re
    m = re.search(r'gsd\s*[=:]\s*([\d.]+)', query, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r'([\d.]+)\s*m/px', query, re.IGNORECASE)
    if m:
        return m.group(1)
    return DEFAULT_GSD


def encode_image(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def get_image_path(task: dict) -> Path | None:
    for f in task.get("files", []):
        if f.get("type") == "image" and f.get("path"):
            p = DATA_ROOT / f["path"]
            if p.exists():
                return p
    return None


def run_ablation(models: list[str]):
    # 데이터 로드
    with open(BENCH_PATH, encoding="utf-8") as f:
        bench = json.load(f)
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)
    with open(REF_PATH, encoding="utf-8") as f:
        ref = json.load(f)

    # M1 태스크 40개 추출
    m1_ids = [r["task_id"] for r in ref["direct"]["results"] if r["gt_type"] == "M1"]
    m1_tasks = [(tid, bench[tid]) for tid in m1_ids if tid in bench]
    print(f"M1 태스크: {len(m1_tasks)}개")

    # 체크포인트 로드
    checkpoint: dict = {}
    if OUT_PATH.exists():
        with open(OUT_PATH, encoding="utf-8") as f:
            checkpoint = json.load(f)
        print(f"체크포인트 발견: {sum(len(v) for v in checkpoint.values())} 결과 로드됨")

    conditions = list(PROMPT_BUILDERS.keys())
    total_tasks = len(models) * len(conditions) * len(m1_tasks)
    done = 0

    for model_name in models:
        vlm = OllamaVLM(model_name=model_name)
        model_key = model_name.replace("/", "_").replace(":", "_")

        if model_key not in checkpoint:
            checkpoint[model_key] = {}

        for condition in conditions:
            cond_key = condition
            if cond_key not in checkpoint[model_key]:
                checkpoint[model_key][cond_key] = []

            already_done = {r["task_id"] for r in checkpoint[model_key][cond_key]}
            remaining = [(tid, t) for tid, t in m1_tasks if tid not in already_done]

            if not remaining:
                done += len(m1_tasks)
                print(f"  [{model_name}][{condition}] 이미 완료")
                continue

            print(f"\n  [{model_name}][{condition}] {len(remaining)}개 실행 중...")
            prompt_fn = PROMPT_BUILDERS[condition]

            for tid, task in remaining:
                query = task["dialogs"][0]["content"] if task.get("dialogs") else ""
                gt_answer = task.get("gt_answer")
                img_path = get_image_path(task)
                gsd = _extract_gsd(query)

                prompt = prompt_fn(query, gsd)

                try:
                    resp = vlm.generate(prompt, image_path=img_path, max_tokens=512)
                    answer = resp.text
                    correct = evaluate_answer(answer, gt_answer)["correct"]
                except Exception as e:
                    answer = f"ERROR: {e}"
                    correct = False

                checkpoint[model_key][cond_key].append({
                    "task_id": tid,
                    "condition": condition,
                    "correct": correct,
                    "answer": answer[:200],
                    "gsd_used": gsd,
                })
                done += 1

                # 10개마다 저장
                if done % 10 == 0:
                    with open(OUT_PATH, "w", encoding="utf-8") as f:
                        json.dump(checkpoint, f, indent=2, ensure_ascii=False)
                    elapsed_pct = done / total_tasks * 100
                    print(f"    [{done}/{total_tasks} {elapsed_pct:.0f}%] saved")

    # 최종 저장
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)

    # 결과 집계 출력
    print("\n" + "=" * 65)
    print(f"{'Model':20s} {'Condition':12s} {'Fail%':>7} {'N':>5}")
    print("-" * 65)

    summary = {}
    for model_key, cond_data in checkpoint.items():
        for cond, results in cond_data.items():
            if not results:
                continue
            n = len(results)
            fail = sum(1 for r in results if not r["correct"])
            fail_pct = fail / n * 100
            summary[f"{model_key}|{cond}"] = {"fail_pct": fail_pct, "n": n}
            print(f"{model_key:20s} {cond:12s} {fail_pct:>7.1f}% {n:>5}")

    print("=" * 65)
    print(f"결과 저장: {OUT_PATH}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=VLM_MODELS)
    args = parser.parse_args()
    run_ablation(args.models)


if __name__ == "__main__":
    main()
