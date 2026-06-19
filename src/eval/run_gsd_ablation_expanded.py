"""
GSD Injection Ablation Study - Expanded (N=161)

기존 40개(eval split M1) + 신규 121개(excluded split M1, GT 있음)
= 총 161개로 ablation 확장.

기존 gsd_ablation.json 체크포인트를 그대로 이어받으며,
신규 121개 결과를 동일 파일에 추가 저장한다.

완료 후 기존 40개 + 신규 121개를 합산한 통계를 출력한다.

사용법:
    python src/eval/run_gsd_ablation_expanded.py
    python src/eval/run_gsd_ablation_expanded.py --models qwen2.5vl:7b qwen3-vl:8b
    python src/eval/run_gsd_ablation_expanded.py --stats-only   # 추론 없이 통계만 출력
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

BENCH_PATH  = ROOT / "data/thinkgeo/ThinkGeoBench.json"
LABELS_PATH = ROOT / "annotation/thinkgeo_taxonomy_labels.json"
DATA_ROOT   = ROOT / "data/thinkgeo"
REF_PATH    = ROOT / "annotation/backbone_qwen25vl_7b.json"
OUT_PATH    = ROOT / "annotation/gsd_ablation.json"   # 기존과 동일 파일 (이어쓰기)

VLM_MODELS = ["qwen2.5vl:7b", "qwen3-vl:8b", "blaifa/InternVL3_5:8b"]

# 제외된 M1 중 GT 있는 121개 task_id (분석으로 확정)
EXPANSION_IDS = [
    '0', '9', '11', '14', '17', '20', '24', '26', '32', '33', '36', '38',
    '44', '49', '51', '53', '57', '63', '65', '67', '77', '85', '90', '94',
    '105', '110', '116', '117', '123', '130', '136', '154', '159', '160',
    '164', '167', '169', '170', '172', '173', '174', '181', '182', '183',
    '188', '191', '192', '205', '206', '208', '213', '216', '222', '224',
    '225', '226', '238', '250', '252', '254', '258', '260', '262', '271',
    '274', '276', '278', '287', '306', '307', '308', '309', '312', '313',
    '318', '320', '324', '325', '328', '329', '330', '331', '332', '333',
    '334', '400', '401', '402', '403', '404', '405', '406', '407', '408',
    '409', '410', '411', '412', '413', '414', '415', '416', '417', '418',
    '419', '420', '421', '422', '423', '424', '425', '426', '427', '428',
    '429', '430', '431', '432', '433', '434', '435',
]


# ── 프롬프트 템플릿 (기존과 동일) ──────────────────────────────────────────────

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

DEFAULT_GSD = "0.3"


def _extract_gsd(query: str) -> str:
    import re
    # GSD = 0.015  /  GSD: 0.015  /  GSD 0.015 (공백 구분)
    m = re.search(r'gsd\s*[=:\s]\s*([\d.]+)', query, re.IGNORECASE)
    if m:
        return m.group(1)
    # 0.015 m/px  /  0.015 m/pixel  /  0.015m/px
    m = re.search(r'([\d.]+)\s*m\s*/\s*p(?:ixel|x)\b', query, re.IGNORECASE)
    if m:
        return m.group(1)
    # scale of 0.139 m per pixel  /  0.139 meters per pixel
    m = re.search(r'([\d.]+)\s*m(?:eter)?s?\s+per\s+pixel', query, re.IGNORECASE)
    if m:
        return m.group(1)
    # 0.0917 m/pixel resolution  (slash+word boundary variant)
    m = re.search(r'([\d.]+)\s*(?:meters?|m)\s*/\s*pixel', query, re.IGNORECASE)
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


def get_existing_m1_ids() -> set[str]:
    """기존 gsd_ablation.json에서 이미 추론된 task_id 집합 반환."""
    if not OUT_PATH.exists():
        return set()
    with open(OUT_PATH, encoding="utf-8") as f:
        ckpt = json.load(f)
    # 어느 모델/조건이든 한 번이라도 완료된 태스크
    seen = set()
    for cond_data in ckpt.values():
        for results in cond_data.values():
            for r in results:
                seen.add(str(r["task_id"]))
    return seen


def print_stats(checkpoint: dict, orig_ids: set[str], expand_ids: set[str]):
    """기존 40개 / 신규 121개 / 합산 161개 통계 출력."""
    conditions = list(PROMPT_BUILDERS.keys())
    print()
    print("=" * 75)
    print(f"{'Model':22s} {'Cond':10s} {'orig(40)':>10} {'new(121)':>10} {'all(161)':>10}")
    print("-" * 75)

    for model_key, cond_data in checkpoint.items():
        for cond in conditions:
            results = cond_data.get(cond, [])
            orig = [r for r in results if str(r["task_id"]) in orig_ids]
            new  = [r for r in results if str(r["task_id"]) in expand_ids]
            all_ = orig + new

            def fail_pct(lst):
                if not lst:
                    return "   -"
                return f"{sum(1 for r in lst if not r['correct'])/len(lst)*100:6.1f}%"

            print(
                f"{model_key:22s} {cond:10s}"
                f" {fail_pct(orig):>10}"
                f" {fail_pct(new):>10}"
                f" {fail_pct(all_):>10}"
            )
        print()
    print("=" * 75)


def run_expansion(models: list[str], stats_only: bool = False):
    with open(BENCH_PATH, encoding="utf-8") as f:
        bench = json.load(f)

    # 기존 eval M1 40개 ID
    with open(REF_PATH, encoding="utf-8") as f:
        ref = json.load(f)
    orig_ids = {str(r["task_id"]) for r in ref["direct"]["results"] if r["gt_type"] == "M1"}

    expand_ids = set(EXPANSION_IDS)
    new_tasks = [(tid, bench[tid]) for tid in EXPANSION_IDS if tid in bench]
    print(f"기존 eval M1: {len(orig_ids)}개")
    print(f"확장 M1:      {len(new_tasks)}개")
    print(f"합산:         {len(orig_ids) + len(new_tasks)}개")

    # 체크포인트 로드
    checkpoint: dict = {}
    if OUT_PATH.exists():
        with open(OUT_PATH, encoding="utf-8") as f:
            checkpoint = json.load(f)
        done_count = sum(
            len(v) for cd in checkpoint.values() for v in cd.values()
        )
        print(f"체크포인트: {done_count}개 결과 로드됨")

    if stats_only:
        print_stats(checkpoint, orig_ids, expand_ids)
        return

    conditions = list(PROMPT_BUILDERS.keys())
    total = len(models) * len(conditions) * len(new_tasks)
    done = 0

    for model_name in models:
        vlm = OllamaVLM(model_name=model_name)
        model_key = model_name.replace("/", "_").replace(":", "_")

        if model_key not in checkpoint:
            checkpoint[model_key] = {}

        for condition in conditions:
            if condition not in checkpoint[model_key]:
                checkpoint[model_key][condition] = []

            already_done = {str(r["task_id"]) for r in checkpoint[model_key][condition]}
            remaining = [(tid, t) for tid, t in new_tasks if tid not in already_done]

            if not remaining:
                done += len(new_tasks)
                print(f"  [{model_name}][{condition}] 이미 완료 ({len(new_tasks)}개)")
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

                checkpoint[model_key][condition].append({
                    "task_id": tid,
                    "condition": condition,
                    "correct": correct,
                    "answer": answer[:200],
                    "gsd_used": gsd,
                    "source": "expanded",   # 신규 태스크 표시
                })
                done += 1

                if done % 10 == 0:
                    with open(OUT_PATH, "w", encoding="utf-8") as f:
                        json.dump(checkpoint, f, indent=2, ensure_ascii=False)
                    print(f"    [{done}/{total} {done/total*100:.0f}%] saved")

    # 최종 저장
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)

    print_stats(checkpoint, orig_ids, expand_ids)
    print(f"\n결과 저장: {OUT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=VLM_MODELS)
    parser.add_argument("--stats-only", action="store_true",
                        help="추론 없이 현재 체크포인트 통계만 출력")
    args = parser.parse_args()
    run_expansion(args.models, stats_only=args.stats_only)


if __name__ == "__main__":
    main()
