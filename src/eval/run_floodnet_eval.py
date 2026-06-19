"""
FloodNet D/M2 Backbone Evaluation

D/M2 failure gap replication on FloodNet Training set.
  D  -> Condition_Recognition + Yes_No
  M2 -> Simple_Counting + Complex_Counting

qwen3-vl:8b, direct prompting

Usage:
    python src/eval/run_floodnet_eval.py
    python src/eval/run_floodnet_eval.py --sample-per-type 150
"""
import argparse
import io
import json
import random
import sys
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

FLOODNET_Q   = ROOT / "data/FloodNet/Track2/Questions/Training Question.json"
FLOODNET_IMG = ROOT / "data/FloodNet/Track2/Images/Train_Image"
OUT_PATH     = ROOT / "annotation/floodnet_backbone_qwen3vl_8b.json"

# FloodNet 질문 유형 → Taxonomy 매핑
TYPE_MAP = {
    "Condition_Recognition": "D",
    "Yes_No": "D",
    "Simple_Counting": "M2",
    "Complex_Counting": "M2",
}

DIRECT_PROMPT = (
    "Look at the image carefully and answer the following question concisely.\n"
    "Question: {query}\n"
    "Answer:"
)

MAX_IMAGE_PX = 1024


def load_image_b64(img_path: Path) -> str | None:
    from PIL import Image
    import base64, io
    if not img_path.exists():
        return None
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMAGE_PX:
        scale = MAX_IMAGE_PX / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

SEED = 42


def score_answer(pred: str, gt: str, q_type: str) -> bool:
    """FloodNet 답변 채점."""
    pred = pred.strip().lower()
    gt = str(gt).strip().lower()

    # 숫자 정규화
    num_words = {"zero": "0", "one": "1", "two": "2", "three": "3",
                 "four": "4", "five": "5", "six": "6", "seven": "7",
                 "eight": "8", "nine": "9", "ten": "10"}
    for word, num in num_words.items():
        pred = pred.replace(word, num)
        gt = gt.replace(word, num)

    # 정확 매칭 먼저
    if pred == gt:
        return True

    # 부분 포함 (D 타입 서술형)
    if q_type in ("Condition_Recognition",):
        # flooded / non-flooded
        if gt in pred:
            return True

    # Yes/No 정규화
    if q_type == "Yes_No":
        for token in ["yes", "no"]:
            if gt == token and token in pred.split():
                return True
        return pred == gt

    # 숫자 추출 비교 (Counting)
    import re
    pred_nums = re.findall(r"\b\d+\b", pred)
    gt_nums = re.findall(r"\b\d+\b", gt)
    if pred_nums and gt_nums and pred_nums[0] == gt_nums[0]:
        return True

    return False


def main(sample_per_type: int = 200, model_name: str = "qwen3-vl:8b"):
    random.seed(SEED)

    with open(FLOODNET_Q, encoding="utf-8") as f:
        all_q = json.load(f)

    # 유형별로 분리 및 subsample
    by_type: dict[str, list[dict]] = defaultdict(list)
    for idx, item in all_q.items():
        q_type = item["Question_Type"]
        img_path = FLOODNET_IMG / item["Image_ID"]
        if img_path.exists():
            by_type[q_type].append({
                "idx": idx,
                "image_id": item["Image_ID"],
                "question": item["Question"],
                "gt": item["Ground_Truth"],
                "q_type": q_type,
                "taxonomy": TYPE_MAP[q_type],
                "img_path": str(img_path),
            })

    print("FloodNet Training set 유형별 이미지 있는 샘플:")
    for qt, items in by_type.items():
        print(f"  {qt} ({TYPE_MAP[qt]}): {len(items)} 개 → subsample {min(sample_per_type, len(items))}")

    # subsample
    sampled: list[dict] = []
    for qt, items in by_type.items():
        n = min(sample_per_type, len(items))
        sampled.extend(random.sample(items, n))
    print(f"\n총 평가 샘플: {len(sampled)}")

    # 체크포인트
    results = []
    done_ids = set()
    if OUT_PATH.exists():
        with open(OUT_PATH, encoding="utf-8") as f:
            saved = json.load(f)
            results = saved.get("results", [])
            done_ids = {r["idx"] for r in results}
        print(f"체크포인트: {len(done_ids)}개 이미 완료")

    vlm = OllamaVLM(model_name=model_name)

    for i, item in enumerate(sampled):
        if item["idx"] in done_ids:
            continue

        prompt = DIRECT_PROMPT.format(query=item["question"])
        img_path = Path(item["img_path"])
        img_b64 = load_image_b64(img_path)

        try:
            if img_b64:
                resp = vlm.generate_b64(prompt, img_b64, max_tokens=128)
            else:
                resp = vlm.generate(prompt, image_path=img_path, max_tokens=128)
            answer = resp.text
            correct = score_answer(answer, item["gt"], item["q_type"])
        except Exception as e:
            answer = f"ERROR: {e}"
            correct = False

        results.append({
            "idx": item["idx"],
            "q_type": item["q_type"],
            "taxonomy": item["taxonomy"],
            "correct": correct,
            "answer": answer[:200],
            "gt": item["gt"],
        })
        done_ids.add(item["idx"])

        if len(results) % 50 == 0 or len(results) == len(sampled):
            _save(results, model_name)
            per_tax = _per_taxonomy(results)
            print(f"  [{len(results)}/{len(sampled)}] "
                  f"D={per_tax.get('D', {}).get('accuracy', 0):.1%} "
                  f"M2={per_tax.get('M2', {}).get('accuracy', 0):.1%}")

    _save(results, model_name)

    print("\n" + "=" * 50)
    per_tax = _per_taxonomy(results)
    for tax in ["D", "M2"]:
        s = per_tax.get(tax, {})
        print(f"  {tax}: N={s.get('total', 0)}, "
              f"acc={s.get('accuracy', 0):.1%}, "
              f"fail={s.get('failure_rate', 0):.1%}")
    d_fail = per_tax.get("D", {}).get("failure_rate", 0)
    m2_fail = per_tax.get("M2", {}).get("failure_rate", 0)
    gap = m2_fail - d_fail
    print(f"  D/M2 gap: +{gap*100:.1f}pp")
    print(f"저장: {OUT_PATH}")


def _per_taxonomy(results: list[dict]) -> dict:
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        t = r["taxonomy"]
        stats[t]["total"] += 1
        if r["correct"]:
            stats[t]["correct"] += 1
    out = {}
    for t in ["D", "M2"]:
        s = stats[t]
        acc = round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0.0
        out[t] = {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": acc,
            "failure_rate": round(1 - acc, 4),
        }
    return out


def _save(results: list[dict], model_name: str):
    per_tax = _per_taxonomy(results)
    total_c = sum(1 for r in results if r["correct"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "dataset": "FloodNet-Track2-Train",
            "method": "direct_prompting",
            "accuracy": round(total_c / len(results), 4) if results else 0,
            "correct": total_c,
            "total": len(results),
            "per_taxonomy": per_tax,
            "results": results,
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-vl:8b")
    parser.add_argument("--sample-per-type", type=int, default=200)
    args = parser.parse_args()
    main(sample_per_type=args.sample_per_type, model_name=args.model)
