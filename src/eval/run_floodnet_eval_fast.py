"""
Experiment B (fast): FloodNet D/M2 evaluation with image resizing.

D  -> Condition_Recognition + Yes_No  (subsample 100 each)
M2 -> Simple_Counting + Complex_Counting (subsample 100 each)
Model: qwen3-vl:8b, direct prompting, images resized to 1024px

Usage:
    python src/eval/run_floodnet_eval_fast.py
"""
import base64
import io
import json
import random
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

FLOODNET_Q   = ROOT / "data/FloodNet/Track2/Questions/Training Question.json"
FLOODNET_IMG = ROOT / "data/FloodNet/Track2/Images/Train_Image"
# OUT_PATH and model set via CLI in main(); default matches original run
OUT_PATH     = ROOT / "annotation/floodnet_backbone_qwen3vl_8b.json"

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

MAX_IMAGE_PX = 512
SAMPLE_PER_TYPE = 25  # 25 per question type × 4 types = 100 tasks total (~2hr)
SEED = 42


def load_image_b64(img_path: Path) -> str | None:
    from PIL import Image
    if not img_path.exists():
        return None
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if max(w, h) > MAX_IMAGE_PX:
            scale = MAX_IMAGE_PX / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def score_answer(pred: str, gt: str, q_type: str) -> bool:
    import re
    pred = pred.strip().lower()
    gt = str(gt).strip().lower()
    num_words = {"zero": "0", "one": "1", "two": "2", "three": "3",
                 "four": "4", "five": "5", "six": "6", "seven": "7",
                 "eight": "8", "nine": "9", "ten": "10"}
    for w, n in num_words.items():
        pred = pred.replace(w, n)
        gt = gt.replace(w, n)
    if pred == gt:
        return True
    if q_type == "Condition_Recognition" and gt in pred:
        return True
    if q_type == "Yes_No":
        for token in ["yes", "no"]:
            if gt == token and token in pred.split():
                return True
    pred_nums = re.findall(r"\b\d+\b", pred)
    gt_nums = re.findall(r"\b\d+\b", gt)
    if pred_nums and gt_nums and pred_nums[0] == gt_nums[0]:
        return True
    return False


def main():
    random.seed(SEED)

    with open(FLOODNET_Q, encoding="utf-8") as f:
        all_q = json.load(f)

    by_type: dict[str, list] = defaultdict(list)
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

    print("FloodNet subsample:")
    sampled = []
    for qt, items in by_type.items():
        n = min(SAMPLE_PER_TYPE, len(items))
        sel = random.sample(items, n)
        sampled.extend(sel)
        print(f"  {qt} ({TYPE_MAP[qt]}): {n}")
    print(f"Total: {len(sampled)}")

    # checkpoint
    results = []
    done_ids = set()
    if OUT_PATH.exists():
        with open(OUT_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        results = saved.get("results", [])
        done_ids = {r["idx"] for r in results}
        print(f"Checkpoint: {len(done_ids)} done")

    vlm = OllamaVLM("qwen3-vl:8b")

    for i, item in enumerate(sampled):
        if item["idx"] in done_ids:
            continue

        prompt = DIRECT_PROMPT.format(query=item["question"])
        img_b64 = load_image_b64(Path(item["img_path"]))

        t0 = time.monotonic()
        try:
            if img_b64:
                resp = vlm.generate_b64(prompt, img_b64, max_tokens=64)
            else:
                resp = vlm.generate(prompt, image_path=None, max_tokens=64)
            answer = resp.text
            correct = score_answer(answer, item["gt"], item["q_type"])
            elapsed = round(time.monotonic() - t0, 1)
        except Exception as e:
            answer = f"ERROR: {e}"
            correct = False
            elapsed = 0.0

        results.append({
            "idx": item["idx"],
            "q_type": item["q_type"],
            "taxonomy": item["taxonomy"],
            "correct": correct,
            "answer": answer[:200],
            "gt": item["gt"],
        })
        done_ids.add(item["idx"])

        if len(results) % 20 == 0 or len(results) == len(sampled):
            _save(results)
            pt = _per_tax(results)
            print(f"  [{len(results)}/{len(sampled)}] {elapsed}s | "
                  f"D={pt.get('D',{}).get('accuracy',0):.1%} "
                  f"M2={pt.get('M2',{}).get('accuracy',0):.1%}")

    _save(results)
    pt = _per_tax(results)
    print("\n=== FloodNet D/M2 Results ===")
    for t in ["D", "M2"]:
        s = pt.get(t, {})
        print(f"  {t}: N={s.get('total',0)}, acc={s.get('accuracy',0):.1%}, fail={s.get('failure_rate',0):.1%}")
    d_fail = pt.get("D", {}).get("failure_rate", 0)
    m2_fail = pt.get("M2", {}).get("failure_rate", 0)
    print(f"  D/M2 gap: +{(m2_fail-d_fail)*100:.1f}pp")
    print(f"  Saved: {OUT_PATH}")


def _per_tax(results):
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
        out[t] = {"correct": s["correct"], "total": s["total"],
                  "accuracy": acc, "failure_rate": round(1 - acc, 4)}
    return out


def _save(results):
    pt = _per_tax(results)
    total_c = sum(r["correct"] for r in results)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model": "qwen3-vl:8b",
            "dataset": "FloodNet-Track2-Train",
            "method": "direct_prompting",
            "sample_per_type": SAMPLE_PER_TYPE,
            "accuracy": round(total_c / len(results), 4) if results else 0,
            "correct": total_c,
            "total": len(results),
            "per_taxonomy": pt,
            "results": results,
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
