"""
FloodNet D/M2 evaluation — multi-backbone version.

Balanced 100 per question type (N=400), seed 42, images resized to 512px.
Direct prompting; same protocol as the qwen3-vl run.

Usage:
    python src/eval/run_floodnet_multi.py --model qwen2.5vl:7b --out floodnet_qwen25vl_7b.json
"""
import argparse
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

from models.ollama_vlm import OllamaVLM

FLOODNET_Q   = ROOT / "data/FloodNet/Track2/Questions/Training Question.json"
FLOODNET_IMG = ROOT / "data/FloodNet/Track2/Images/Train_Image"

TYPE_MAP = {
    "Condition_Recognition": "D", "Yes_No": "D",
    "Simple_Counting": "M2", "Complex_Counting": "M2",
}
DIRECT_PROMPT = (
    "Look at the image carefully and answer the following question concisely.\n"
    "Question: {query}\n"
    "Answer:"
)
MAX_IMAGE_PX = 512
SAMPLE_PER_TYPE = 100   # balanced N=400
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
    pred = pred.strip().lower(); gt = str(gt).strip().lower()
    num = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
           "six":"6","seven":"7","eight":"8","nine":"9","ten":"10"}
    for w, n in num.items():
        pred = pred.replace(w, n); gt = gt.replace(w, n)
    if pred == gt:
        return True
    if q_type == "Condition_Recognition" and gt in pred:
        return True
    if q_type == "Yes_No":
        for tok in ["yes", "no"]:
            if gt == tok and tok in pred.split():
                return True
    pn = re.findall(r"\b\d+\b", pred); gn = re.findall(r"\b\d+\b", gt)
    if pn and gn and pn[0] == gn[0]:
        return True
    return False


def per_tax(results):
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        stats[r["taxonomy"]]["total"] += 1
        if r["correct"]:
            stats[r["taxonomy"]]["correct"] += 1
    out = {}
    for t in ["D", "M2"]:
        s = stats[t]
        acc = round(s["correct"] / s["total"], 4) if s["total"] else 0.0
        out[t] = {"correct": s["correct"], "total": s["total"],
                  "accuracy": acc, "failure_rate": round(1 - acc, 4)}
    return out


def save(out_path, model, results):
    pt = per_tax(results)
    total_c = sum(r["correct"] for r in results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model, "dataset": "FloodNet-Track2-Train",
            "method": "direct_prompting", "sample_per_type": SAMPLE_PER_TYPE,
            "accuracy": round(total_c / len(results), 4) if results else 0,
            "correct": total_c, "total": len(results),
            "per_taxonomy": pt, "results": results,
        }, f, indent=2, ensure_ascii=False)


def main(model: str, out_file: str):
    random.seed(SEED)
    out_path = ROOT / "annotation" / out_file

    with open(FLOODNET_Q, encoding="utf-8") as f:
        all_q = json.load(f)
    by_type = defaultdict(list)
    for idx, item in all_q.items():
        qt = item["Question_Type"]
        img = FLOODNET_IMG / item["Image_ID"]
        if img.exists():
            by_type[qt].append({
                "idx": idx, "image_id": item["Image_ID"],
                "question": item["Question"], "gt": item["Ground_Truth"],
                "q_type": qt, "taxonomy": TYPE_MAP[qt], "img_path": str(img),
            })
    sampled = []
    print(f"FloodNet subsample ({model}):")
    for qt, items in by_type.items():
        n = min(SAMPLE_PER_TYPE, len(items))
        sampled.extend(random.sample(items, n))
        print(f"  {qt} ({TYPE_MAP[qt]}): {n}")
    print(f"Total: {len(sampled)}")

    results = []
    done_ids = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            results = json.load(f).get("results", [])
        done_ids = {r["idx"] for r in results}
        print(f"Checkpoint: {len(done_ids)} done")

    vlm = OllamaVLM(model)
    for i, item in enumerate(sampled):
        if item["idx"] in done_ids:
            continue
        prompt = DIRECT_PROMPT.format(query=item["question"])
        b64 = load_image_b64(Path(item["img_path"]))
        t0 = time.monotonic()
        try:
            resp = vlm.generate_b64(prompt, b64, max_tokens=64) if b64 \
                else vlm.generate(prompt, image_path=None, max_tokens=64)
            answer = resp.text
            correct = score_answer(answer, item["gt"], item["q_type"])
        except Exception as e:
            answer = f"ERROR: {e}"; correct = False
        elapsed = round(time.monotonic() - t0, 1)

        results.append({
            "idx": item["idx"], "q_type": item["q_type"],
            "taxonomy": item["taxonomy"], "correct": correct,
            "answer": str(answer)[:200], "gt": item["gt"],
        })
        done_ids.add(item["idx"])

        if len(results) % 20 == 0 or len(results) == len(sampled):
            save(out_path, model, results)
            pt = per_tax(results)
            print(f"  [{len(results)}/{len(sampled)}] {elapsed}s "
                  f"D_fail={pt['D']['failure_rate']*100:.0f}% "
                  f"M2_fail={pt['M2']['failure_rate']*100:.0f}%")

    save(out_path, model, results)
    pt = per_tax(results)
    print("\n=== FloodNet Results ({}) ===".format(model))
    for t in ["D", "M2"]:
        s = pt[t]
        print(f"  {t}: N={s['total']} fail={s['failure_rate']*100:.1f}%")
    print(f"  D/M2 gap: +{(pt['M2']['failure_rate']-pt['D']['failure_rate'])*100:.1f}pp")
    # Yes/No vs Counting subgroup
    yn = [r for r in results if r["q_type"] == "Yes_No"]
    cnt = [r for r in results if r["taxonomy"] == "M2"]
    if yn and cnt:
        ynf = 1 - sum(r["correct"] for r in yn) / len(yn)
        cnf = 1 - sum(r["correct"] for r in cnt) / len(cnt)
        print(f"  Yes/No-D vs Counting-M2: {ynf*100:.1f}% vs {cnf*100:.1f}% = +{(cnf-ynf)*100:.1f}pp")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.model, args.out)
