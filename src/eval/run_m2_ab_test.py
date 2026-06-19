"""
M2 prompt A/B test at FIXED resolution (fair comparison for routed prompting).

For each backbone, evaluate the 83 eval-split M2 tasks under TWO prompts at the
SAME image resolution (1024px):
  - direct   : the plain direct-prompting baseline prompt
  - counting : the counting-specific routed prompt

This isolates the prompt effect from any resolution confound, because the
original backbone_*.json baselines used full-resolution images whereas the
routed M2 inference downsamples to 1024px. Here both conditions share 1024px.

M1 prompt effect is already measured cleanly inside gsd_ablation.json
(baseline vs few_shot at identical resolution), so only M2 needs this A/B run.

Usage:
    python src/eval/run_m2_ab_test.py --model qwen3-vl:8b --out m2ab_qwen3vl_8b.json
    python src/eval/run_m2_ab_test.py --model qwen2.5vl:7b --out m2ab_qwen25vl_7b.json
    python src/eval/run_m2_ab_test.py --model blaifa/InternVL3_5:8b --out m2ab_internvl35_8b.json
"""
import argparse
import base64
import io
import json
import sys
import time
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

BENCH_PATH = ROOT / "data/thinkgeo/ThinkGeoBench.json"
EVAL_REF   = ROOT / "annotation/backbone_qwen3vl_8b.json"   # defines the 83 M2 eval IDs
DATA_ROOT  = ROOT / "data/thinkgeo"

MAX_IMAGE_PX = 1024

DIRECT_PROMPT = (
    "Look at the image carefully and answer the following question concisely.\n"
    "Question: {query}\n"
    "Answer:"
)
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
                s = MAX_IMAGE_PX / max(w, h)
                img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode()
    return None


def main(model: str, out_file: str):
    out_path = ROOT / "annotation" / out_file
    with open(BENCH_PATH, encoding="utf-8") as f:
        bench = json.load(f)
    with open(EVAL_REF, encoding="utf-8") as f:
        ref = json.load(f)

    m2_ids = [r["task_id"] for r in ref["direct"]["results"] if r["gt_type"] == "M2"]
    m2_tasks = [(tid, bench[tid]) for tid in m2_ids if tid in bench]
    print(f"M2 A/B test on {model}: {len(m2_tasks)} tasks × 2 prompts @ {MAX_IMAGE_PX}px")

    # checkpoint
    results = {"direct": {}, "counting": {}}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            saved = json.load(f)
        results["direct"] = {r["task_id"]: r for r in saved.get("direct", [])}
        results["counting"] = {r["task_id"]: r for r in saved.get("counting", [])}
        print(f"  Checkpoint: direct={len(results['direct'])} counting={len(results['counting'])}")

    vlm = OllamaVLM(model)

    # Cache images once (both prompts use same image)
    for i, (tid, task) in enumerate(m2_tasks):
        need_direct = tid not in results["direct"]
        need_count = tid not in results["counting"]
        if not need_direct and not need_count:
            continue

        query = task.get("dialogs", [{}])[0].get("content", "")
        gt_answer = task.get("gt_answer")
        img_b64 = load_image_b64(task)

        for cond, prompt_tmpl in [("direct", DIRECT_PROMPT), ("counting", COUNTING_PROMPT)]:
            if tid in results[cond]:
                continue
            prompt = prompt_tmpl.format(query=query)
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
            results[cond][tid] = {
                "task_id": tid, "condition": cond,
                "correct": correct, "answer": str(answer)[:200],
            }

        if (i + 1) % 10 == 0 or (i + 1) == len(m2_tasks):
            _save(out_path, model, results)
            df = 1 - sum(r["correct"] for r in results["direct"].values()) / max(len(results["direct"]), 1)
            cf = 1 - sum(r["correct"] for r in results["counting"].values()) / max(len(results["counting"]), 1)
            print(f"  [{i+1}/{len(m2_tasks)}] direct_fail={df*100:.1f}% counting_fail={cf*100:.1f}%")

    _save(out_path, model, results)

    d = results["direct"].values(); c = results["counting"].values()
    df = 1 - sum(r["correct"] for r in d) / len(d)
    cf = 1 - sum(r["correct"] for r in c) / len(c)
    print("\n" + "=" * 50)
    print(f"M2 A/B @ {MAX_IMAGE_PX}px ({model})")
    print(f"  direct prompt:   fail={df*100:.1f}%  (N={len(d)})")
    print(f"  counting prompt: fail={cf*100:.1f}%  (N={len(c)})")
    print(f"  counting effect: {(cf-df)*100:+.1f}pp")
    print(f"  Saved: {out_path}")


def _save(out_path, model, results):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d = list(results["direct"].values())
    c = list(results["counting"].values())
    df = 1 - sum(r["correct"] for r in d) / len(d) if d else 0
    cf = 1 - sum(r["correct"] for r in c) / len(c) if c else 0
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model, "resolution_px": MAX_IMAGE_PX,
            "n_m2": len(d),
            "direct_fail_rate": round(df, 4),
            "counting_fail_rate": round(cf, 4),
            "counting_effect_pp": round((cf - df) * 100, 1),
            "direct": d, "counting": c,
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.model, args.out)
