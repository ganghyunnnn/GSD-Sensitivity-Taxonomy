"""
M2 counting-prompt search at FIXED 1024px resolution.

The basic counting prompt showed weak/inconsistent effect vs direct. Here we test
several more sophisticated M2 counting strategies, all at the same 1024px so the
comparison is prompt-only (no resolution confound):

  direct      : plain baseline prompt (control)
  counting    : basic counting instruction (already tested; included as anchor)
  grid        : instruct systematic region-by-region scan to avoid missed/double counts
  cot_count   : chain-of-thought enumeration (list then total)
  systematic  : explicit "scan rows top-to-bottom, tally, then sum" procedure

Usage:
    python src/eval/run_m2_prompt_search.py --model qwen3-vl:8b --out m2search_qwen3vl_8b.json
"""
import argparse, base64, io, json, sys, time
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
EVAL_REF   = ROOT / "annotation/backbone_qwen3vl_8b.json"
DATA_ROOT  = ROOT / "data/thinkgeo"
MAX_IMAGE_PX = 1024

PROMPTS = {
    "direct": (
        "Look at the image carefully and answer the following question concisely.\n"
        "Question: {query}\nAnswer:"
    ),
    "grid": (
        "You are counting objects in an overhead remote-sensing image.\n"
        "Mentally divide the image into a 3x3 grid. Count the target objects in each "
        "cell one cell at a time, then sum across all nine cells. This avoids missing "
        "objects and double-counting at boundaries.\n"
        "Question: {query}\n"
        "Give only the final total number as your answer.\nAnswer:"
    ),
    "cot_count": (
        "You are counting objects in an overhead remote-sensing image.\n"
        "First, briefly enumerate each distinct target object you can identify "
        "(e.g., 1, 2, 3, ...). Then state the final total.\n"
        "Question: {query}\n"
        "End your response with 'Total: <number>'.\nAnswer:"
    ),
    "systematic": (
        "You are an expert image analyst counting objects in aerial imagery.\n"
        "Procedure:\n"
        "  1. Scan the image row by row from top to bottom.\n"
        "  2. Keep a running tally of each distinct target instance.\n"
        "  3. Do not count the same object twice; ignore objects that are not the target.\n"
        "  4. Report the final count as a single number.\n"
        "Question: {query}\nAnswer:"
    ),
}


def load_image_b64(task):
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
                img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
            buf = io.BytesIO(); img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode()
    return None


def main(model, out_file, conditions):
    out_path = ROOT / "annotation" / out_file
    with open(BENCH_PATH, encoding="utf-8") as f:
        bench = json.load(f)
    with open(EVAL_REF, encoding="utf-8") as f:
        ref = json.load(f)
    m2_ids = [r["task_id"] for r in ref["direct"]["results"] if r["gt_type"] == "M2"]
    m2_tasks = [(t, bench[t]) for t in m2_ids if t in bench]
    print(f"M2 prompt search on {model}: {len(m2_tasks)} tasks x {len(conditions)} prompts @ {MAX_IMAGE_PX}px")

    results = {c: {} for c in conditions}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            saved = json.load(f)
        for c in conditions:
            results[c] = {r["task_id"]: r for r in saved.get("conditions", {}).get(c, [])}

    vlm = OllamaVLM(model)
    for i, (tid, task) in enumerate(m2_tasks):
        if all(tid in results[c] for c in conditions):
            continue
        query = task.get("dialogs", [{}])[0].get("content", "")
        gt = task.get("gt_answer")
        b64 = load_image_b64(task)
        for c in conditions:
            if tid in results[c]:
                continue
            prompt = PROMPTS[c].format(query=query)
            try:
                resp = vlm.generate_b64(prompt, b64, max_tokens=256) if b64 \
                    else vlm.generate(prompt, image_path=None, max_tokens=256)
                ans = resp.text
                ok = evaluate_answer(ans, gt)["correct"]
            except Exception as e:
                ans = f"ERROR: {e}"; ok = False
            results[c][tid] = {"task_id": tid, "correct": ok, "answer": str(ans)[:200]}
        if (i+1) % 10 == 0 or (i+1) == len(m2_tasks):
            _save(out_path, model, conditions, results)
            msg = " ".join(f"{c}={(1-sum(r['correct'] for r in results[c].values())/max(len(results[c]),1))*100:.0f}%" for c in conditions)
            print(f"  [{i+1}/{len(m2_tasks)}] {msg}")
    _save(out_path, model, conditions, results)
    print("\n=== M2 prompt search ({}) fail rates ===".format(model))
    for c in conditions:
        rs = results[c].values()
        fr = (1 - sum(r["correct"] for r in rs)/len(rs))*100
        print(f"  {c:12s}: {fr:.1f}%")
    print(f"  Saved: {out_path}")


def _save(out_path, model, conditions, results):
    payload = {"model": model, "resolution_px": MAX_IMAGE_PX,
               "fail_rates": {}, "conditions": {}}
    for c in conditions:
        rs = list(results[c].values())
        payload["conditions"][c] = rs
        payload["fail_rates"][c] = round(1 - sum(r["correct"] for r in rs)/len(rs), 4) if rs else None
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", nargs="+",
                    default=["direct", "grid", "cot_count", "systematic"])
    args = ap.parse_args()
    main(args.model, args.out, args.conditions)
