"""
3-type taxonomy (D/M1/M2) 기준 Router 정확도 평가.
- Rule-based: gsd_classifier.py (D/M1/M2)
- LLM: gpt-oss:20b, qwen3.5:9b

사용법:
    cd <repo root>
    python src/router/eval_router_3type.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent.parent


def load_gt() -> dict[str, str]:
    """eval split 189개의 primary GT type (M1 > M2 > D)."""
    with open(ROOT / "annotation/thinkgeo_taxonomy_labels.json", encoding="utf-8") as f:
        labels = json.load(f)
    with open(ROOT / "annotation/backbone_qwen3vl_8b.json", encoding="utf-8") as f:
        backbone = json.load(f)
    eval_ids = {r["task_id"] for r in backbone["direct"]["results"]}

    def primary(types):
        for t in ["M1", "M2", "D"]:
            if t in types:
                return t
        return types[0]

    return {
        tid: primary(labels[tid]["annotation"]["types"])
        for tid in eval_ids
        if tid in labels
    }


def load_queries(eval_ids: set) -> dict[str, str]:
    with open(ROOT / "annotation/thinkgeo_taxonomy_labels.json", encoding="utf-8") as f:
        labels = json.load(f)
    return {tid: labels[tid]["query"] for tid in eval_ids if tid in labels}


def compute_metrics(preds: dict[str, str], gt: dict[str, str]) -> dict:
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    correct = 0
    for tid, pred in preds.items():
        gold = gt.get(tid)
        if gold is None:
            continue
        if pred == gold:
            correct += 1
        for t in ["D", "M1", "M2"]:
            in_p = (pred == t); in_g = (gold == t)
            if in_p and in_g:   tp[t] += 1
            elif in_p:          fp[t] += 1
            elif in_g:          fn[t] += 1

    total = len(preds)
    per_type = {}
    f1s = []
    for t in ["D", "M1", "M2"]:
        p = tp[t] / (tp[t] + fp[t]) if (tp[t] + fp[t]) > 0 else 0
        r = tp[t] / (tp[t] + fn[t]) if (tp[t] + fn[t]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        support = tp[t] + fn[t]
        per_type[t] = {"precision": round(p, 4), "recall": round(r, 4),
                       "f1": round(f1, 4), "support": support}
        if support > 0:
            f1s.append(f1)

    return {
        "accuracy": round(correct / total, 4) if total > 0 else 0,
        "correct": correct,
        "total": total,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0,
        "per_type": per_type,
    }


def print_metrics(name: str, m: dict):
    print(f"\n=== {name} ===")
    print(f"  Accuracy : {m['correct']}/{m['total']} = {m['accuracy']:.1%}")
    print(f"  Macro-F1 : {m['macro_f1']:.4f}")
    print(f"  {'Type':<4} {'P':>7} {'R':>7} {'F1':>7} {'N':>6}")
    print(f"  {'-'*38}")
    for t, s in m["per_type"].items():
        if s["support"] == 0:
            continue
        print(f"  {t:<4} {s['precision']:>7.3f} {s['recall']:>7.3f} {s['f1']:>7.3f} {s['support']:>6}")


def main():
    gt = load_gt()
    queries = load_queries(set(gt.keys()))
    print(f"평가 태스크: {len(gt)}개")
    print("GT 분포:", Counter(gt.values()))

    results = {}

    # --- Rule-based Router ---
    print("\n[Rule-based Router 실행 중...]")
    from taxonomy.gsd_classifier import GSDSensitivityClassifier
    classifier = GSDSensitivityClassifier()
    rule_preds = {tid: classifier.classify(queries[tid]).task_type.value
                  for tid in gt if tid in queries}
    m_rule = compute_metrics(rule_preds, gt)
    print_metrics("Rule-based (GSD Classifier)", m_rule)
    results["rule"] = m_rule

    # --- LLM Router: gpt-oss:20b ---
    print("\n[LLM Router 실행 중 (gpt-oss:20b)...]")
    try:
        from models.ollama_llm import OllamaLLM
        from router.llm_router import LLMRouter
        llm = OllamaLLM(model_name="gpt-oss:20b", think=True)  # gpt-oss:20b는 think=True 필요
        router = LLMRouter(llm, think=False)
        gpt_preds = {}
        items = [(tid, queries[tid]) for tid in gt if tid in queries]
        for i, (tid, q) in enumerate(items):
            gpt_preds[tid] = router.route(q).value
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(items)}...")
        m_gpt = compute_metrics(gpt_preds, gt)
        print_metrics("LLM Router (gpt-oss:20b)", m_gpt)
        results["gpt_oss_20b"] = m_gpt
    except Exception as e:
        print(f"  오류: {e}")

    # --- LLM Router: qwen3.5:9b ---
    print("\n[LLM Router 실행 중 (qwen3.5:9b)...]")
    try:
        from models.ollama_llm import OllamaLLM
        from router.llm_router import LLMRouter
        llm2 = OllamaLLM(model_name="qwen3.5:9b")
        router2 = LLMRouter(llm2, think=False)
        q35_preds = {}
        items = [(tid, queries[tid]) for tid in gt if tid in queries]
        for i, (tid, q) in enumerate(items):
            q35_preds[tid] = router2.route(q).value
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(items)}...")
        m_q35 = compute_metrics(q35_preds, gt)
        print_metrics("LLM Router (qwen3.5:9b)", m_q35)
        results["qwen35_9b"] = m_q35
    except Exception as e:
        print(f"  오류: {e}")

    out_path = ROOT / "annotation/router_eval_3type.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    import os
    os.chdir(ROOT)
    main()
