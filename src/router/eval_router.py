"""
Task Router 정확도 평가.
규칙 기반 vs LLM 기반 Router를 189개 레이블 태스크로 비교.

사용법:
    cd A:/MARA
    python src/router/eval_router.py [--router rule|llm|both]
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from taxonomy.definitions import TaskType
from router.rule_router import RuleRouter
from router.llm_router import LLMRouter


ROOT = Path(__file__).parent.parent.parent


def load_labels(path: str) -> dict[str, dict]:
    with open(ROOT / path, encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(
    predictions: dict[str, list[str]],
    gt: dict[str, list[str]],
) -> dict:
    """
    멀티레이블 평가 지표 계산.
    - Exact Match: 예측 집합 == 정답 집합
    - Per-type F1 (Macro): 각 타입을 독립 이진 분류로 보고 F1 산출
    """
    types = [t.value for t in TaskType]
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    exact_match = 0
    total = 0

    for task_id, pred_types in predictions.items():
        gold_types = gt.get(task_id)
        if gold_types is None:
            continue
        total += 1
        pred_set = set(pred_types)
        gold_set = set(gold_types)

        if pred_set == gold_set:
            exact_match += 1

        for t in types:
            in_pred = t in pred_set
            in_gold = t in gold_set
            if in_pred and in_gold:
                tp[t] += 1
            elif in_pred and not in_gold:
                fp[t] += 1
            elif not in_pred and in_gold:
                fn[t] += 1

    per_type = {}
    f1s = []
    for t in types:
        p = tp[t] / (tp[t] + fp[t]) if (tp[t] + fp[t]) > 0 else 0
        r = tp[t] / (tp[t] + fn[t]) if (tp[t] + fn[t]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        support = tp[t] + fn[t]
        per_type[t] = {"precision": round(p, 4), "recall": round(r, 4),
                       "f1": round(f1, 4), "support": support}
        if support > 0:
            f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0

    return {
        "exact_match": round(exact_match / total, 4) if total > 0 else 0,
        "macro_f1": round(macro_f1, 4),
        "exact_match_count": exact_match,
        "total": total,
        "per_type": per_type,
    }


def print_metrics(name: str, metrics: dict):
    print(f"\n=== {name} ===")
    print(f"  Exact Match : {metrics['exact_match_count']}/{metrics['total']} = {metrics['exact_match']:.1%}")
    print(f"  Macro-F1    : {metrics['macro_f1']:.4f}")
    print(f"  {'Type':<6} {'P':>7} {'R':>7} {'F1':>7} {'N':>6}")
    print(f"  {'-'*40}")
    for t, s in metrics["per_type"].items():
        if s["support"] == 0:
            continue
        print(f"  {t:<6} {s['precision']:>7.3f} {s['recall']:>7.3f} {s['f1']:>7.3f} {s['support']:>6}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", default="both", choices=["rule", "llm", "both"])
    parser.add_argument("--llm-model", default="qwen3:8b")
    args = parser.parse_args()

    labels = load_labels("annotation/thinkgeo_taxonomy_labels.json")

    # 189개 평가 태스크 ID (task_level 파일에서 추출)
    task_level_path = ROOT / "annotation/task_level_vanilla_react.json"
    with open(task_level_path, encoding="utf-8") as f:
        task_level = json.load(f)
    eval_ids = {r["task_id"] for r in task_level}
    eval_labels = {tid: v for tid, v in labels.items() if tid in eval_ids}
    print(f"평가 태스크: {len(eval_labels)}개")

    # 멀티레이블 GT: types 사용
    gt: dict[str, list[str]] = {
        tid: v["annotation"]["types"]
        for tid, v in eval_labels.items()
    }

    results = {}

    if args.router in ("rule", "both"):
        print("\n[규칙 기반 Router 실행 중...]")
        router = RuleRouter()
        preds = {tid: [t.value for t in router.route(v["query"])] for tid, v in eval_labels.items()}
        metrics = compute_metrics(preds, gt)
        print_metrics("Rule-based Router", metrics)
        results["rule"] = metrics

    rule_preds: dict[str, list[str]] = {}
    if args.router in ("rule", "both"):
        router = RuleRouter()
        rule_preds = {tid: [t.value for t in router.route(v["query"])] for tid, v in eval_labels.items()}

    llm_preds: dict[str, list[str]] = {}
    if args.router in ("llm", "both"):
        print(f"\n[LLM Router 실행 중... 모델: {args.llm_model}]")
        from models.ollama_llm import OllamaLLM
        llm = OllamaLLM(model_name=args.llm_model)
        router = LLMRouter(llm, think=False)

        for i, (tid, v) in enumerate(eval_labels.items()):
            preds_multi = router.route_multi(v["query"])
            llm_preds[tid] = [t.value for t in preds_multi]
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(eval_labels)} ...")

        metrics = compute_metrics(llm_preds, gt)
        print_metrics(f"LLM Router ({args.llm_model})", metrics)
        results["llm"] = metrics

    # 불일치 케이스 추출 (both 모드일 때)
    if args.router == "both" and rule_preds and llm_preds:
        disagreements = []
        for tid in eval_labels:
            r = rule_preds.get(tid)
            l = llm_preds.get(tid)
            gold = gt.get(tid, [])
            if r and l and set(r) != set(l):
                disagreements.append({
                    "task_id": tid,
                    "query": eval_labels[tid]["query"],
                    "rule_pred": r,
                    "llm_pred": l,
                    "current_label": gold,
                    "manual_label": None,   # 수동 검토 채울 필드
                    "notes": "",
                })

        dis_path = ROOT / "annotation/router_disagreements.json"
        with open(dis_path, "w", encoding="utf-8") as f:
            json.dump(disagreements, f, indent=2, ensure_ascii=False)
        print(f"\n불일치 케이스: {len(disagreements)}개 → {dis_path}")

    # 저장
    out_path = ROOT / "annotation/router_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
