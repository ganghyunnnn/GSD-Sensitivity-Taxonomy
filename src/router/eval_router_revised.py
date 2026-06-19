"""
수정된 Ground Truth 기준으로 Router 정확도 재평가.
- router_disagreements.json의 manual_label을 Ground Truth로 사용
- 비불일치 케이스는 thinkgeo_taxonomy_labels.json의 annotation.primary_type 사용
- disagreements.json 덮어쓰기 없음
"""
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from taxonomy.definitions import TaskType
from router.rule_router import RuleRouter
from router.llm_router import LLMRouter
from router.hybrid_router import HybridRouter

ROOT = Path(__file__).parent.parent.parent


def load_ground_truth() -> dict[str, list[str]]:
    """
    189개 평가 태스크의 최종 Ground Truth 레이블을 반환 (멀티레이블).
    manual_label이 있으면 우선 사용 (list로 변환), 없으면 annotation.all_types 사용.
    """
    with open(ROOT / "annotation/thinkgeo_taxonomy_labels.json", encoding="utf-8") as f:
        labels = json.load(f)
    with open(ROOT / "annotation/router_disagreements.json", encoding="utf-8") as f:
        disagreements = json.load(f)
    with open(ROOT / "annotation/task_level_vanilla_react.json", encoding="utf-8") as f:
        task_level = json.load(f)

    eval_ids = {r["task_id"] for r in task_level}

    # manual_label 우선 적용 (str이면 list로 변환, 이미 list면 그대로)
    manual_override = {}
    for d in disagreements:
        if d.get("manual_label"):
            ml = d["manual_label"]
            manual_override[d["task_id"]] = ml if isinstance(ml, list) else [ml]

    gt: dict[str, list[str]] = {}
    for tid in eval_ids:
        if tid in manual_override:
            gt[tid] = manual_override[tid]
        elif tid in labels:
            gt[tid] = labels[tid]["annotation"]["types"]
    return gt


def compute_metrics(preds: dict[str, list[str]], gt: dict[str, list[str]]) -> dict:
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

    for tid, pred_types in preds.items():
        gold_types = gt.get(tid)
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
        per_type[t] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        if support > 0:
            f1s.append(f1)

    return {
        "exact_match": round(exact_match / total, 4) if total > 0 else 0,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0,
        "exact_match_count": exact_match,
        "total": total,
        "per_type": per_type,
    }


def print_metrics(name: str, m: dict):
    print(f"\n=== {name} ===")
    print(f"  Exact Match : {m['exact_match_count']}/{m['total']} = {m['exact_match']:.1%}")
    print(f"  Macro-F1    : {m['macro_f1']:.4f}")
    print(f"  {'Type':<6} {'P':>7} {'R':>7} {'F1':>7} {'N':>6}")
    print(f"  {'-'*40}")
    for t, s in m["per_type"].items():
        if s["support"] == 0:
            continue
        print(f"  {t:<6} {s['precision']:>7.3f} {s['recall']:>7.3f} {s['f1']:>7.3f} {s['support']:>6}")


def main():
    gt = load_ground_truth()
    # 멀티레이블 분포: 각 타입이 몇 개 태스크에 등장하는지
    type_counts: Counter = Counter()
    for types in gt.values():
        type_counts.update(types)
    print(f"평가 태스크: {len(gt)}개")
    print("GT 타입별 등장 횟수:", {t: type_counts[t] for t in ["D","M1","M2","M3","M4","M5"]})

    results = {}

    # --- Rule-based Router ---
    print("\n[Rule-based Router 실행 중...]")
    rule_router = RuleRouter()
    with open(ROOT / "annotation/thinkgeo_taxonomy_labels.json", encoding="utf-8") as f:
        labels_full = json.load(f)
    rule_preds: dict[str, list[str]] = {
        tid: [t.value for t in rule_router.route(labels_full[tid]["query"])]
        for tid in gt if tid in labels_full
    }
    metrics_rule = compute_metrics(rule_preds, gt)
    print_metrics("Rule-based Router", metrics_rule)
    results["rule"] = metrics_rule

    # --- LLM Router (gpt-oss:20b) ---
    print("\n[LLM Router 실행 중 (gpt-oss:20b)...]")
    try:
        from models.ollama_llm import OllamaLLM
        llm = OllamaLLM(model_name="gpt-oss:20b")
        llm_router = LLMRouter(llm, think=False)

        llm_preds: dict[str, list[str]] = {}
        eval_items = [(tid, labels_full[tid]["query"]) for tid in gt if tid in labels_full]
        for i, (tid, query) in enumerate(eval_items):
            pred = llm_router.route(query)
            llm_preds[tid] = [pred.value]
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(eval_items)} 완료...")

        metrics_llm = compute_metrics(llm_preds, gt)
        print_metrics("LLM Router (gpt-oss:20b)", metrics_llm)
        results["llm"] = metrics_llm

        # 불일치 분석 (GT 기준)
        both_wrong = sum(
            1 for tid in gt
            if tid in rule_preds and tid in llm_preds
            and set(rule_preds[tid]) != set(gt[tid]) and set(llm_preds[tid]) != set(gt[tid])
        )
        rule_only_correct = sum(
            1 for tid in gt
            if tid in rule_preds and tid in llm_preds
            and set(rule_preds[tid]) == set(gt[tid]) and set(llm_preds[tid]) != set(gt[tid])
        )
        llm_only_correct = sum(
            1 for tid in gt
            if tid in rule_preds and tid in llm_preds
            and set(rule_preds[tid]) != set(gt[tid]) and set(llm_preds[tid]) == set(gt[tid])
        )
        print(f"\n  [불일치 분석]")
        print(f"  Rule만 맞음: {rule_only_correct}개")
        print(f"  LLM만 맞음:  {llm_only_correct}개")
        print(f"  둘 다 틀림:  {both_wrong}개")

    except Exception as e:
        print(f"  LLM Router 오류: {e}")

    # --- Hybrid Router (Rule + M5 override pattern) ---
    print("\n[Hybrid Router 실행 중 (Rule + M5 override)...]")
    hybrid_router = HybridRouter(llm=None)  # 패턴 기반만
    hybrid_preds: dict[str, list[str]] = {
        tid: [t.value for t in hybrid_router.route(labels_full[tid]["query"])]
        for tid in gt if tid in labels_full
    }
    metrics_hybrid = compute_metrics(hybrid_preds, gt)
    print_metrics("Hybrid Router (Rule + M5 pattern)", metrics_hybrid)
    results["hybrid"] = metrics_hybrid

    rule_only = sum(1 for tid in gt if tid in rule_preds and tid in hybrid_preds
                    and set(rule_preds[tid]) == set(gt[tid]) and set(hybrid_preds[tid]) != set(gt[tid]))
    hybrid_only = sum(1 for tid in gt if tid in rule_preds and tid in hybrid_preds
                      and set(rule_preds[tid]) != set(gt[tid]) and set(hybrid_preds[tid]) == set(gt[tid]))
    both_wrong2 = sum(1 for tid in gt if tid in rule_preds and tid in hybrid_preds
                      and set(rule_preds[tid]) != set(gt[tid]) and set(hybrid_preds[tid]) != set(gt[tid]))
    print(f"\n  [Rule vs Hybrid 불일치]")
    print(f"  Rule만 맞음:   {rule_only}개")
    print(f"  Hybrid만 맞음: {hybrid_only}개")
    print(f"  둘 다 틀림:    {both_wrong2}개")

    # --- Hybrid + LLM (qwen3.5:9b as M5 arbiter) ---
    print("\n[Hybrid+LLM Router 실행 중 (Rule + M5 pattern + qwen3.5:9b arbiter)...]")
    try:
        from models.ollama_llm import OllamaLLM
        llm = OllamaLLM(model_name="qwen3.5:9b")
        hybrid_llm_router = HybridRouter(llm=llm)

        hybrid_llm_preds: dict[str, list[str]] = {}
        eval_items = [(tid, labels_full[tid]["query"]) for tid in gt if tid in labels_full]
        for i, (tid, query) in enumerate(eval_items):
            pred = hybrid_llm_router.route(query)
            hybrid_llm_preds[tid] = [pred.value]
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(eval_items)} 완료...")

        metrics_hybrid_llm = compute_metrics(hybrid_llm_preds, gt)
        print_metrics("Hybrid+LLM Router (Rule + M5 arbiter)", metrics_hybrid_llm)
        results["hybrid_llm"] = metrics_hybrid_llm

        hybrid_only2 = sum(1 for tid in gt if tid in rule_preds and tid in hybrid_llm_preds
                           and set(rule_preds[tid]) != set(gt[tid]) and set(hybrid_llm_preds[tid]) == set(gt[tid]))
        rule_only2 = sum(1 for tid in gt if tid in rule_preds and tid in hybrid_llm_preds
                         and set(rule_preds[tid]) == set(gt[tid]) and set(hybrid_llm_preds[tid]) != set(gt[tid]))
        print(f"\n  [Rule vs Hybrid+LLM 불일치]")
        print(f"  Rule만 맞음:       {rule_only2}개")
        print(f"  Hybrid+LLM만 맞음: {hybrid_only2}개")
    except Exception as e:
        print(f"  Hybrid+LLM Router 오류: {e}")

    out_path = ROOT / "annotation/router_eval_revised.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    import os
    os.chdir(ROOT)
    main()
