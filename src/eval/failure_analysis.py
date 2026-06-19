"""
Phase 1 핵심 분석: Taxonomy 유형별 실패율

ThinkGeo 실험 결과(Vanilla ReAct, Direct Prompting)를
taxonomy 레이블과 교차 분석하여 측정형 태스크의 실패율이
서술형보다 높은지 검증한다.

사용법:
    cd A:/MARA
    python src/eval/failure_analysis.py
"""

import json
from pathlib import Path
from collections import defaultdict
import sys
sys.path.insert(0, str(Path(__file__).parent))
from evaluator import evaluate_answer


ROOT = Path(__file__).parent.parent.parent


def load_labels(path: str) -> dict[str, dict]:
    with open(ROOT / path, encoding="utf-8") as f:
        return json.load(f)


def load_bench(path: str) -> dict[str, dict]:
    with open(ROOT / path, encoding="utf-8") as f:
        return json.load(f)


def load_experiment(path: str) -> dict[str, dict]:
    with open(ROOT / path, encoding="utf-8") as f:
        data = json.load(f)
    results = {}
    for r in data.get("results", []):
        results[r["task_id"]] = r
    return results


def get_model_answer(result: dict) -> str:
    """실험 결과에서 모델 답변 텍스트 추출."""
    ans = result.get("answer", {})
    if isinstance(ans, dict):
        result_obj = ans.get("result", {})
        if isinstance(result_obj, dict):
            return result_obj.get("answer", "")
        return str(result_obj)
    return str(ans)


def evaluate_per_type(
    system_name: str,
    exp_results: dict[str, dict],
    labels: dict[str, dict],
    bench: dict[str, dict],
) -> dict:
    type_stats: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "correct": 0, "failure": 0, "task_ids": []}
    )
    task_level: list[dict] = []

    for task_id, result in exp_results.items():
        label = labels.get(task_id)
        bench_entry = bench.get(task_id)
        if label is None or bench_entry is None:
            continue

        types = label["annotation"]["types"]
        gt_answer = bench_entry.get("gt_answer") or {}
        model_answer = get_model_answer(result)
        if not gt_answer:
            continue

        eval_result = evaluate_answer(model_answer, gt_answer)
        correct = eval_result["correct"]

        # 멀티레이블: 태스크가 속한 모든 타입에 집계
        for t in types:
            type_stats[t]["total"] += 1
            type_stats[t]["task_ids"].append(task_id)
            if correct:
                type_stats[t]["correct"] += 1
            else:
                type_stats[t]["failure"] += 1

        task_level.append({
            "task_id": task_id,
            "types": types,
            "correct": correct,
            "whitelist_pass": eval_result["whitelist_pass"],
            "blacklist_pass": eval_result["blacklist_pass"],
        })

    # overall은 태스크 단위 (중복 집계 없이)
    total_all = len(task_level)
    correct_all = sum(1 for t in task_level if t["correct"])

    by_type = {}
    for t in ["D", "M1", "M2"]:
        s = type_stats.get(t, {"total": 0, "correct": 0, "failure": 0, "task_ids": []})
        acc = s["correct"] / s["total"] if s["total"] > 0 else None
        fail_rate = s["failure"] / s["total"] if s["total"] > 0 else None
        by_type[t] = {
            "total": s["total"],
            "correct": s["correct"],
            "failure": s["failure"],
            "accuracy": round(acc, 4) if acc is not None else None,
            "failure_rate": round(fail_rate, 4) if fail_rate is not None else None,
        }

    return {
        "system": system_name,
        "overall": {
            "total": total_all,
            "correct": correct_all,
            "accuracy": round(correct_all / total_all, 4) if total_all > 0 else None,
        },
        "by_type": by_type,
        "task_level": task_level,
    }


def print_table(analysis: dict):
    sys_name = analysis["system"]
    overall = analysis["overall"]
    print(f"\n{'='*65}")
    print(f"  System: {sys_name}")
    print(f"  Overall: {overall['correct']}/{overall['total']} = {overall['accuracy']:.1%}")
    print(f"{'='*65}")
    print(f"  {'Type':<6} {'N':>6} {'Correct':>8} {'Acc':>8} {'FailRate':>10}")
    print(f"  {'-'*50}")
    for t in ["D", "M1", "M2"]:
        stat = analysis["by_type"][t]
        if stat["total"] == 0:
            continue
        acc = f"{stat['accuracy']:.1%}" if stat["accuracy"] is not None else "N/A"
        fail = f"{stat['failure_rate']:.1%}" if stat["failure_rate"] is not None else "N/A"
        marker = " <--" if t != "D" and stat["failure_rate"] is not None and stat["failure_rate"] > 0.5 else ""
        print(f"  {t:<6} {stat['total']:>6} {stat['correct']:>8} {acc:>8} {fail:>10}{marker}")


def main():
    print("Loading labels and benchmark...")
    labels = load_labels("annotation/thinkgeo_taxonomy_labels.json")
    bench = load_bench("data/thinkgeo/ThinkGeoBench.json")
    print(f"  Labels: {len(labels)} tasks")
    print(f"  Bench: {len(bench)} tasks")

    log_dir = ROOT / "experiments/logs"
    experiments = {
        "vanilla_react": sorted(log_dir.glob("thinkgeo_vanilla_react_*.json"))[-1],
        "direct_prompting": sorted(log_dir.glob("thinkgeo_direct_prompting_*.json"))[-1],
    }

    all_analyses = []
    for sys_name, exp_path in experiments.items():
        print(f"\nLoading: {exp_path.name}")
        exp_results = load_experiment(str(exp_path.relative_to(ROOT)))
        print(f"  -> {len(exp_results)} results")

        analysis = evaluate_per_type(sys_name, exp_results, labels, bench)
        # task_level은 저장용이므로 출력에서 제외
        print_table(analysis)
        all_analyses.append({k: v for k, v in analysis.items() if k != "task_level"})

        # task_level은 별도 저장
        task_out = ROOT / f"annotation/task_level_{sys_name}.json"
        with open(task_out, "w", encoding="utf-8") as f:
            json.dump(analysis["task_level"], f, indent=2, ensure_ascii=False)
        print(f"  Task-level saved: {task_out.name}")

    # 집계 저장
    out_path = ROOT / "annotation/failure_analysis_by_type.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_analyses, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    # 핵심 가설 검증
    print("\n\n=== Hypothesis: D-type vs M-type Failure Rate ===")
    for analysis in all_analyses:
        sys_name = analysis["system"]
        d_stat = analysis["by_type"]["D"]
        m_types = ["M1", "M2"]
        m_total = sum(analysis["by_type"][t]["total"] for t in m_types)
        m_correct = sum(analysis["by_type"][t]["correct"] for t in m_types)
        m_fail = 1 - (m_correct / m_total) if m_total > 0 else None

        d_fail = d_stat["failure_rate"]

        if d_fail is not None and m_fail is not None:
            diff = m_fail - d_fail
            supported = diff > 0.05
            print(f"\n  [{sys_name}]")
            print(f"    D-type fail: {d_fail:.1%} (n={d_stat['total']})")
            print(f"    M-type fail: {m_fail:.1%} (n={m_total})")
            print(f"    Diff (M-D): {diff:+.1%} -> {'SUPPORTED' if supported else 'NOT SUPPORTED'}")


if __name__ == "__main__":
    main()
