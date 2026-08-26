"""
Phase 1: ThinkGeo 전체 태스크 Taxonomy 레이블링

사용법:
    python src/taxonomy/label_thinkgeo.py \
        --bench data/thinkgeo/ThinkGeoBench.json \
        --out annotation/thinkgeo_taxonomy_labels.json \
        [--subset data/thinkgeo/subset_tasks.json]

출력:
    annotation/thinkgeo_taxonomy_labels.json
    annotation/thinkgeo_taxonomy_summary.json  (통계)
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from taxonomy.classifier import TaxonomyClassifier, ClassificationResult
from taxonomy.definitions import TaskType


def extract_tasks_from_bench(bench: dict) -> dict[str, dict]:
    """ThinkGeoBench.json에서 태스크 정보 추출."""
    tasks = {}
    for task_id, entry in bench.items():
        # 첫 번째 user 메시지가 query
        query = ""
        for dialog in entry.get("dialogs", []):
            if dialog.get("role") == "user":
                query = dialog.get("content", "")
                break

        # 사용된 도구 목록
        tools_used = set()
        for dialog in entry.get("dialogs", []):
            for tc in dialog.get("tool_calls", []):
                name = tc.get("function", {}).get("name", "")
                if name:
                    tools_used.add(name)
        # 제공된 도구 목록 (tools 필드)
        tools_available = [t["name"] for t in entry.get("tools", [])]

        tasks[task_id] = {
            "query": query,
            "tools_used": sorted(tools_used),
            "tools_available": tools_available,
            "image": entry.get("files", [{}])[0].get("path", "") if entry.get("files") else "",
        }
    return tasks


def extract_tasks_from_subset(subset: dict) -> dict[str, dict]:
    """subset_tasks.json에서 태스크 정보 추출."""
    tasks = {}
    for task_id, entry in subset.items():
        tools = entry.get("tools", [])
        tasks[task_id] = {
            "query": entry.get("query", ""),
            "tools_used": tools,
            "tools_available": tools,
            "image": entry.get("image_path", ""),
        }
    return tasks


def result_to_dict(r: ClassificationResult) -> dict:
    return {
        "types": [t.value for t in r.all_types],
        "confidence": r.confidence,
        "evidence": r.evidence,
        "notes": r.notes,
        "reviewed": False,  # 수동 검토 여부 플래그
    }


def compute_summary(labels: dict) -> dict:
    type_occurrence: Counter = Counter()   # 각 타입이 등장한 태스크 수
    confidence_counter: Counter = Counter()
    combo_counter: Counter = Counter()     # 타입 조합별 태스크 수
    pure_d_count = 0

    for entry in labels.values():
        types = entry["annotation"]["types"]
        conf = entry["annotation"]["confidence"]

        for t in types:
            type_occurrence[t] += 1
        confidence_counter[conf] += 1
        combo_counter[tuple(sorted(types))] += 1
        if types == ["D"]:
            pure_d_count += 1

    total = len(labels)
    measurement_count = total - pure_d_count

    return {
        "total_tasks": total,
        "type_occurrence": dict(type_occurrence),          # 타입별 등장 태스크 수
        "type_occurrence_ratio": {k: round(v / total, 4) for k, v in type_occurrence.items()},
        "description_count": pure_d_count,
        "measurement_count": measurement_count,
        "measurement_ratio": round(measurement_count / total, 4) if total else 0,
        "confidence_distribution": dict(confidence_counter),
        "top_type_combos": [
            {"types": list(k), "count": v}
            for k, v in sorted(combo_counter.items(), key=lambda x: -x[1])[:10]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="ThinkGeo Taxonomy Labeling")
    parser.add_argument("--bench", default="data/thinkgeo/ThinkGeoBench.json")
    parser.add_argument("--subset", default=None)
    parser.add_argument("--out", default="annotation/thinkgeo_taxonomy_labels.json")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent  # repo root

    bench_path = root / args.bench
    out_path = root / args.out
    summary_path = out_path.parent / "thinkgeo_taxonomy_summary.json"

    print(f"벤치마크 로드: {bench_path}")
    with open(bench_path, encoding="utf-8") as f:
        bench = json.load(f)

    tasks = extract_tasks_from_bench(bench)

    # 서브셋이 있으면 추가 정보로 보강
    if args.subset:
        subset_path = root / args.subset
        print(f"서브셋 로드: {subset_path}")
        with open(subset_path, encoding="utf-8") as f:
            subset = json.load(f)
        subset_tasks = extract_tasks_from_subset(subset)
        for tid, info in subset_tasks.items():
            if tid in tasks:
                # tools_used 정보 보강
                tasks[tid]["tools_used"] = list(set(
                    tasks[tid]["tools_used"] + info["tools_used"]
                ))

    print(f"분류 시작: {len(tasks)}개 태스크")
    clf = TaxonomyClassifier()

    labels = {}
    for task_id, task_info in tasks.items():
        result = clf.classify(task_info["query"], task_info["tools_used"])
        labels[task_id] = {
            "task_id": task_id,
            "query": task_info["query"],
            "image": task_info["image"],
            "tools_used": task_info["tools_used"],
            "tools_available": task_info["tools_available"],
            "annotation": result_to_dict(result),
        }

    summary = compute_summary(labels)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== 레이블링 완료 ===")
    print(f"출력: {out_path}")
    print(f"요약: {summary_path}")
    print(f"\n태스크 유형 분포:")
    for t in ["D", "M1", "M2", "M3", "M4", "M5"]:
        count = summary["type_distribution"].get(t, 0)
        ratio = summary["type_ratio"].get(t, 0)
        bar = "█" * int(ratio * 40)
        print(f"  {t:3s}: {count:4d} ({ratio:.1%}) {bar}")
    print(f"\n신뢰도:")
    for conf, count in summary["confidence_distribution"].items():
        print(f"  {conf}: {count}")


if __name__ == "__main__":
    main()
