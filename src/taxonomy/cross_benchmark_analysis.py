"""
Cross-Benchmark GSD-Sensitivity Distribution Analysis
REDESIGN_PLAN.md Phase 1-C

Algorithm 1을 5개 RS VQA 벤치마크에 자동 적용하여
D/M 분포를 계산. 논문 신규 Table 생성용.

지원 벤치마크:
  - RSVQA-LR    (data/rsvqa_lr/ or data/RSVQA-LR/)
  - RSVQA-HR    (data/RSVQA-HR/)    [선택]
  - EarthVQA    (data/earthvqa/)    [선택]
  - FloodNet    (data/floodnet_vqa/) [선택]
  - RSIVQA      (data/rsivqa/)      [선택]
  - ThinkGeo    (annotation/)       [기존 레이블 활용]
"""

import json
import sys
import csv
from pathlib import Path
from collections import Counter

# 경로 설정
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from taxonomy.gsd_classifier import GSDSensitivityClassifier, TaskType


# ── 벤치마크별 로더 ──────────────────────────────────────────────────────────

def load_rsvqa_lr(data_dir: Path) -> list[dict]:
    """RSVQA-LR: all_questions.json 또는 split별 JSON."""
    fpath = data_dir / "all_questions.json"
    if not fpath.exists():
        fpath = data_dir / "LR_split_test_questions.json"
    with open(fpath, encoding="utf-8") as f:
        d = json.load(f)
    questions = d.get("questions", d) if isinstance(d, dict) else d
    return [{"question": q["question"], "original_type": q.get("type", "")}
            for q in questions]


def load_rsvqa_hr(data_dir: Path) -> list[dict]:
    """RSVQA-HR: 동일 포맷."""
    fpath = data_dir / "all_questions.json"
    if not fpath.exists():
        fpath = data_dir / "USGS_split_test_questions.json"
    if not fpath.exists():
        return []
    with open(fpath, encoding="utf-8") as f:
        d = json.load(f)
    questions = d.get("questions", d) if isinstance(d, dict) else d
    return [{"question": q["question"], "original_type": q.get("type", "")}
            for q in questions]


def load_earthvqa(data_dir: Path) -> list[dict]:
    """EarthVLSet / EarthVQA: {image: [{Type, Question, Answer}, ...]} 포맷.
    실제 경로: data/EarthVL/EarthVQA/{Train,Val,Test}_QA.json
    """
    items = []
    # EarthVLSet 실제 포맷: {img_name: [{Type, Question, Answer}, ...]}
    for split_file in ["Train_QA.json", "Val_QA.json", "Test_QA.json"]:
        fpath = data_dir / split_file
        if fpath.exists():
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict):
                for qa_list in d.values():
                    if isinstance(qa_list, list):
                        for qa in qa_list:
                            q = qa.get("Question", qa.get("question", ""))
                            t = qa.get("Type", qa.get("type", ""))
                            if q:
                                items.append({"question": q, "original_type": t})
            continue
        # 폴백: 구형 단일 파일 포맷
    if items:
        return items
    for fname in ["qa_pairs.json", "questions.json", "EarthVQA_QA.json"]:
        fpath = data_dir / fname
        if fpath.exists():
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, list):
                raw = d
            elif "qa_pairs" in d:
                raw = d["qa_pairs"]
            elif "questions" in d:
                raw = d["questions"]
            else:
                raw = list(d.values()) if isinstance(d, dict) else []
            return [{"question": item.get("question", item.get("q", "")),
                     "original_type": item.get("type", item.get("question_type", ""))}
                    for item in raw if item.get("question") or item.get("q")]
    return []


def load_floodnet(data_dir: Path) -> list[dict]:
    """FloodNet VQA: Track2/Questions/ 아래 split JSON들을 합산."""
    # 실제 구조: Track2/Questions/{Test,Training,Valid} Question.json
    split_files = [
        data_dir / "Track2" / "Questions" / "Test_Question.json",
        data_dir / "Track2" / "Questions" / "Training Question.json",
        data_dir / "Track2" / "Questions" / "Valid Question.json",
    ]
    items = []
    for fpath in split_files:
        if fpath.exists():
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)
            # 포맷: {"0": {"Image_ID":..., "Question":..., "Question_Type":...}, ...}
            if isinstance(d, dict):
                for v in d.values():
                    q = v.get("Question", v.get("question", ""))
                    t = v.get("Question_Type", v.get("type", ""))
                    if q:
                        items.append({"question": q, "original_type": t})
    if items:
        return items
    # 폴백: 구형 단일 파일 포맷
    for fname in ["questions.json", "floodnet_vqa.json", "FloodNet_QA.json"]:
        fpath = data_dir / fname
        if fpath.exists():
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)
            raw = list(d.values()) if isinstance(d, dict) else d
            return [{"question": item.get("Question", item.get("question", "")),
                     "original_type": item.get("Question_Type", item.get("type", ""))}
                    for item in raw if item.get("Question") or item.get("question")]
    return []


def load_rsivqa(data_dir: Path) -> list[dict]:
    """RSIVQA."""
    for fname in ["questions.json", "rsivqa_questions.json"]:
        fpath = data_dir / fname
        if fpath.exists():
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)
            questions = d.get("questions", d) if isinstance(d, dict) else d
            return [{"question": q.get("question", ""), "original_type": q.get("type", "")}
                    for q in questions]
    return []


def load_thinkgeo(annotation_dir: Path) -> list[dict]:
    """ThinkGeo: 기존 taxonomy labels 활용 (실제 어노테이션 ground truth)."""
    fpath = annotation_dir / "thinkgeo_taxonomy_labels.json"
    with open(fpath, encoding="utf-8") as f:
        d = json.load(f)
    items = list(d.values()) if isinstance(d, dict) else d
    return [{"question": item.get("query", ""),
             "original_type": item["annotation"]["primary_type"],
             "gold_label": item["annotation"]["primary_type"]}
            for item in items]


# ── 분석 함수 ─────────────────────────────────────────────────────────────────

def analyze_benchmark(name: str, items: list[dict],
                      clf: GSDSensitivityClassifier) -> dict:
    """하나의 벤치마크에 Algorithm 1 적용 → 분포 통계."""
    questions = [item["question"] for item in items]
    results = clf.classify_batch(questions)

    total = len(results)
    if total == 0:
        return {}

    type_counts = Counter(r.task_type for r in results)
    deterministic = sum(1 for r in results if r.confidence == "deterministic")
    m_total = total - type_counts[TaskType.D]

    row = {
        "benchmark":   name,
        "N":           total,
        "D%":          type_counts[TaskType.D]  / total * 100,
        "M1%":         type_counts[TaskType.M1] / total * 100,
        "M2%":         type_counts[TaskType.M2] / total * 100,
        "M3%":         type_counts[TaskType.M3] / total * 100,
        "M4%":         type_counts[TaskType.M4] / total * 100,
        "M5%":         type_counts[TaskType.M5] / total * 100,
        "M_total%":    m_total / total * 100,
        "coverage%":   deterministic / total * 100,
    }

    # ThinkGeo는 gold label이 있으므로 Algorithm 1 정확도 계산 가능
    if "gold_label" in items[0]:
        correct = sum(1 for item, r in zip(items, results)
                      if item.get("gold_label") == r.task_type.value)
        row["alg1_accuracy%"] = correct / total * 100
        row["alg1_n"] = total

    return row


def print_table(rows: list[dict]):
    """콘솔 출력용 ASCII 테이블."""
    print("\n" + "="*90)
    print(f"{'Benchmark':<15} {'N':>8} {'D%':>6} {'M1%':>6} {'M2%':>6} "
          f"{'M3%':>6} {'M4%':>6} {'M5%':>6} {'M%':>7} {'Cover%':>8}")
    print("-"*90)
    for r in rows:
        print(f"{r['benchmark']:<15} {r['N']:>8,} {r['D%']:>6.1f} {r['M1%']:>6.1f} "
              f"{r['M2%']:>6.1f} {r['M3%']:>6.1f} {r['M4%']:>6.1f} "
              f"{r['M5%']:>6.1f} {r['M_total%']:>7.1f} {r['coverage%']:>8.1f}")
        if "alg1_accuracy%" in r:
            print(f"  -> Algorithm 1 accuracy on gold labels: {r['alg1_accuracy%']:.1f}%")
    print("="*90)


def save_latex_table(rows: list[dict], out_path: Path):
    """논문 직접 사용 가능한 LaTeX 테이블."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{GSD-Sensitivity Distribution Across RS VQA Benchmarks}",
        r"\label{tab:cross_benchmark}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\hline",
        r"Benchmark & $N$ & D\% & M1\% & M2\% & M3\% & M4\% & M5\% & M\% \\",
        r"\hline",
    ]
    for r in rows:
        line = (f"{r['benchmark']} & {r['N']:,} & "
                f"{r['D%']:.1f} & {r['M1%']:.1f} & {r['M2%']:.1f} & "
                f"{r['M3%']:.1f} & {r['M4%']:.1f} & {r['M5%']:.1f} & "
                f"{r['M_total%']:.1f} \\\\")
        lines.append(line)
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"LaTeX saved: {out_path}")


def save_csv(rows: list[dict], out_path: Path):
    # 모든 row의 키를 합집합으로 fieldnames 결정
    all_keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore",
                                restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {out_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    clf = GSDSensitivityClassifier()
    data_root = ROOT / "data"
    annotation_dir = ROOT / "annotation"

    # 벤치마크 로더 테이블 (이름: (로더함수, 데이터경로))
    loaders = {
        "RSVQA-LR":  (load_rsvqa_lr,  data_root / "RSVQA-LR"),
        "RSVQA-HR":  (load_rsvqa_hr,  data_root / "RSVQA-HR"),
        "EarthVQA":  (load_earthvqa,  data_root / "EarthVL" / "EarthVQA"),
        "FloodNet":  (load_floodnet,  data_root / "FloodNet"),
        "RSIVQA":    (load_rsivqa,    data_root / "rsivqa"),
        "ThinkGeo":  (load_thinkgeo,  annotation_dir),
    }

    rows = []
    for name, (loader_fn, dpath) in loaders.items():
        if name == "ThinkGeo":
            items = loader_fn(dpath)
        elif not dpath.exists():
            print(f"[SKIP] {name}: no data ({dpath})")
            continue
        else:
            items = loader_fn(dpath)

        if not items:
            print(f"[SKIP] {name}: load failed")
            continue

        row = analyze_benchmark(name, items, clf)
        if row:
            rows.append(row)
            acc_str = (f", alg1_acc={row['alg1_accuracy%']:.1f}%"
                       if "alg1_accuracy%" in row else "")
            print(f"[OK] {name}: N={row['N']:,}, M={row['M_total%']:.1f}%, "
                  f"cover={row['coverage%']:.1f}%{acc_str}")

    if not rows:
        print("분석할 데이터가 없습니다.")
        return

    print_table(rows)

    # 결과 저장
    exp_dir = ROOT / "experiments"
    save_csv(rows, exp_dir / "cross_benchmark_distribution.csv")
    save_latex_table(rows, exp_dir / "cross_benchmark_table.tex")

    # JSON 저장 (상세)
    out_json = exp_dir / "cross_benchmark_distribution.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"JSON 저장: {out_json}")

    # RSVQA-LR original type vs Algorithm 1 비교 (검증)
    _validate_rsvqa_mapping(data_root / "RSVQA-LR", clf)

    # FloodNet original type vs Algorithm 1 비교 (검증)
    _validate_floodnet_mapping(data_root / "FloodNet", clf)


def _validate_rsvqa_mapping(data_dir: Path, clf: GSDSensitivityClassifier):
    """
    RSVQA-LR의 원본 type(count/comp/presence/rural_urban)과
    Algorithm 1 분류 결과를 비교하여 매핑 일치율 계산.
    """
    if not data_dir.exists():
        return

    items = load_rsvqa_lr(data_dir)

    # RSVQA 원본 type → D/M 매핑 (expected)
    rsvqa_to_dm = {
        "presence":   TaskType.D,
        "rural_urban": TaskType.D,
        "count":      TaskType.M2,
        "comp":       TaskType.M5,
        # HR 전용
        "area":       TaskType.M1,
        "yes_no":     TaskType.D,
    }

    total = correct = 0
    mismatch_examples = []

    for item in items:
        orig = item.get("original_type", "")
        expected = rsvqa_to_dm.get(orig)
        if expected is None:
            continue
        result = clf.classify(item["question"])
        total += 1
        if result.task_type == expected:
            correct += 1
        elif len(mismatch_examples) < 5:
            mismatch_examples.append({
                "question":  item["question"],
                "rsvqa_type": orig,
                "expected_dm": expected.value,
                "algorithm1":  result.task_type.value,
                "rule":        result.rule_id,
            })

    if total > 0:
        acc = correct / total * 100
        print(f"\n[RSVQA-LR 매핑 검증]")
        print(f"  원본 type → D/M 매핑 일치율: {acc:.1f}% ({correct}/{total})")
        if mismatch_examples:
            print("  불일치 예시:")
            for ex in mismatch_examples:
                print(f"    Q: {ex['question'][:60]}")
                print(f"       RSVQA={ex['rsvqa_type']} → expected={ex['expected_dm']}, "
                      f"alg1={ex['algorithm1']} (rule={ex['rule']})")

        # 저장
        result_path = ROOT / "experiments" / "rsvqa_mapping_validation.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({
                "total": total,
                "correct": correct,
                "accuracy": acc,
                "mismatch_examples": mismatch_examples,
            }, f, indent=2, ensure_ascii=False)
        print(f"  saved: {result_path}")


def _validate_floodnet_mapping(data_dir: Path, clf: GSDSensitivityClassifier):
    """
    FloodNet의 원본 Question_Type과 Algorithm 1 분류 결과 비교.
    FloodNet 타입 → D/M 매핑:
      Condition_Recognition → M4 (flood/damage threshold judgment)
      Yes_No                → D  (descriptive yes/no)
      Simple_Counting       → M2 (cardinality)
      Complex_Counting      → M2 or M5 (comparison counting)
    """
    if not data_dir.exists():
        return

    items = load_floodnet(data_dir)
    if not items:
        return

    floodnet_to_dm = {
        "Condition_Recognition": TaskType.M4,
        # FloodNet Yes_No = "Is the road flooded?" 형태 — 전부 flood threshold → M4
        "Yes_No":                TaskType.M4,
        "Simple_Counting":       TaskType.M2,
        "Complex_Counting":      TaskType.M2,  # 비교형이면 M5도 가능
    }

    total = correct = 0
    mismatch_examples = []
    type_stats: dict = {}

    for item in items:
        orig = item.get("original_type", "")
        expected = floodnet_to_dm.get(orig)
        if expected is None:
            continue
        result = clf.classify(item["question"])
        total += 1
        hit = (result.task_type == expected or
               (orig == "Complex_Counting" and result.task_type == TaskType.M5))
        if hit:
            correct += 1
        elif len(mismatch_examples) < 5:
            mismatch_examples.append({
                "question":    item["question"],
                "floodnet_type": orig,
                "expected_dm": expected.value,
                "algorithm1":  result.task_type.value,
                "rule":        result.rule_id,
            })
        # 타입별 통계
        if orig not in type_stats:
            type_stats[orig] = {"total": 0, "correct": 0}
        type_stats[orig]["total"] += 1
        if hit:
            type_stats[orig]["correct"] += 1

    if total > 0:
        acc = correct / total * 100
        print(f"\n[FloodNet 매핑 검증]")
        print(f"  원본 type → D/M 매핑 일치율: {acc:.1f}% ({correct}/{total})")
        for t, s in type_stats.items():
            t_acc = s["correct"] / s["total"] * 100
            print(f"    {t}: {t_acc:.1f}% ({s['correct']}/{s['total']})")
        if mismatch_examples:
            print("  불일치 예시:")
            for ex in mismatch_examples:
                print(f"    Q: {ex['question'][:60]}")
                print(f"       FloodNet={ex['floodnet_type']} → expected={ex['expected_dm']}, "
                      f"alg1={ex['algorithm1']} (rule={ex['rule']})")

        result_path = ROOT / "experiments" / "floodnet_mapping_validation.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({
                "total": total, "correct": correct, "accuracy": acc,
                "type_stats": type_stats,
                "mismatch_examples": mismatch_examples,
            }, f, indent=2, ensure_ascii=False)
        print(f"  saved: {result_path}")


if __name__ == "__main__":
    main()
