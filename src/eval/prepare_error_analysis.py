"""
Experiment C: 실패 유형 분석 준비 도구
기존 backbone 결과에서 M1/M2 실패 사례를 추출하고
수동 분류를 위한 CSV를 생성한다.

사용법:
    python src/eval/prepare_error_analysis.py
    → annotation/error_analysis_template.csv 생성
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

BENCH_PATH  = ROOT / "data/thinkgeo/ThinkGeoBench.json"
LABELS_PATH = ROOT / "annotation/thinkgeo_taxonomy_labels.json"
OUT_CSV     = ROOT / "annotation/error_analysis_template.csv"

# 분석 대상 백본 (모든 모델의 실패를 합산)
BACKBONE_FILES = {
    "Q2.5-VL-7B":  ROOT / "annotation/backbone_qwen25vl_7b.json",
    "Q3-VL-8B":    ROOT / "annotation/backbone_qwen3vl_8b.json",
    "IVL3.5-8B":   ROOT / "annotation/backbone_internvl35_8b.json",
}

# ── 실패 유형 분류 카테고리 ────────────────────────────────────────────────────

M1_CATEGORIES = {
    "GSD_IGNORED":      "GSD 무시 — 시각/정성적 답변 또는 픽셀값 직접 반환",
    "UNIT_ERROR":       "단위 변환 시도했으나 오류 (m↔px 혼동, 배율 오류 등)",
    "CALC_ERROR":       "올바른 접근이나 산술 오류 (계산 실수)",
    "OBJECT_NOT_FOUND": "측정 대상 객체를 이미지에서 찾지 못함",
    "REFUSAL":          "이미지로 판단 불가능하다며 답변 거부",
    "WRONG_INTERP":     "질문 자체를 잘못 이해 (D-type처럼 답변)",
    "OTHER":            "기타 (구체적으로 notes 기재)",
}

M2_CATEGORIES = {
    "UNDER_COUNT":      "과소 계산 — GT보다 낮은 수치 반환",
    "OVER_COUNT":       "과대 계산 — GT보다 높은 수치 반환",
    "RESOLUTION_FAIL":  "해상도 부족으로 객체 미인식 (흐림, 작은 객체 등)",
    "WRONG_OBJECTS":    "잘못된 객체 카테고리 카운트",
    "PARTIAL_COUNT":    "일부 객체만 계수 후 중단 또는 추정",
    "REFUSAL":          "이미지로 판단 불가능하다며 답변 거부",
    "OTHER":            "기타 (구체적으로 notes 기재)",
}


def load_failures():
    """전 모델 M1/M2 실패 사례 합산 (태스크 ID 기준 중복 제거)."""
    with open(BENCH_PATH, encoding="utf-8") as f:
        bench = json.load(f)
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)

    # task_id → 실패 정보 dict (여러 모델의 응답 포함)
    failures: dict[str, dict] = {}

    for model_name, path in BACKBONE_FILES.items():
        if not path.exists():
            print(f"[SKIP] {model_name}: {path} 없음")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for r in data["direct"]["results"]:
            if r["correct"]:
                continue
            tid = r["task_id"]
            gt_type = r["gt_type"]
            if gt_type not in ("M1", "M2"):
                continue

            if tid not in failures:
                task = bench.get(tid, {})
                query = ""
                if task.get("dialogs"):
                    query = task["dialogs"][0].get("content", "")
                gt_answer = task.get("gt_answer", {})
                gt_str = ""
                if isinstance(gt_answer, dict):
                    wl = gt_answer.get("whitelist", [])
                    gt_str = str(wl[0][0]) if wl and wl[0] else ""
                elif isinstance(gt_answer, list):
                    gt_str = str(gt_answer[0]) if gt_answer else ""

                failures[tid] = {
                    "task_id":  tid,
                    "gt_type":  gt_type,
                    "query":    query,
                    "gt_answer": gt_str,
                    "models":   {},
                }

            failures[tid]["models"][model_name] = r["answer"][:200]

    return list(failures.values())


def write_csv(failures: list[dict]):
    m1 = sorted([f for f in failures if f["gt_type"] == "M1"],
                key=lambda x: x["task_id"])
    m2 = sorted([f for f in failures if f["gt_type"] == "M2"],
                key=lambda x: x["task_id"])

    fieldnames = [
        "task_id", "gt_type", "query", "gt_answer",
        "Q2.5-VL-7B_answer", "Q3-VL-8B_answer", "IVL3.5-8B_answer",
        "category",   # 분류자가 채울 항목
        "notes",      # 분류자가 채울 항목
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in m1 + m2:
            row = {
                "task_id":   item["task_id"],
                "gt_type":   item["gt_type"],
                "query":     item["query"],
                "gt_answer": item["gt_answer"],
                "Q2.5-VL-7B_answer":  item["models"].get("Q2.5-VL-7B", ""),
                "Q3-VL-8B_answer":    item["models"].get("Q3-VL-8B", ""),
                "IVL3.5-8B_answer":   item["models"].get("IVL3.5-8B", ""),
                "category":  "",
                "notes":     "",
            }
            writer.writerow(row)

    print(f"CSV 생성: {OUT_CSV}")
    print(f"  M1 실패: {len(m1)}개")
    print(f"  M2 실패: {len(m2)}개")
    print(f"  합계:    {len(m1)+len(m2)}개\n")

    print("M1 분류 카테고리:")
    for k, v in M1_CATEGORIES.items():
        print(f"  {k:20s}: {v}")
    print("\nM2 분류 카테고리:")
    for k, v in M2_CATEGORIES.items():
        print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    failures = load_failures()
    print(f"총 실패 사례: {len(failures)}개 (M1+M2)")
    write_csv(failures)
