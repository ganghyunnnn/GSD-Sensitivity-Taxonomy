"""
전체 점검 후 확정된 21개 수정 사항 적용.
"""
import json
from pathlib import Path

# 최종 확정 수정 목록 (task_id str -> 정답 레이블)
CORRECTIONS = {
    # 기존 14개
    "22":  "M5",  # M1→M5: "which side has more water" = 좌/우 비교
    "56":  "D",   # M4→D:  "are all no-damage buildings in NE?" = 위치 서술
    "145": "D",   # M1→D:  "describe physical characteristics" = 색·구조 서술, GSD 불필요
    "150": "M5",  # D→M5:  "are they similarly affected?" = 두 건물 피해 비교
    "350": "M5",  # M1→M5: "average area of X compared to Y" = 두 집단 면적 비교
    "355": "M4",  # M1→M4: "% of buildings disappeared" = 소실 비율, GSD 불필요
    "366": "M5",  # M1→M5: "which group covers more area" = 두 집단 면적 비교
    "369": "M5",  # M2→M5: "ratio of destroyed to minor-damage" = 두 집단 수 비교
    "384": "M2",  # M4→M2: "total surveying time" = 파괴 건물 세기 후 산술
    "385": "M5",  # M1→M5: "which building has largest area" = 최대값 비교
    "387": "M5",  # M1→M5: "which quadrant has highest concentration" = 분면 비교
    "388": "M5",  # M4→M5: "ratio of minor-damage to major-damage" = 두 집단 비교
    "397": "M1",  # M4→M1: "mean distance from center to centroids" = 공간 거리 측정
    "398": "M4",  # M1→M4: "% with major structural failure" = 피해 비율, GSD 불필요
    # 추가 7개 (auto-fill M2 → M5: 비교형 질문)
    "335": "M5",  # "how many more destroyed on right vs left"
    "337": "M5",  # "which quadrant has most... by how many does it exceed"
    "338": "M5",  # "ratio of completely destroyed to partially damaged"
    "346": "M5",  # "which quadrant contains highest number"
    "363": "M5",  # "how many more major-damage compared to no-damage"
    "392": "M5",  # "which quadrant contains highest number of majorly damaged"
    "396": "M5",  # "which quadrant had highest count of destroyed"
}

DISAGREEMENTS_PATH = Path("annotation/router_disagreements.json")
LABELS_PATH = Path("annotation/thinkgeo_taxonomy_labels.json")


def main():
    # --- 1. router_disagreements.json ---
    with open(DISAGREEMENTS_PATH, encoding="utf-8") as f:
        disagreements = json.load(f)

    dis_changed = []
    for item in disagreements:
        tid = item["task_id"]
        if tid in CORRECTIONS:
            new = CORRECTIONS[tid]
            old = item.get("manual_label") or item.get("current_label")
            if old != new:
                item["manual_label"] = new
                dis_changed.append((tid, old, new))

    with open(DISAGREEMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(disagreements, f, ensure_ascii=False, indent=2)

    # --- 2. thinkgeo_taxonomy_labels.json ---
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)

    lbl_changed = []
    for tid, entry in labels.items():
        if tid in CORRECTIONS:
            new = CORRECTIONS[tid]
            old = entry["annotation"].get("types", [])
            new_types = [new]
            if old != new_types:
                entry["annotation"]["types"] = new_types
                entry["annotation"]["reviewed"] = True
                entry["annotation"]["notes"] = "manual_reviewed"
                lbl_changed.append((tid, old, new_types))

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    # --- 결과 출력 ---
    print(f"=== router_disagreements.json: {len(dis_changed)}개 수정 ===")
    for tid, old, new in sorted(dis_changed, key=lambda x: int(x[0])):
        print(f"  task_id={tid}: {old} -> {new}")

    print(f"\n=== thinkgeo_taxonomy_labels.json: {len(lbl_changed)}개 수정 ===")
    for tid, old, new in sorted(lbl_changed, key=lambda x: int(x[0])):
        print(f"  task_id={tid}: {old} -> {new}")

    # 잔여 null 확인
    null_remaining = [
        item["task_id"] for item in disagreements
        if item.get("manual_label") is None
    ]
    if null_remaining:
        print(f"\n[WARNING] 미결 task_id: {null_remaining}")
    else:
        print("\n모든 케이스 레이블 완료.")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent.parent)
    main()
