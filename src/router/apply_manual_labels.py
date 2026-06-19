"""
CSV의 manual_label 컬럼을 읽어 router_disagreements.json 및
thinkgeo_taxonomy_labels.json에 반영하는 스크립트.
"""
import csv
import json
from pathlib import Path

DISAGREEMENTS_PATH = Path("annotation/router_disagreements.json")
LABELS_PATH = Path("annotation/thinkgeo_taxonomy_labels.json")
CSV_PATH = Path("annotation/manual_review.csv")

VALID_LABELS = {"D", "M1", "M2", "M3", "M4", "M5"}


def main():
    # CSV 읽기
    manual = {}  # task_id -> label
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["manual_label"].strip().upper()
            if label and label in VALID_LABELS:
                manual[row["task_id"]] = label
            elif label:
                print(f"[WARN] task_id={row['task_id']} 잘못된 레이블: '{label}' (무시됨)")

    if not manual:
        print("채워진 manual_label이 없습니다. CSV를 확인해주세요.")
        return

    print(f"적용할 레이블 {len(manual)}개: {manual}")

    # router_disagreements.json 업데이트
    with open(DISAGREEMENTS_PATH, encoding="utf-8") as f:
        disagreements = json.load(f)

    updated = 0
    for item in disagreements:
        tid = item["task_id"]
        if tid in manual:
            item["manual_label"] = manual[tid]
            updated += 1

    with open(DISAGREEMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(disagreements, f, ensure_ascii=False, indent=2)
    print(f"router_disagreements.json 업데이트: {updated}개")

    # thinkgeo_taxonomy_labels.json 업데이트
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)

    label_updated = 0
    for item in labels:
        tid = str(item["task_id"])
        if tid in manual:
            item["label"] = manual[tid]
            item["source"] = "manual"
            label_updated += 1

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"thinkgeo_taxonomy_labels.json 업데이트: {label_updated}개")

    # 아직 null인 케이스 확인
    remaining = [
        item["task_id"]
        for item in disagreements
        if item.get("manual_label") is None
    ]
    if remaining:
        print(f"\n아직 미결인 task_id ({len(remaining)}개): {remaining}")
    else:
        print("\n모든 케이스 레이블 완료!")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent.parent)
    main()
