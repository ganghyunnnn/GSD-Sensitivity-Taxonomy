"""
thinkgeo_taxonomy_labels.json 스키마 마이그레이션.
  Before: { "primary_type": "M1", "all_types": ["M1", "M4"], ... }
  After:  { "types": ["M1", "M4"], ... }

사용법:
    cd A:/rs-taxonomy
    python src/taxonomy/migrate_to_multilabel.py [--dry-run]
"""
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
LABELS_PATH = ROOT / "annotation/thinkgeo_taxonomy_labels.json"


def migrate(labels: dict) -> tuple[dict, int]:
    migrated = {}
    changed = 0
    for tid, entry in labels.items():
        ann = entry["annotation"]
        new_ann = {k: v for k, v in ann.items() if k not in ("primary_type", "all_types")}

        # all_types → types (없으면 primary_type으로 폴백)
        if "all_types" in ann:
            new_ann["types"] = ann["all_types"]
            changed += 1
        elif "primary_type" in ann:
            new_ann["types"] = [ann["primary_type"]]
            changed += 1
        else:
            new_ann["types"] = entry.get("types", ["D"])

        migrated[tid] = {**entry, "annotation": new_ann}
    return migrated, changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="변경 없이 결과만 출력")
    args = parser.parse_args()

    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)

    print(f"로드: {len(labels)}개 태스크")

    # 이미 마이그레이션됐는지 확인
    sample = next(iter(labels.values()))["annotation"]
    if "types" in sample and "all_types" not in sample and "primary_type" not in sample:
        print("이미 마이그레이션 완료된 파일입니다.")
        return

    migrated, changed = migrate(labels)

    # 검증: 샘플 출력
    sample_ids = list(migrated.keys())[:3]
    print("\n[샘플 확인]")
    for tid in sample_ids:
        ann = migrated[tid]["annotation"]
        print(f"  task_id={tid}: types={ann['types']}")

    if args.dry_run:
        print(f"\n[dry-run] {changed}개 항목 변환 예정. 파일 저장 안 함.")
        return

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(migrated, f, indent=2, ensure_ascii=False)

    print(f"\n마이그레이션 완료: {changed}개 항목 변환 → {LABELS_PATH}")


if __name__ == "__main__":
    main()
