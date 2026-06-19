"""
router_disagreements.json의 모든 케이스를 CSV로 내보내는 스크립트.
"""
import csv
import json
from pathlib import Path

DISAGREEMENTS_PATH = Path("annotation/router_disagreements.json")
CSV_PATH = Path("annotation/all_disagreements_review.csv")


def main():
    with open(DISAGREEMENTS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "no", "task_id", "query", "rule_pred", "llm_pred",
            "current_label", "manual_label", "notes"
        ])
        writer.writeheader()
        for i, item in enumerate(data, 1):
            writer.writerow({
                "no": i,
                "task_id": item["task_id"],
                "query": item["query"],
                "rule_pred": item["rule_pred"],
                "llm_pred": item["llm_pred"],
                "current_label": item["current_label"],
                "manual_label": item.get("manual_label", "") or "",
                "notes": item.get("notes", "") or "",
            })

    print(f"저장 완료: {CSV_PATH} ({len(data)}개 케이스)")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent.parent)
    main()
