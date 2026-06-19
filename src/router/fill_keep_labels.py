"""
점검 결과 current_label이 맞다고 판단한 케이스들의 manual_label을 채운다.
"""
import json
from pathlib import Path

DISAGREEMENTS_PATH = Path("annotation/router_disagreements.json")

# 유지 결정 케이스: manual_label = current_label
KEEP = {
    "46":  "D",    # "is non-flooded house in SW corner?" = 위치 서술
    "293": "M1",   # "estimate total width of track" = 물리적 폭 측정
    "294": "D",    # "rate condition of swimming pools" = 상태 서술
    "319": "D",    # "estimate whether it affects tree growth" = 정성적 서술
    "321": "D",    # "estimate whether it severely affects tree growth" = 정성적 서술
    "343": "M2",   # "how many destroyed in upper vs lower half" = 세기 우선
    "365": "M4",   # "percentage of buildings completely destroyed" = 단일 피해 비율
    "379": "M4",   # "ratio of major damaged to total buildings" = 단일 피해 비율
    "382": "M2",   # "how many buildings destroyed compared to pre" = 세기 우선
    "386": "M2",   # "how many buildings changed from no-damage to major damage" = 세기
    "390": "M4",   # "proportion of buildings classified as major damaged" = 단일 비율
}


def main():
    with open(DISAGREEMENTS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    filled = 0
    for item in data:
        tid = item["task_id"]
        if tid in KEEP and item.get("manual_label") is None:
            item["manual_label"] = KEEP[tid]
            filled += 1

    with open(DISAGREEMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    null_remaining = [d["task_id"] for d in data if d.get("manual_label") is None]
    print(f"manual_label 채움: {filled}개")
    if null_remaining:
        print(f"[WARNING] 여전히 미결: {null_remaining}")
    else:
        print("모든 107개 케이스 레이블 완료.")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent.parent)
    main()
