"""
전체 436개 점검 후 발견된 추가 28개 수정 사항 적용.

[변경 근거 요약]

M1 → D (GSD 없음 / 측정 불필요):
  25  "Can you tell the colors of the cars parked between the non-flooded houses?" → 색상 서술
  37  "Is there any route between the flooded houses?" → 경로 서술
  92  "What is the color of the smallest car in the image?" → 색상 서술 (GSD 없음)
  95  "Based on the area each car occupies, do they appear to be the same model?" → 시각 판단 (GSD 없음)
  104 "Are the parked cars spaced evenly...?" → 정성적 간격 서술 (GSD 없음)
  108 "Draw bboxes around the two cars that are farthest apart from each other" → bbox 태스크 (GSD 없음)
  111 "Can you add the word 'left' in white color right after the word 'only'?" → 이미지 편집
  118 "Draw bboxes around the two cars that are closest to each other" → bbox 태스크 (GSD 없음)
  140 "Out of all planes head towards east belong to same categories?" → 카테고리 분류 (방향은 필터)
  194 "Are all the planes of same length?" → 시각 비교 (GSD 없음)
  196 "Detect the smallest plane on the runway and draw a box on it" → bbox 태스크
  200 "Detect the smallest baseball diamond, and draw box on it" → bbox 태스크
  211 "Do both planes appear to be the same size in the aerial image?" → 시각 비교
  217 "Draw a bounding box around the largest detected baseball diamond" → bbox 태스크
  299 "Are any vehicles travelling on the main highway?" → 탐지/서술

M1 → M5 (GSD 없이 이미지 크기로 영역 비교):
  47  "Considering the dimension of 4592x3072, which half has more swimming pools?" → 두 반쪽 비교
  124 "Considering the dimension 1280x685, which half has more cars?" → 두 반쪽 비교
  133 "Considering the dimension 1372x941, which quadrant contains the most cars?" → 4개 분면 비교

M4 → M5 (두 집단 비교):
  39  "which side, left or right, has more flooded houses?" → 좌/우 침수 건물 수 비교
  50  "Among the minor damaged buildings, which one is the largest?" → 피해 집단 내 크기 비교
  58  "Can you plot the ratio of destroyed buildings to major-damage buildings?" → 두 피해 집단 비율
  59  "Can you check if major damaged buildings are more in numbers than destroyed buildings?" → 두 집단 수 비교
  64  "which zone has more destroyed buildings?" → 4개 분면 피해 건물 수 비교

M2 → M4:
  62  "how many percentage of the buildings are destroyed out of total buildings?" → 피해 비율(%)

M2 → M1 (GSD + 면적 계산):
  400 "convert pixel count to area in square meters using GSD 0.139 m/pixel" → 면적+GSD
  401 "convert pixel count to area in square meters using GSD 0.139 m/pixel" → 면적+GSD

D → M5 (두 집단 비교):
  186 "calculate the ratio of large to small vehicles" → 두 차량 집단 비율 비교
  220 "Which category has the more count, soccerball-field or baseball diamond?" → 두 집단 수 비교

[유지 확인]
  105: GSD(0.9067) 명시됨 → M1 유지
"""
import json
from pathlib import Path

LABELS_PATH = Path("annotation/thinkgeo_taxonomy_labels.json")
DISAGREEMENTS_PATH = Path("annotation/router_disagreements.json")

CORRECTIONS = {
    # M1 → D
    "25":  "D",
    "37":  "D",
    "92":  "D",
    "95":  "D",
    "104": "D",
    "108": "D",
    "111": "D",
    "118": "D",
    "140": "D",
    "194": "D",
    "196": "D",
    "200": "D",
    "211": "D",
    "217": "D",
    "299": "D",
    # M1 → M5
    "47":  "M5",
    "124": "M5",
    "133": "M5",
    # M4 → M5
    "39":  "M5",
    "50":  "M5",
    "58":  "M5",
    "59":  "M5",
    "64":  "M5",
    # M2 → M4
    "62":  "M4",
    # M2 → M1
    "400": "M1",
    "401": "M1",
    # D → M5
    "186": "M5",
    "220": "M5",
}


def main():
    # thinkgeo_taxonomy_labels.json 업데이트
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)

    changed = []
    for tid, new_label in CORRECTIONS.items():
        if tid in labels:
            old = labels[tid]["annotation"].get("types", [])
            new_types = [new_label]
            if old != new_types:
                labels[tid]["annotation"]["types"] = new_types
                labels[tid]["annotation"]["reviewed"] = True
                labels[tid]["annotation"]["notes"] = "manual_reviewed"
                changed.append((int(tid), old, new_types))

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"thinkgeo_taxonomy_labels.json: {len(changed)}개 수정")
    for tid, old, new in sorted(changed):
        print(f"  [{tid}] {old} -> {new}")

    # router_disagreements.json - 해당 task_id가 있으면 manual_label도 업데이트
    with open(DISAGREEMENTS_PATH, encoding="utf-8") as f:
        disagreements = json.load(f)

    dis_changed = 0
    for item in disagreements:
        tid = item["task_id"]
        if tid in CORRECTIONS:
            new = CORRECTIONS[tid]
            if item.get("manual_label") != new:
                item["manual_label"] = new
                dis_changed += 1

    with open(DISAGREEMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(disagreements, f, ensure_ascii=False, indent=2)

    if dis_changed:
        print(f"router_disagreements.json: {dis_changed}개 추가 수정")

    # 최종 분포 확인 (멀티레이블: 타입별 등장 태스크 수)
    from collections import Counter
    occurrence: Counter = Counter()
    for v in labels.values():
        for t in v["annotation"].get("types", []):
            occurrence[t] += 1
    print("\n=== 최종 타입별 등장 횟수 ===")
    for t in ["D", "M1", "M2", "M3", "M4", "M5"]:
        print(f"  {t}: {occurrence[t]}개")
    print(f"  총계(태스크): {len(labels)}개")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent.parent)
    main()
