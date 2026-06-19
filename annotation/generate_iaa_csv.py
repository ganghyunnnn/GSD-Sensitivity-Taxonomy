"""
IAA (Inter-Annotator Agreement) CSV 생성 스크립트

- 전체 436개 태스크에서 20% 층화 샘플(~87개)을 추출
- 기존 레이블은 포함하지 않음 (anchoring bias 방지)
- 재주석 완료 후 compute_iaa.py로 Cohen's κ 계산
"""
import csv
import json
import random
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = ROOT / "annotation/thinkgeo_taxonomy_labels.json"
OUT_CSV     = ROOT / "annotation/iaa_sample.csv"
SEED        = 42
SAMPLE_RATE = 0.20   # 20%


def main():
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)

    # 타입 조합별로 그룹화
    groups: dict[str, list] = defaultdict(list)
    for tid, v in labels.items():
        key = "+".join(sorted(v["annotation"]["types"]))
        groups[key].append((tid, v))

    rng = random.Random(SEED)
    sampled = []
    for key, items in sorted(groups.items()):
        n = max(1, round(len(items) * SAMPLE_RATE))
        picked = rng.sample(items, min(n, len(items)))
        sampled.extend(picked)

    # task_id 순 정렬
    sampled.sort(key=lambda x: int(x[0]))

    fieldnames = [
        "task_id",
        "image",
        "query",
        "type_annotator",   # 재주석 시 채울 항목 (D / M1 / M2 / D+M1 / D+M2 / M1+M2 / D+M1+M2)
        "notes",            # 선택 기재
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tid, v in sampled:
            writer.writerow({
                "task_id":        tid,
                "image":          v.get("image", ""),
                "query":          v["query"],
                "type_annotator": "",
                "notes":          "",
            })

    # 타입별 분포 출력
    from collections import Counter
    dist = Counter()
    for tid, v in sampled:
        key = "+".join(sorted(v["annotation"]["types"]))
        dist[key] += 1

    print(f"IAA CSV 생성: {OUT_CSV}")
    print(f"총 샘플: {len(sampled)}개 / 전체 436개 ({len(sampled)/436*100:.1f}%)")
    print("\n타입별 분포:")
    for k, n in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {k:15s}: {n:3d}개")
    print()
    print("사용 방법:")
    print("  1. annotation/iaa_sample.csv 를 엑셀/Sheets에서 열기")
    print("  2. 각 행의 query를 읽고 type_annotator 컬럼에 타입 입력")
    print("     (D / M1 / M2 / D+M1 / D+M2 / M1+M2 / D+M1+M2)")
    print("  3. 완료 후 python annotation/compute_iaa.py 실행")


if __name__ == "__main__":
    main()
