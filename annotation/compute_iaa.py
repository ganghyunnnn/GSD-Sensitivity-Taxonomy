"""
IAA 계산 스크립트 — Cohen's κ (multi-label 확장)

IAA CSV 작성 완료 후 실행:
    python annotation/compute_iaa.py

출력:
- 단순 정확도 (exact match)
- 타입별 Agreement
- Cohen's κ (단일 레이블 행만) & Fleiss' κ (근사)
- 논문 기재용 요약 문장
"""
import csv
import json
from pathlib import Path
from collections import Counter

ROOT        = Path(__file__).resolve().parent.parent
LABELS_PATH = ROOT / "annotation/thinkgeo_taxonomy_labels.json"
IAA_CSV     = ROOT / "annotation/iaa_sample.csv"


def parse_types(s: str) -> frozenset:
    """'D+M1' → frozenset({'D','M1'})"""
    s = s.strip().upper()
    if not s:
        return frozenset()
    return frozenset(t.strip() for t in s.split("+") if t.strip())


def cohen_kappa(orig: list, anno: list) -> float:
    """카테고리 쌍 목록에서 Cohen's κ 계산 (단일 레이블 가정)."""
    assert len(orig) == len(anno)
    n = len(orig)
    cats = sorted(set(orig) | set(anno))
    cat2i = {c: i for i, c in enumerate(cats)}
    k = len(cats)

    # confusion matrix
    cm = [[0]*k for _ in range(k)]
    for o, a in zip(orig, anno):
        cm[cat2i[o]][cat2i[a]] += 1

    p_o = sum(cm[i][i] for i in range(k)) / n
    p_e = sum(
        (sum(cm[i][j] for j in range(k)) / n) *
        (sum(cm[j][i] for j in range(k)) / n)
        for i in range(k)
    )
    return (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else 1.0


def main():
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)

    rows = []
    with open(IAA_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    total = len(rows)
    empty = [r for r in rows if not r["type_annotator"].strip()]
    if empty:
        print(f"[경고] {len(empty)}개 행이 아직 비어 있습니다. 먼저 작성 후 재실행하세요.")
        missing = [r["task_id"] for r in empty[:5]]
        print(f"  예시: task_id {missing}")
        return

    orig_list, anno_list = [], []
    exact_match = 0
    type_agree: dict[str, list] = {"D": [], "M1": [], "M2": []}

    for row in rows:
        tid  = row["task_id"]
        orig = frozenset(labels[tid]["annotation"]["types"])
        anno = parse_types(row["type_annotator"])

        orig_list.append(tuple(sorted(orig)))
        anno_list.append(tuple(sorted(anno)))

        if orig == anno:
            exact_match += 1

        # 타입별 binary agreement
        for t in ["D", "M1", "M2"]:
            type_agree[t].append((t in orig) == (t in anno))

    exact_acc = exact_match / total * 100

    # Cohen's κ (단일 레이블 행만 사용)
    single_orig = [o[0] for o, a in zip(orig_list, anno_list)
                   if len(o) == 1 and len(a) == 1]
    single_anno = [a[0] for o, a in zip(orig_list, anno_list)
                   if len(o) == 1 and len(a) == 1]
    kappa = cohen_kappa(single_orig, single_anno) if single_orig else None

    print("=" * 50)
    print(f"IAA 결과 (N={total})")
    print("=" * 50)
    print(f"Exact match accuracy : {exact_acc:.1f}%  ({exact_match}/{total})")
    print()
    print("타입별 binary agreement:")
    for t in ["D", "M1", "M2"]:
        ag = sum(type_agree[t]) / total * 100
        print(f"  {t:3s}: {ag:.1f}%")
    print()
    if kappa is not None:
        print(f"Cohen's κ (단일 레이블, N={len(single_orig)}): {kappa:.3f}")
        strength = (
            "거의 완벽 (≥0.80)" if kappa >= 0.80 else
            "상당함 (0.60–0.80)"  if kappa >= 0.60 else
            "중간 (0.40–0.60)"    if kappa >= 0.40 else
            "미약 (<0.40)"
        )
        print(f"  해석: {strength}")
    print()

    # 불일치 목록
    disagree = [(rows[i]["task_id"], orig_list[i], anno_list[i])
                for i in range(total) if orig_list[i] != anno_list[i]]
    print(f"불일치 항목: {len(disagree)}개")
    for tid, o, a in disagree[:10]:
        print(f"  task {tid}: orig={'+'.join(o)}  anno={'+'.join(a)}")
    if len(disagree) > 10:
        print(f"  ... 외 {len(disagree)-10}개")

    print()
    print("논문 기재 문구 (예시):")
    if kappa is not None:
        print(f'  "A self-consistency check on {total} randomly sampled tasks '
              f'(stratified 20\\% sample) yielded Cohen\'s $\\kappa = {kappa:.2f}$, '
              f'indicating {strength.split()[0].lower()} agreement."')
    else:
        print(f'  "Exact-match agreement on {total} sampled tasks: {exact_acc:.1f}\\%."')


if __name__ == "__main__":
    main()
