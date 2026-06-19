"""
Inter-Annotator Agreement (IAA) 측정.

50개 stratified sample에서 두 annotator의 레이블 일치도를
Cohen's Kappa로 계산한다.

사용법:
  python annotation/iaa_measurement.py --ann1 labels_annotator1.json --ann2 labels_annotator2.json
  python annotation/iaa_measurement.py --blind  # blind re-annotation 모드
"""

import json
import argparse
import sys
import io
from pathlib import Path
from collections import defaultdict

# Windows 콘솔에서 UTF-8 출력 강제 설정
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


CATEGORIES = ["D", "M1", "M2"]

# stratified 샘플 기본 할당 (합계 ~30, 3-type taxonomy 기준)
DEFAULT_N_PER_TYPE = {"D": 30, "M1": 29, "M2": 29}

# thinkgeo_taxonomy_labels.json 기준 경로
TAXONOMY_LABELS_PATH = Path(__file__).parent / "thinkgeo_taxonomy_labels.json"


# ---------------------------------------------------------------------------
# 1. 데이터 로드
# ---------------------------------------------------------------------------

def load_labels(path: str) -> dict:
    """JSON에서 task_id → list[type] 딕셔너리로 로드 (멀티레이블).

    세 가지 형식을 모두 지원한다.
    - 단순 str 형식: {"task_id": "M1", ...}  → ["M1"]로 변환
    - 단순 list 형식: {"task_id": ["M1","M4"], ...}
    - thinkgeo_taxonomy_labels.json 형식:
      {"task_id": {"annotation": {"types": ["M1","M4"]}, ...}, ...}

    Returns
    -------
    dict
        {task_id: list[str]} 형태의 딕셔너리
    """
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] 파일을 찾을 수 없습니다: {path}", file=sys.stderr)
        sys.exit(1)

    with open(p, encoding="utf-8") as f:
        raw = json.load(f)

    labels = {}
    for key, value in raw.items():
        if isinstance(value, str):
            labels[key] = [value]
        elif isinstance(value, list):
            labels[key] = value
        elif isinstance(value, dict):
            ann = value.get("annotation", {})
            if isinstance(ann, dict):
                # 신규 멀티레이블 형식
                if "types" in ann:
                    labels[key] = ann["types"]
                # 구형 단일레이블 폴백
                elif "primary_type" in ann:
                    labels[key] = [ann["primary_type"]]
            else:
                v = value.get("type") or value.get("label")
                if v is not None:
                    labels[key] = [v] if isinstance(v, str) else v
        # 인식 불가 형식은 건너뜀

    return labels


# ---------------------------------------------------------------------------
# 2. Cohen's Kappa (sklearn 없이 직접 구현)
# ---------------------------------------------------------------------------

def _binary_kappa(ann1: dict, ann2: dict, category: str) -> float:
    """특정 타입에 대한 이진 Cohen's Kappa 계산.

    각 태스크에 대해 annotator가 해당 타입을 포함했는지(1) 아닌지(0)로 비교.

    κ = (Po - Pe) / (1 - Pe)
    """
    common_ids = set(ann1.keys()) & set(ann2.keys())
    if not common_ids:
        return float("nan")

    # 2x2 confusion matrix: [[TN, FP], [FN, TP]]
    tp = fp = fn = tn = 0
    for tid in common_ids:
        a1 = category in ann1[tid]
        a2 = category in ann2[tid]
        if a1 and a2:
            tp += 1
        elif not a1 and a2:
            fp += 1
        elif a1 and not a2:
            fn += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    if total == 0:
        return float("nan")

    po = (tp + tn) / total
    p_pos = ((tp + fn) / total) * ((tp + fp) / total)
    p_neg = ((tn + fp) / total) * ((tn + fn) / total)
    pe = p_pos + p_neg

    if pe == 1.0:
        return float("nan")

    return (po - pe) / (1.0 - pe)


def cohen_kappa_multilabel(ann1: dict, ann2: dict, categories: list) -> dict:
    """멀티레이블 IAA: 타입별 이진 Kappa + 평균 Kappa 계산.

    Parameters
    ----------
    ann1, ann2 : dict
        {task_id: list[str]} 딕셔너리
    categories : list
        레이블 카테고리 목록

    Returns
    -------
    dict
        {"per_type": {type: kappa}, "mean_kappa": float}
    """
    per_type = {}
    kappas = []
    for cat in categories:
        k = _binary_kappa(ann1, ann2, cat)
        per_type[cat] = round(k, 4) if k == k else None  # nan → None
        if k == k:
            kappas.append(k)

    mean_kappa = round(sum(kappas) / len(kappas), 4) if kappas else float("nan")
    return {"per_type": per_type, "mean_kappa": mean_kappa}


# ---------------------------------------------------------------------------
# 3. 전체 일치 지표 계산
# ---------------------------------------------------------------------------

def compute_agreement(ann1: dict, ann2: dict) -> dict:
    """멀티레이블 IAA 계산.

    Parameters
    ----------
    ann1, ann2 : dict
        {task_id: list[str]} 딕셔너리

    Returns
    -------
    dict 키:
        - exact_match_rate : float  — set(ann1) == set(ann2) 비율
        - mean_kappa       : float  — 타입별 이진 Kappa 평균
        - per_type_kappa   : dict   — {type: kappa}
        - disagreements    : list[dict]
        - n_common         : int
    """
    common_ids = sorted(set(ann1.keys()) & set(ann2.keys()))
    n_common = len(common_ids)

    if n_common == 0:
        print("[WARNING] 공통 task_id가 없습니다.", file=sys.stderr)
        return {
            "exact_match_rate": float("nan"),
            "mean_kappa": float("nan"),
            "per_type_kappa": {},
            "disagreements": [],
            "n_common": 0,
        }

    exact = 0
    disagreements = []
    for tid in common_ids:
        l1, l2 = set(ann1[tid]), set(ann2[tid])
        if l1 == l2:
            exact += 1
        else:
            disagreements.append({
                "task_id": tid,
                "ann1": sorted(l1),
                "ann2": sorted(l2),
            })

    exact_match_rate = exact / n_common
    kappa_result = cohen_kappa_multilabel(ann1, ann2, CATEGORIES)

    return {
        "exact_match_rate": exact_match_rate,
        "mean_kappa": kappa_result["mean_kappa"],
        "per_type_kappa": kappa_result["per_type"],
        "disagreements": disagreements,
        "n_common": n_common,
    }


# ---------------------------------------------------------------------------
# 4. Stratified Sampling
# ---------------------------------------------------------------------------

def stratified_sample(labels: dict, n_per_type: dict = None) -> list:
    """타입별 stratified sampling.

    Parameters
    ----------
    labels : dict
        {task_id: label} 딕셔너리
    n_per_type : dict, optional
        {type: n} 딕셔너리. 기본값 DEFAULT_N_PER_TYPE 사용.

    Returns
    -------
    list
        샘플링된 task_id 리스트
    """
    if n_per_type is None:
        n_per_type = DEFAULT_N_PER_TYPE

    # 타입별 task_id 그룹화 (멀티레이블: 첫 번째 타입 기준으로 stratify)
    by_type = defaultdict(list)
    for tid, types in labels.items():
        primary = types[0] if types else "D"
        by_type[primary].append(tid)

    sampled = []
    for t, n in n_per_type.items():
        pool = by_type.get(t, [])
        if len(pool) < n:
            print(
                f"[WARNING] {t} 타입의 가용 샘플({len(pool)})이 요청 수({n})보다 적습니다. "
                f"전체 사용.",
                file=sys.stderr,
            )
            chosen = pool
        else:
            # 결정론적 재현성을 위해 정렬 후 앞에서 n개 선택
            # (랜덤 시드 없이도 재현 가능하도록)
            chosen = sorted(pool)[:n]
        sampled.extend(chosen)

    return sampled


# ---------------------------------------------------------------------------
# 5. 출력 헬퍼
# ---------------------------------------------------------------------------

def _print_results(result: dict) -> None:
    """compute_agreement 결과를 표준 형식으로 출력."""
    n = result["n_common"]
    em = result["exact_match_rate"]
    mean_k = result["mean_kappa"]
    per_type = result["per_type_kappa"]
    disagreements = result["disagreements"]

    print(f"\n공통 항목 수: {n}")
    print(f"Exact Match Rate : {em * 100:.1f}%  (set 완전 일치)")
    print(f"Mean Kappa       : {mean_k:.3f}  (target >= 0.70, 타입별 이진 Kappa 평균)")
    print()
    print("Per-type binary Kappa:")
    print(f"  {'Type':<6} {'Kappa':>8}")
    print(f"  {'-'*18}")
    for cat in CATEGORIES:
        k = per_type.get(cat)
        k_str = f"{k:.3f}" if k is not None else "N/A"
        print(f"  {cat:<6} {k_str:>8}")
    print()
    print(f"Disagreement cases: {len(disagreements)}")
    if disagreements:
        print("  task_id  | Annotator1        | Annotator2")
        print("  " + "-" * 48)
        for d in disagreements:
            a1 = ",".join(d["ann1"])
            a2 = ",".join(d["ann2"])
            print(f"  {d['task_id']:>7}  | {a1:<17} | {a2}")


# ---------------------------------------------------------------------------
# 6. main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inter-Annotator Agreement (IAA) 측정 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ann1",
        type=str,
        default=None,
        help="첫 번째 annotator의 레이블 JSON 파일 경로 (task_id → type)",
    )
    parser.add_argument(
        "--ann2",
        type=str,
        default=None,
        help="두 번째 annotator의 레이블 JSON 파일 경로 (task_id → type)",
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        help=(
            "Blind re-annotation 모드: thinkgeo_taxonomy_labels.json에서 "
            "stratified sample 50개를 추출하여 샘플 목록을 출력한다."
        ),
    )
    parser.add_argument(
        "--taxonomy",
        type=str,
        default=str(TAXONOMY_LABELS_PATH),
        help=f"Taxonomy 레이블 JSON 경로 (기본값: {TAXONOMY_LABELS_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="샘플 목록을 저장할 JSON 파일 경로 (--blind 모드에서만 사용)",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # blind 모드: stratified sample 출력
    # ------------------------------------------------------------------
    if args.blind:
        print("=== Blind Re-annotation 모드 ===")
        print(f"Taxonomy 파일: {args.taxonomy}")

        taxonomy = load_labels(args.taxonomy)
        sampled_ids = stratified_sample(taxonomy, DEFAULT_N_PER_TYPE)

        print(f"\nStratified sample {len(sampled_ids)}개 추출:")
        print(f"  할당: {DEFAULT_N_PER_TYPE}")
        print()
        print("task_id 목록 (레이블링 대상):")
        print("-" * 40)

        sample_entries = []
        for i, tid in enumerate(sampled_ids, 1):
            true_label = taxonomy.get(tid, "?")
            print(f"  {i:>3}. task_id={tid:>5}  (정답 숨김)")
            sample_entries.append({"task_id": tid, "your_label": ""})

        print("-" * 40)
        print(
            "\n위 task_id에 대해 thinkgeo_taxonomy_labels.json의 query를 보고\n"
            "레이블을 부여하세요. 레이블 종류: " + ", ".join(CATEGORIES)
        )
        print(
            "\n작성 방법: 아래 JSON 형식으로 저장 후\n"
            "  python annotation/iaa_measurement.py --ann1 <원본> --ann2 <내 레이블>\n"
            "명령으로 IAA를 계산하세요.\n"
        )

        if args.output:
            out_path = Path(args.output)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {e["task_id"]: e["your_label"] for e in sample_entries},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"샘플 목록을 {out_path}에 저장했습니다. your_label 필드를 채우세요.")

        return

    # ------------------------------------------------------------------
    # 일반 모드: ann1 vs ann2 비교
    # ------------------------------------------------------------------
    if args.ann1 is None or args.ann2 is None:
        parser.error("--ann1 과 --ann2 파일 경로가 모두 필요합니다. (또는 --blind 사용)")

    print(f"Annotator 1: {args.ann1}")
    print(f"Annotator 2: {args.ann2}")

    ann1 = load_labels(args.ann1)
    ann2 = load_labels(args.ann2)

    print(f"  ann1 항목 수: {len(ann1)}")
    print(f"  ann2 항목 수: {len(ann2)}")

    result = compute_agreement(ann1, ann2)
    _print_results(result)

    # 기준치 도달 여부 안내
    mean_k = result["mean_kappa"]
    if mean_k == mean_k:  # nan 체크
        if mean_k >= 0.70:
            print(f"\n[PASS] Mean κ={mean_k:.3f} ≥ 0.70 — IAA 기준 충족")
        else:
            print(f"\n[FAIL] Mean κ={mean_k:.3f} < 0.70 — disagreement 케이스를 검토하고 가이드라인을 보강하세요.")


if __name__ == "__main__":
    main()
