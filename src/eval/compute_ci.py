"""
Bootstrap 95% CI 계산 스크립트.

소표본에서 F1 score의 신뢰 구간을 추정한다.
실험 결과가 확정되면 하드코딩된 TP/FP/FN 값을 업데이트하여 사용한다.

사용법:
  python src/eval/compute_ci.py
"""

import json
import random
import sys
import io

# Windows 콘솔에서 UTF-8 출력 강제 설정
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 1. F1 계산 (TP, FP, FN 기반)
# ---------------------------------------------------------------------------

def _f1_from_lists(y_true: list, y_pred: list) -> float:
    """이진 레이블(1=양성) 리스트에서 F1 스코어 계산.

    Parameters
    ----------
    y_true, y_pred : list of int (0 or 1)

    Returns
    -------
    float
        F1 스코어. TP=0이면 0.0 반환.
    """
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)

    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return (2 * tp) / denom


# ---------------------------------------------------------------------------
# 2. Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_f1_ci(
    y_true: list,
    y_pred: list,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple:
    """Bootstrap 리샘플링으로 F1 스코어의 신뢰 구간 추정.

    Parameters
    ----------
    y_true : list of int
        실제 레이블 (0 or 1)
    y_pred : list of int
        예측 레이블 (0 or 1)
    n_bootstrap : int
        리샘플링 반복 횟수 (기본값 10,000)
    ci : float
        신뢰 수준 (기본값 0.95 → 95% CI)
    seed : int
        재현성을 위한 random seed

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) — 백분위수 기반 CI
    """
    n = len(y_true)
    if n == 0:
        return (float("nan"), float("nan"))

    rng = random.Random(seed)
    indices = list(range(n))

    f1_samples = []
    for _ in range(n_bootstrap):
        sample_idx = [rng.choice(indices) for _ in range(n)]
        yt = [y_true[i] for i in sample_idx]
        yp = [y_pred[i] for i in sample_idx]
        f1_samples.append(_f1_from_lists(yt, yp))

    f1_samples.sort()

    alpha = 1.0 - ci
    lower_idx = int((alpha / 2.0) * n_bootstrap)
    upper_idx = int((1.0 - alpha / 2.0) * n_bootstrap) - 1

    lower_idx = max(0, min(lower_idx, n_bootstrap - 1))
    upper_idx = max(0, min(upper_idx, n_bootstrap - 1))

    return (f1_samples[lower_idx], f1_samples[upper_idx])


# ---------------------------------------------------------------------------
# 3. 타입별 CI 계산 유틸리티
# ---------------------------------------------------------------------------

def _reconstruct_binary_labels(n: int, tp: int, fp: int, fn: int) -> tuple:
    """TP, FP, FN 추정값으로부터 이진 레이블 리스트를 역산.

    Parameters
    ----------
    n : int
        전체 샘플 수
    tp : int
        True Positive 수
    fp : int
        False Positive 수
    fn : int
        False Negative 수

    Returns
    -------
    tuple[list, list]
        (y_true, y_pred) — 이진 레이블 리스트
    """
    tn = n - tp - fp - fn
    if tn < 0:
        tn = 0

    y_true = [1] * (tp + fn) + [0] * (fp + tn)
    y_pred = [1] * tp + [0] * fn + [1] * fp + [0] * tn

    return y_true, y_pred


def compute_type_ci(type_results: list[dict]) -> list[dict]:
    """타입별 F1 결과를 하드코딩하여 Bootstrap 95% CI 계산.

    Parameters
    ----------
    type_results : list of dict
        각 dict는 {"name", "f1", "n", "tp", "fp", "fn"} 포함

    Returns
    -------
    list of dict
        CI 계산 결과
    """
    ci_results = []
    header = f"{'Type':<14} {'F1':>6} {'Recomputed':>10} {'95% CI':>20}"
    print(header)
    print("-" * len(header))

    for r in type_results:
        y_true, y_pred = _reconstruct_binary_labels(
            n=r["n"], tp=r["tp"], fp=r["fp"], fn=r["fn"]
        )
        actual_f1 = _f1_from_lists(y_true, y_pred)
        lower, upper = bootstrap_f1_ci(
            y_true, y_pred, n_bootstrap=10000, ci=0.95, seed=42
        )

        print(
            f"  {r['name']:<12} {r['f1']:>6.3f} {actual_f1:>10.3f}"
            f"  [{lower:.3f}, {upper:.3f}]"
        )

        ci_results.append({
            "name": r["name"],
            "f1_reported": r["f1"],
            "f1_computed": actual_f1,
            "ci_lower": lower,
            "ci_upper": upper,
        })

    return ci_results


# ---------------------------------------------------------------------------
# 4. main (실험 결과 확정 후 type_results 채워서 사용)
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Bootstrap 95% CI - Taxonomy Type F1 Scores")
    print("=" * 60)
    print()
    print("NOTE: 실험 결과 확정 후 type_results 딕셔너리를 업데이트하세요.")
    print()

    # TODO: 실험 결과가 확정되면 아래 값을 업데이트
    # 형식: {"name": "<type>_<system>", "f1": float, "n": int, "tp": int, "fp": int, "fn": int}
    type_results: list[dict] = []

    if not type_results:
        print("실험 결과가 아직 입력되지 않았습니다.")
        return

    ci_results = compute_type_ci(type_results)

    print()
    print("참고 - 추정 검증:")
    for r in ci_results:
        diff = abs(r["f1_reported"] - r["f1_computed"])
        status = "OK" if diff < 0.05 else "WARNING: 추정 오차 큼"
        print(
            f"  {r['name']}: 보고값={r['f1_reported']:.3f}, "
            f"재계산={r['f1_computed']:.3f}, "
            f"오차={diff:.3f} [{status}]"
        )


if __name__ == "__main__":
    main()
