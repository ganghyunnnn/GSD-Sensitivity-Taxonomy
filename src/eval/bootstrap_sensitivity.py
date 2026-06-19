"""
Bootstrap Sensitivity Analysis for M1 Prevalence Bias (C2)

평가 집합 내 M1 비율(21.2%)이 실제 분포(51.4%)보다 낮은 선택 편향이
D/M 실패율 격차의 통계적 유의성에 미치는 영향을 정량화한다.

방법:
    현재 평가 집합(D=66, M1=40, M2=83)에서 M1 비율을 21%~51%로 변화시키며
    M overall과 D의 실패율 격차가 유의미한지 bootstrap으로 검증.

사용법:
    python src/eval/bootstrap_sensitivity.py
    python src/eval/bootstrap_sensitivity.py --n-bootstrap 5000
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy.stats import norm as _norm

def proportions_ztest(count, nobs, alternative="larger"):
    """Minimal two-proportion z-test (one-sided, larger)."""
    p1 = count[0] / nobs[0]
    p2 = count[1] / nobs[1]
    p_pool = (count[0] + count[1]) / (nobs[0] + nobs[1])
    se = (p_pool * (1 - p_pool) * (1/nobs[0] + 1/nobs[1])) ** 0.5
    if se == 0:
        return 0.0, 0.5
    z = (p1 - p2) / se
    pv = 1 - _norm.cdf(z) if alternative == "larger" else 2 * (1 - _norm.cdf(abs(z)))
    return z, pv


# ──────────────────────────────────────────────
# 관찰된 데이터 (Table 2, Vanilla ReAct 기준)
# ──────────────────────────────────────────────
OBSERVED = {
    "D":  {"n": 66,  "fail": 30},   # 45.5%
    "M1": {"n": 40,  "fail": 31},   # 77.5%
    "M2": {"n": 83,  "fail": 56},   # 67.5%
}

# 재이론화된 M1 비율 시나리오 (eval split 21.2% → excluded split 51.4%)
M1_PREVALENCE_SCENARIOS = np.linspace(0.21, 0.51, 11)

TOTAL_EVAL = OBSERVED["D"]["n"] + OBSERVED["M1"]["n"] + OBSERVED["M2"]["n"]  # 189


def simulate_dm_gap(
    m1_prev: float,
    observed: dict,
    n_bootstrap: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    주어진 M1 비율(m1_prev)에서 bootstrap으로 D/M 격차의 분포를 추정한다.

    M1 비율을 높이면 M overall 실패율이 높아지므로 (M1 실패율이 가장 높으므로)
    D/M 격차는 현재보다 더 커질 것을 보인다 (lower bound 논증).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    total = TOTAL_EVAL
    n_d  = observed["D"]["n"]
    n_m1 = observed["M1"]["n"]
    n_m2 = observed["M2"]["n"]

    p_d  = observed["D"]["fail"]  / n_d
    p_m1 = observed["M1"]["fail"] / n_m1
    p_m2 = observed["M2"]["fail"] / n_m2

    # 시나리오 N 계산: M1 비율에 따라 D, M1, M2 크기 재배분
    # D 비율은 고정(29.6%), M2는 나머지
    d_prev  = observed["D"]["n"] / total          # 34.9%
    m2_prev = 1.0 - d_prev - m1_prev
    if m2_prev <= 0:
        return {"m1_prev": m1_prev, "gap_mean": np.nan, "gap_ci_lo": np.nan,
                "gap_ci_hi": np.nan, "p_value": np.nan, "significant": False}

    n_d_new  = max(1, round(total * d_prev))
    n_m1_new = max(1, round(total * m1_prev))
    n_m2_new = max(1, total - n_d_new - n_m1_new)

    gaps = []
    p_values = []

    for _ in range(n_bootstrap):
        # bootstrap 샘플: 관찰된 실패율을 모수로 이항 샘플링
        f_d  = rng.binomial(n_d_new,  p_d)  / n_d_new
        f_m1 = rng.binomial(n_m1_new, p_m1) / n_m1_new
        f_m2 = rng.binomial(n_m2_new, p_m2) / n_m2_new

        # M overall 가중 평균
        n_m_new = n_m1_new + n_m2_new
        f_m = (f_m1 * n_m1_new + f_m2 * n_m2_new) / n_m_new

        gap = f_m - f_d
        gaps.append(gap)

        # two-proportion z-test
        f_d_c  = rng.binomial(n_d_new,  p_d)
        f_m_c  = rng.binomial(n_m_new,  (p_m1 * n_m1_new + p_m2 * n_m2_new) / n_m_new)
        _, pv = proportions_ztest(
            [f_m_c, f_d_c], [n_m_new, n_d_new], alternative="larger"
        )
        p_values.append(pv)

    gaps = np.array(gaps)
    return {
        "m1_prev": m1_prev,
        "n_d": n_d_new,
        "n_m1": n_m1_new,
        "n_m2": n_m2_new,
        "gap_mean": float(np.mean(gaps) * 100),
        "gap_ci_lo": float(np.percentile(gaps, 2.5) * 100),
        "gap_ci_hi": float(np.percentile(gaps, 97.5) * 100),
        "frac_significant": float(np.mean(np.array(p_values) < 0.05)),
        "significant": bool(np.mean(np.array(p_values) < 0.05) >= 0.80),
    }


def run_sensitivity(n_bootstrap: int = 2000, output_path: Path | None = None):
    rng = np.random.default_rng(42)
    results = []

    print(f"{'M1%':>6} {'N_D':>5} {'N_M1':>6} {'N_M2':>6} "
          f"{'Gap(pp)':>9} {'95% CI':>18} {'Sig≥80%':>8}")
    print("-" * 70)

    for m1_prev in M1_PREVALENCE_SCENARIOS:
        res = simulate_dm_gap(m1_prev, OBSERVED, n_bootstrap=n_bootstrap, rng=rng)
        results.append(res)

        ci_str = f"[{res['gap_ci_lo']:.1f}, {res['gap_ci_hi']:.1f}]"
        sig_str = "YES" if res["significant"] else "no"
        print(f"{m1_prev*100:>6.1f} {res['n_d']:>5} {res['n_m1']:>6} {res['n_m2']:>6} "
              f"{res['gap_mean']:>9.1f} {ci_str:>18} {sig_str:>8}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    # 핵심 결론 요약
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER (C2 lower-bound argument):")
    obs_gap = (
        (OBSERVED["M1"]["fail"] + OBSERVED["M2"]["fail"])
        / (OBSERVED["M1"]["n"] + OBSERVED["M2"]["n"])
        - OBSERVED["D"]["fail"] / OBSERVED["D"]["n"]
    ) * 100
    print(f"  Observed D/M gap (current 21% M1): {obs_gap:.1f} pp")

    # M1 비율 51.4% 시나리오
    res51 = [r for r in results if abs(r["m1_prev"] - 0.51) < 0.01]
    if res51:
        r = res51[0]
        print(f"  Projected gap (51% M1 scenario):  {r['gap_mean']:.1f} pp "
              f"[{r['gap_ci_lo']:.1f}, {r['gap_ci_hi']:.1f}]")
        print(f"  Significant (p<0.05) in ≥80% bootstrap: {r['significant']}")

    all_sig = all(r["significant"] for r in results if not np.isnan(r.get("gap_mean", float("nan"))))
    print(f"  Gap significant across all M1% scenarios: {all_sig}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Bootstrap M1 prevalence sensitivity analysis")
    parser.add_argument("--n-bootstrap", type=int, default=2000,
                        help="Number of bootstrap iterations (default: 2000)")
    parser.add_argument("--output", type=str,
                        default="annotation/bootstrap_sensitivity.json",
                        help="Output JSON path")
    args = parser.parse_args()

    output_path = Path(args.output)
    run_sensitivity(n_bootstrap=args.n_bootstrap, output_path=output_path)


if __name__ == "__main__":
    main()
