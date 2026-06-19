"""
실험 결과를 논문 LaTeX 플레이스홀더에 자동 삽입.

RTERM: Routed prompting 결과 (실험 A)
FTERM: FloodNet 평가 결과 (실험 B)

사용법:
    python src/eval/fill_paper_placeholders.py
    python src/eval/fill_paper_placeholders.py --dry-run
"""
import argparse
import json
import re
from pathlib import Path
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parent.parent.parent
ROUTED_PATH  = ROOT / "annotation/routed_eval_qwen3vl_8b.json"
FLOODNET_PATH = ROOT / "annotation/floodnet_backbone_qwen3vl_8b.json"
TEX_PATH = ROOT / "paper/Identifying_the_Measurement_Gap_in_Remote_Sensing_VQA_with_a_GSD-Sensitive_Taxonomy.tex"

# baseline (직접 프롬프팅) 수치 - backbone_qwen3vl_8b.json에서 가져옴
BASELINE_PATH = ROOT / "annotation/backbone_qwen3vl_8b.json"


def z_test_one_sided(n1, fail1, n2, fail2):
    """두 비율의 단측 z-test (p2 > p1)."""
    p1 = fail1 / n1
    p2 = fail2 / n2
    p_pool = (fail1 + fail2) / (n1 + n2)
    se = (p_pool * (1 - p_pool) * (1/n1 + 1/n2)) ** 0.5
    if se == 0:
        return 1.0
    z = (p2 - p1) / se
    return scipy_stats.norm.sf(z)


def compute_routed_terms():
    if not ROUTED_PATH.exists():
        return None
    with open(ROUTED_PATH, encoding="utf-8") as f:
        routed = json.load(f)
    with open(BASELINE_PATH, encoding="utf-8") as f:
        baseline = json.load(f)

    bp = baseline["direct"]["per_type"]
    rp = routed["per_type"]

    d_direct  = round(100 * (1 - bp["D"]["accuracy"]), 1)
    m1_direct = round(100 * (1 - bp["M1"]["accuracy"]), 1)
    m2_direct = round(100 * (1 - bp["M2"]["accuracy"]), 1)

    d_routed  = round(100 * rp["D"]["failure_rate"], 1)
    m1_routed = round(100 * rp["M1"]["failure_rate"], 1)
    m2_routed = round(100 * rp["M2"]["failure_rate"], 1)

    # M overall (M1 + M2 합산)
    m1_n = rp["M1"]["total"]; m1_c = rp["M1"]["correct"]
    m2_n = rp["M2"]["total"]; m2_c = rp["M2"]["correct"]
    m_n = m1_n + m2_n; m_c = m1_c + m2_c
    m_routed_fail = round(100 * (1 - m_c / m_n), 1)
    gap_routed = round(m_routed_fail - d_routed, 1)

    # baseline gap
    m1_n_b = bp["M1"]["total"]; m1_c_b = bp["M1"]["correct"]
    m2_n_b = bp["M2"]["total"]; m2_c_b = bp["M2"]["correct"]
    m_n_b = m1_n_b + m2_n_b; m_c_b = m1_c_b + m2_c_b
    m_direct_fail = round(100 * (1 - m_c_b / m_n_b), 1)
    gap_direct = round(m_direct_fail - d_direct, 1)

    m1_delta = round(m1_direct - m1_routed, 1)
    m2_delta = round(m2_direct - m2_routed, 1)

    return {
        "d_direct":  f"{d_direct:.1f}\\%",
        "m1_direct": f"{m1_direct:.1f}\\%",
        "m2_direct": f"{m2_direct:.1f}\\%",
        "gap_direct": f"+{gap_direct}\\,pp",
        "d_routed":  f"{d_routed:.1f}\\%",
        "m1_routed": f"{m1_routed:.1f}\\%",
        "m2_routed": f"{m2_routed:.1f}\\%",
        "gap_routed": f"+{gap_routed}\\,pp",
        "m1_delta":  str(m1_delta),
        "m2_delta":  str(m2_delta),
        "m1_fail":   str(m1_routed),
    }


def compute_floodnet_terms():
    if not FLOODNET_PATH.exists():
        return None
    with open(FLOODNET_PATH, encoding="utf-8") as f:
        fn = json.load(f)

    pt = fn["per_taxonomy"]
    d_fail  = round(100 * pt["D"]["failure_rate"], 1)
    m2_fail = round(100 * pt["M2"]["failure_rate"], 1)
    gap = round(m2_fail - d_fail, 1)
    nd = pt["D"]["total"]
    nm2 = pt["M2"]["total"]

    # p-value
    p_val = z_test_one_sided(nd, int(pt["D"]["failure_rate"] * nd),
                             nm2, int(pt["M2"]["failure_rate"] * nm2))

    return {
        "nd":      str(nd),
        "nm2":     str(nm2),
        "d_fail":  f"{d_fail:.1f}",
        "m2_fail": f"{m2_fail:.1f}",
        "gap":     f"+{gap:.1f}",
    }


def fill_tex(tex: str, prefix: str, terms: dict | None) -> str:
    if terms is None:
        return tex
    for key, val in terms.items():
        # LaTeX files use \_ for underscores inside commands
        latex_key = key.replace("_", r"\_")
        placeholder = f"\\{prefix}{{{latex_key}}}"
        tex = tex.replace(placeholder, val)
    return tex


def main(dry_run: bool = False):
    with open(TEX_PATH, encoding="utf-8") as f:
        tex = f.read()

    r_terms = compute_routed_terms()
    f_terms = compute_floodnet_terms()

    if r_terms:
        print("Routed prompting terms:", r_terms)
    else:
        print("Routed eval not found, skipping RTERM fills")

    if f_terms:
        print("FloodNet terms:", f_terms)
    else:
        print("FloodNet eval not found, skipping FTERM fills")

    new_tex = fill_tex(tex, "RTERM", r_terms)
    new_tex = fill_tex(new_tex, "FTERM", f_terms)

    if dry_run:
        remaining = re.findall(r"\\[RF]TERM\{[^}]+\}", new_tex)
        if remaining:
            print(f"\n미채워진 플레이스홀더 {len(remaining)}개:", remaining)
        else:
            print("\n모든 플레이스홀더 채워짐")
    else:
        with open(TEX_PATH, "w", encoding="utf-8") as f:
            f.write(new_tex)
        print(f"\n저장: {TEX_PATH}")
        remaining = re.findall(r"\\[RF]TERM\{[^}]+\}", new_tex)
        if remaining:
            print(f"미채워진 플레이스홀더 {len(remaining)}개 남음")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
