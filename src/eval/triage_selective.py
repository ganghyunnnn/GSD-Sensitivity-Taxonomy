"""
Design A -- Taxonomy-gated selective prediction (triage / abstention).

The GSD resolvability criterion (GSD <= d/s) is used as a *competence gate*:
resolution-limited M2 queries (GSD > d/s, where object instances are physically
unresolvable) are DEFERRED ("insufficient resolution; acquire finer imagery")
instead of answered with a confident-but-wrong count.

This is a reliability / selective-prediction result (not a raw-accuracy claim):
we report the risk-coverage trade-off -- error rate on the *answered* subset vs.
coverage -- across three backbones, by joining the per-item resolvability flag
(intrinsic to image+question, from experiments/m2_resolution_analysis.json) with
each backbone's per-item M2 correctness.

No new inference: recomputed entirely on stored results.

Usage:
    python src/eval/triage_selective.py
"""
import io
import json
import sys
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent
ANN = ROOT / "annotation"
EXP = ROOT / "experiments"

BACKBONES = {
    "Qwen3-VL-8B":    "backbone_qwen3vl_8b.json",
    "Qwen2.5-VL-7B":  "backbone_qwen25vl_7b.json",
    "InternVL3.5-8B": "backbone_internvl35_8b.json",
}


def load_resolvability():
    """task_id -> resolvable (bool) | None if GSD unknown. Intrinsic to image+question."""
    d = json.load(open(EXP / "m2_resolution_analysis.json", encoding="utf-8"))
    flag = {}
    for r in d["records"]:
        tid = str(r["task_id"])
        if r.get("gsd") is None:
            flag[tid] = None
        else:
            flag[tid] = (r["gsd"] <= r["threshold_gsd"])
    return flag


def load_m2_correct(fname):
    """task_id -> correct (bool) for the 83 M2 tasks under direct prompting."""
    d = json.load(open(ANN / fname, encoding="utf-8"))
    res = d["direct"]["results"]
    return {str(r["task_id"]): bool(r["correct"]) for r in res if r.get("gt_type") == "M2"}


def main():
    resolv = load_resolvability()
    print("=== Design A: taxonomy-gated selective prediction (M2 triage) ===")
    print("Policy: ABSTAIN when GSD > d/s (resolution-limited); ANSWER otherwise.\n")
    print(f"{'Backbone':16s} {'Ntot':>4} {'base_err':>8} | "
          f"{'cov':>5} {'ans_err':>7} {'err_avoid':>9} {'sacrificed':>10}")
    out = {}
    for label, fname in BACKBONES.items():
        corr = load_m2_correct(fname)
        # join on tasks that exist in both correctness and resolvability map
        tids = [t for t in corr if t in resolv]
        N = len(tids)
        base_fail = sum(1 for t in tids if not corr[t])
        # triage: abstain only where resolvable is known-False
        answered = [t for t in tids if resolv[t] is not False]   # answer if resolvable or unknown
        abstained = [t for t in tids if resolv[t] is False]
        ans_fail = sum(1 for t in answered if not corr[t])
        # errors avoided = failures among abstained; sacrificed = corrects among abstained
        err_avoided = sum(1 for t in abstained if not corr[t])
        sacrificed = sum(1 for t in abstained if corr[t])
        cov = len(answered) / N * 100 if N else 0
        base_err = base_fail / N * 100 if N else 0
        ans_err = ans_fail / len(answered) * 100 if answered else 0
        print(f"{label:16s} {N:>4} {base_err:>7.1f}% | "
              f"{cov:>4.1f}% {ans_err:>6.1f}% {err_avoided:>9d} {sacrificed:>10d}")
        out[label] = {
            "N": N, "baseline_error_pct": round(base_err, 1),
            "coverage_pct": round(cov, 1), "answered_error_pct": round(ans_err, 1),
            "errors_avoided": err_avoided, "corrects_sacrificed": sacrificed,
            "n_abstained": len(abstained),
            "error_reduction_pp": round(base_err - ans_err, 1),
        }

    # aggregate sanity line
    print("\nInterpretation: 'errors_avoided' = silent wrong answers converted to honest")
    print("abstentions; 'answered_error_pct' < 'baseline_error_pct' = reliability gain on")
    print("the answered subset. Counting remains hard even when resolvable (model-error),")
    print("so the accuracy lift is modest -- the value is in NOT answering the impossible.")

    (EXP).mkdir(exist_ok=True)
    json.dump(
        {"description": "Design A: taxonomy-gated selective prediction on M2. "
                        "Abstain when GSD>d/s (resolution-limited). Recomputed on stored "
                        "results; no new inference.",
         "policy": "abstain if not resolvable (GSD>d/s); answer if resolvable or GSD unknown",
         "backbones": out},
        open(EXP / "triage_selective.json", "w", encoding="utf-8"),
        indent=2, ensure_ascii=False)
    print(f"\nSaved: experiments/triage_selective.json")


if __name__ == "__main__":
    main()
