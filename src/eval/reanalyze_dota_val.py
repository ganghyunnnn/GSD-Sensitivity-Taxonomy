"""
C6 reanalysis of the DOTA-val detector-vs-VLM counting experiment.

Addresses the empty-cell base-rate problem in exact-match counting evaluation:
590/800 cells have GT=0, so exact-match rates are dominated by the zero base
rate (an always-zero responder scores 73.8%). This script produces the honest
replacement numbers:

  1. overall + trivial always-zero baseline (exact, within-1, MAE)
  2. present-cell (GT>0, n=210) stratification with paired tests
  3. ordinal-bucket re-scoring (0 / 1-5 / 6-20 / 21-50 / >50) -> answer-space
     control for C2 (bucket space comparable to a small multiple-choice set)
  4. resolvability filter recomputed with per-image GSD (nominal object size,
     GSD <= d/s, s=10) instead of the pixel-count no-op

Inputs : experiments/dota_val_counting.json, experiments/dota_val_gsd_map.json
Output : experiments/dota_val_reanalysis.json
Run    : uv run python src/eval/reanalyze_dota_val.py
"""
import json
import math
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
COUNTS = ROOT / "experiments/dota_val_counting.json"
GSDMAP = ROOT / "experiments/dota_val_gsd_map.json"
OUT = ROOT / "experiments/dota_val_reanalysis.json"

SYSTEMS = {"det": "det", "vlm_q25": "vlm", "vlm_q3": "vlm_q3", "vlm_ivl": "vlm_ivl"}
BUCKETS = [(0, 0), (1, 5), (6, 20), (21, 50), (51, 10 ** 9)]
S_FLOOR = 10
NOMINAL_M = {"plane": 30.0, "ship": 25.0, "small vehicle": 4.5, "large vehicle": 12.0}


def bucket(x):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= x <= hi:
            return i
    return len(BUCKETS) - 1


def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(100 * (centre - half), 1), round(100 * (centre + half), 1)]


def metrics(rows, key):
    vals = [(r[key], r["gt"]) for r in rows if r.get(key) is not None]
    n = len(vals)
    exact = sum(1 for p, g in vals if p == g)
    within1 = sum(1 for p, g in vals if abs(p - g) <= 1)
    bucket_ok = sum(1 for p, g in vals if bucket(p) == bucket(g))
    mae = sum(abs(p - g) for p, g in vals) / n
    return {"n": n,
            "exact_pct": round(100 * exact / n, 1),
            "exact_wilson95": wilson(exact, n),
            "within1_pct": round(100 * within1 / n, 1),
            "bucket_pct": round(100 * bucket_ok / n, 1),
            "mae": round(mae, 2)}


def trivial_zero(rows):
    n = len(rows)
    exact = sum(1 for r in rows if r["gt"] == 0)
    within1 = sum(1 for r in rows if r["gt"] <= 1)
    bucket_ok = exact  # bucket(0)==bucket(gt) iff gt==0
    mae = sum(r["gt"] for r in rows) / n
    return {"n": n, "exact_pct": round(100 * exact / n, 1),
            "exact_wilson95": wilson(exact, n),
            "within1_pct": round(100 * within1 / n, 1),
            "bucket_pct": round(100 * bucket_ok / n, 1),
            "mae": round(mae, 2)}


def paired_tests(rows, key_a, key_b):
    """det vs vlm paired on identical cells: McNemar (exact) + Wilcoxon |err|."""
    pairs = [(r[key_a], r[key_b], r["gt"]) for r in rows
             if r.get(key_a) is not None and r.get(key_b) is not None]
    a_only = sum(1 for a, b, g in pairs if a == g and b != g)
    b_only = sum(1 for a, b, g in pairs if a != g and b == g)
    mcnemar_p = stats.binomtest(min(a_only, b_only), a_only + b_only).pvalue \
        if (a_only + b_only) else 1.0
    ea = [abs(a - g) for a, b, g in pairs]
    eb = [abs(b - g) for a, b, g in pairs]
    try:
        w_p = stats.wilcoxon(ea, eb).pvalue if any(x != y for x, y in zip(ea, eb)) else 1.0
    except ValueError:
        w_p = 1.0
    return {"n": len(pairs), "a_only_correct": a_only, "b_only_correct": b_only,
            "mcnemar_p": float(f"{mcnemar_p:.2e}"), "wilcoxon_abs_err_p": float(f"{w_p:.2e}")}


def main():
    counts = json.load(open(COUNTS, encoding="utf-8"))
    rows = counts["rows"]
    gsdmap = json.load(open(GSDMAP, encoding="utf-8"))["images"]

    present = [r for r in rows if r["gt"] > 0]

    # resolvability filter with per-image GSD + nominal object size
    resolvable, unresolvable, no_gsd = [], [], []
    for r in rows:
        meta = gsdmap.get(r["image"])
        if not meta or meta["flag"] != "ok":
            no_gsd.append(r)
            continue
        if meta["gsd"] <= NOMINAL_M[r["cls"]] / S_FLOOR:
            resolvable.append(r)
        else:
            unresolvable.append(r)

    out = {
        "buckets": [f"{lo}-{hi if hi < 10**9 else 'inf'}" for lo, hi in BUCKETS],
        "all_800": {"trivial_always_zero": trivial_zero(rows),
                    **{name: metrics(rows, key) for name, key in SYSTEMS.items()}},
        "present_210": {"trivial_always_zero": trivial_zero(present),
                        **{name: metrics(present, key) for name, key in SYSTEMS.items()}},
        "paired_present": {
            f"det_vs_{n}": paired_tests(present, "det", k)
            for n, k in SYSTEMS.items() if n != "det"},
        "resolvability_per_image_gsd": {
            "criterion": f"GSD <= d/s, s={S_FLOOR}px, nominal d={NOMINAL_M}",
            "n_resolvable": len(resolvable),
            "n_unresolvable": len(unresolvable),
            "n_unresolvable_nonzero_gt": sum(1 for r in unresolvable if r["gt"] > 0),
            "n_no_valid_gsd": len(no_gsd),
            "resolvable_only": {
                "trivial_always_zero": trivial_zero(resolvable),
                **{name: metrics(resolvable, key) for name, key in SYSTEMS.items()}},
        },
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
