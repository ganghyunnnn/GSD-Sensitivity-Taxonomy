"""
C2 analysis: matched-pair controlled experiment (3 backbones).

Produces, per backbone:
  1. per-arm accuracy vs its trivial-baseline floor
     (D1 majority 50%, D2 majority class, M1 constant class-median @ +-20%,
      M2 always-zero exact = 0% on occupied cells)
  2. matched D-vs-M pairing on the SAME ship cells (D2 orientation vs M1
     length; McNemar) -- both arms non-degenerate on those cells
  3. GSD manipulation: identical questions on identical images at native /
     0.5x / 0.25x -- D presence stability vs counting MAE degradation
     (Wilcoxon native-vs-0.25x on the 210 occupied cells)
  4. BH correction over all reported p-values.

Inputs : experiments/matched_controlled.json, experiments/dota_val_counting.json
Output : experiments/matched_controlled_analysis.json
Run    : uv run python src/eval/analyze_matched_controlled.py
"""
import json
import statistics
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
MC = ROOT / "experiments/matched_controlled.json"
COUNTS = ROOT / "experiments/dota_val_counting.json"
OUT = ROOT / "experiments/matched_controlled_analysis.json"

BACKBONES = ["qwen25vl", "internvl35", "qwen3vl"]
NATIVE_KEY = {"qwen25vl": "vlm", "internvl35": "vlm_ivl", "qwen3vl": "vlm_q3"}
M1_TOL = 0.2


def correct(q, ans):
    if ans is None:
        return False
    if q["arm"] in ("D1", "D1s", "D2"):
        return ans == q["gt"]
    if q["arm"] == "M1":
        return abs(ans - q["gt"]) <= M1_TOL * q["gt"]
    return int(ans) == q["gt"]


def mcnemar(pairs):
    a = sum(1 for x, y in pairs if x and not y)
    b = sum(1 for x, y in pairs if y and not x)
    p = stats.binomtest(min(a, b), a + b).pvalue if a + b else 1.0
    return {"n": len(pairs), "only_first": a, "only_second": b,
            "p": float(f"{p:.3e}")}


def bh(pvals):
    """Benjamini-Hochberg adjusted p-values, order preserved."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank_from_top in range(m - 1, -1, -1):
        i = order[rank_from_top]
        val = min(prev, pvals[i] * m / (rank_from_top + 1))
        adj[i] = val
        prev = val
    return adj


def main():
    mc = json.load(open(MC, encoding="utf-8"))["questions"]
    qs = list(mc.values())
    counting = json.load(open(COUNTS, encoding="utf-8"))["rows"]
    present = {(r["image"], r["cls"]): r for r in counting if r["gt"] > 0}

    m1_gts = [q["gt"] for q in qs if q["arm"] == "M1"]
    m1_median = statistics.median(m1_gts)
    m1_floor = 100 * sum(1 for g in m1_gts
                         if abs(m1_median - g) <= M1_TOL * g) / len(m1_gts)
    d2_gts = [q["gt"] for q in qs if q["arm"] == "D2"]
    d2_floor = 100 * max(d2_gts.count("horizontal"),
                         d2_gts.count("vertical")) / len(d2_gts)

    result = {"floors": {"D1_majority": 50.0, "D2_majority": round(d2_floor, 1),
                         "M1_const_median_tol20": round(m1_floor, 1),
                         "M2_always_zero_on_occupied_exact": 0.0},
              "backbones": {}}
    all_p = []  # (backbone, test_name, p)

    for b in BACKBONES:
        f = f"ans_{b}"
        arms = {}
        for q in qs:
            if f not in q:
                continue
            key = q["arm"] + (f"@{q['scale']}" if q["scale"] != 1.0 else "")
            arms.setdefault(key, []).append(q)

        acc = {}
        for key, items in arms.items():
            n = len(items)
            c = sum(1 for q in items if correct(q, q.get(f)))
            unparsed = sum(1 for q in items if q.get(f) is None)
            acc[key] = {"n": n, "acc_pct": round(100 * c / n, 1),
                        "unparsed": unparsed}

        # matched D2-vs-M1 on identical ship cells
        d2_by_cell = {(q["image"], q["cls"]): correct(q, q.get(f))
                      for q in arms.get("D2", [])}
        m1_by_cell = {(q["image"], q["cls"]): correct(q, q.get(f))
                      for q in arms.get("M1", [])}
        common = sorted(set(d2_by_cell) & set(m1_by_cell))
        pair_d2_m1 = mcnemar([(d2_by_cell[c], m1_by_cell[c]) for c in common])
        all_p.append((b, "D2_vs_M1_mcnemar", pair_d2_m1["p"]))

        # GSD manipulation: counting MAE on the 210 occupied cells
        mae = {}
        native_err, s25_err = {}, {}
        for scale, tag in ((1.0, "native"), (0.5, "0.5"), (0.25, "0.25")):
            errs = {}
            if scale == 1.0:
                for cell, r in present.items():
                    v = r.get(NATIVE_KEY[b])
                    if v is not None:
                        errs[cell] = abs(v - r["gt"])
            else:
                for q in arms.get(f"M2s@{scale}", []):
                    a = q.get(f)
                    if a is not None:
                        errs[(q["image"], q["cls"])] = abs(int(a) - q["gt"])
            mae[tag] = round(sum(errs.values()) / len(errs), 2) if errs else None
            if scale == 1.0:
                native_err = errs
            if scale == 0.25:
                s25_err = errs
        cells = sorted(set(native_err) & set(s25_err))
        try:
            w = stats.wilcoxon([native_err[c] for c in cells],
                               [s25_err[c] for c in cells])
            w_p = float(f"{w.pvalue:.3e}")
        except ValueError:
            w_p = 1.0
        all_p.append((b, "M2_native_vs_025_wilcoxon", w_p))

        # D presence stability across scales (same 200-cell D1s set)
        d_stab = {k: acc[k]["acc_pct"] for k in ("D1", "D1s@0.5", "D1s@0.25")
                  if k in acc}

        result["backbones"][b] = {
            "arm_accuracy": acc,
            "paired_D2_vs_M1_same_ship_cells": pair_d2_m1,
            "counting_mae_occupied_by_scale": mae,
            "counting_native_vs_025_wilcoxon_p": w_p,
            "d_presence_stability": d_stab,
        }

    adj = bh([p for _, _, p in all_p])
    result["bh_correction"] = [
        {"backbone": b, "test": t, "p": p, "p_bh": float(f"{a:.3e}")}
        for (b, t, p), a in zip(all_p, adj)]

    json.dump(result, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print("floors:", result["floors"])
    for b in BACKBONES:
        r = result["backbones"][b]
        print(f"\n== {b} ==")
        for k in sorted(r["arm_accuracy"]):
            a = r["arm_accuracy"][k]
            print(f"  {k:10s} n={a['n']:3d} acc={a['acc_pct']:5.1f}% unparsed={a['unparsed']}")
        print(f"  D2 vs M1 paired (n={r['paired_D2_vs_M1_same_ship_cells']['n']}): "
              f"p={r['paired_D2_vs_M1_same_ship_cells']['p']}")
        print(f"  MAE by scale: {r['counting_mae_occupied_by_scale']} "
              f"(native vs 0.25x Wilcoxon p={r['counting_native_vs_025_wilcoxon_p']})")
    print("\nBH-adjusted:")
    for row in result["bh_correction"]:
        print(f"  {row['backbone']:10s} {row['test']:28s} p={row['p']}  p_bh={row['p_bh']}")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
