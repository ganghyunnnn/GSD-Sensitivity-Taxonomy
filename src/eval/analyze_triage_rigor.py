"""
Rigorous evaluation of the taxonomy-gated selective-prediction claim (experiment A).

Decisive question: does the GSD competence gate (abstain when GSD>d/s) beat
uncertainty-agnostic abstention, or is it just (a) abstaining on a high-error
subset (no lift over random) and (b) a proxy for the harder dataset (xBD)?

For each of the 3 backbones we compute, on the M2 set:
  1. Operating point: coverage, answered-error, base-error.
  2. Permutation test vs RANDOM abstention at MATCHED coverage
     (does the gate's answered-error beat a random gate of the same size?).
  3. Fisher exact on (abstained vs answered) x (wrong vs right) + BH across backbones.
  4. GSD-margin-ranked risk-coverage curve + AURC vs the random (flat) baseline.
  5. Dataset x resolvable confound table (is the gate ~= a dataset indicator?
     does resolvability separate error WITHIN a dataset?).

No new inference: stored correctness + intrinsic GSD/threshold/dataset only.
(Confidence-abstention baseline is intentionally NOT computed: the stored data
carries no logprob/entropy signal, so an AURC-vs-confidence claim would be
unsupported.)
"""
import io
import json
import sys
import random
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent
ANN = ROOT / "annotation"
EXP = ROOT / "experiments"
random.seed(0)

BACKBONES = {
    "Qwen3-VL-8B":    "backbone_qwen3vl_8b.json",
    "Qwen2.5-VL-7B":  "backbone_qwen25vl_7b.json",
    "InternVL3.5-8B": "backbone_internvl35_8b.json",
}


def load_items():
    """task_id -> {gsd, thr, dataset, resolvable|None}. Intrinsic to image+question."""
    d = json.load(open(EXP / "m2_resolution_analysis.json", encoding="utf-8"))
    items = {}
    for r in d["records"]:
        tid = str(r["task_id"])
        g, thr = r.get("gsd"), r.get("threshold_gsd")
        items[tid] = {
            "gsd": g, "thr": thr, "dataset": r.get("dataset", "other"),
            "resolvable": (g <= thr) if g is not None else None,
            "margin": (thr - g) if (g is not None and thr is not None) else None,
        }
    return items


def load_correct(fname):
    d = json.load(open(ANN / fname, encoding="utf-8"))
    return {str(r["task_id"]): bool(r["correct"])
            for r in d["direct"]["results"] if r.get("gt_type") == "M2"}


def fisher_exact(a, b, c, d):
    """2x2 [[a,b],[c,d]] two-sided Fisher exact p (hypergeometric)."""
    from math import comb
    n = a + b + c + d
    r1, c1 = a + b, a + c
    p_obs = comb(r1, a) * comb(n - r1, c1 - a) / comb(n, c1)
    p = 0.0
    lo = max(0, c1 - (n - r1))
    hi = min(c1, r1)
    for x in range(lo, hi + 1):
        px = comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)
        if px <= p_obs * (1 + 1e-9):
            p += px
    return min(1.0, p)


def bh(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = m - rank + 1
        val = pvals[i] * m / k
        prev = min(prev, val)
        adj[i] = prev
    return adj


def perm_test_vs_random(answered_correct, n_abstain, n_perm=20000):
    """observed = answered-error of the gate. Null: randomly abstain n_abstain of all
    items; p = P(random answered-error <= gate answered-error)."""
    all_correct = answered_correct["all"]  # list of bools over ALL items
    gate_ans = answered_correct["gate_answered"]  # list of bools over gate-answered
    obs_err = 1 - (sum(gate_ans) / len(gate_ans)) if gate_ans else 1.0
    N = len(all_correct)
    n_ans = N - n_abstain
    le = 0
    for _ in range(n_perm):
        samp = random.sample(all_correct, n_ans)
        err = 1 - (sum(samp) / len(samp)) if samp else 1.0
        if err <= obs_err + 1e-12:
            le += 1
    return obs_err, le / n_perm


def aurc_margin(items_corr):
    """Rank by resolvability margin (answer most-resolvable first); sweep coverage.
    Returns (aurc_gate, aurc_random=base_err, curve)."""
    known = [(it["margin"], c) for it, c in items_corr if it["margin"] is not None]
    # most resolvable (largest margin) answered first
    known.sort(key=lambda x: -x[0])
    N = len(known)
    base_err = 1 - sum(c for _, c in known) / N
    curve = []
    cum_correct = 0
    aurc = 0.0
    prev_cov = 0.0
    for k in range(1, N + 1):
        cum_correct += known[k - 1][1]
        cov = k / N
        risk = 1 - cum_correct / k
        curve.append((round(cov, 3), round(risk, 3)))
        aurc += risk * (cov - prev_cov)
        prev_cov = cov
    return aurc, base_err, curve


def main():
    items = load_items()
    print("=" * 72)
    print("RIGOROUS SELECTIVE-PREDICTION ANALYSIS (GSD gate vs random)")
    print("=" * 72)

    fisher_ps = []
    out = {}
    for label, fname in BACKBONES.items():
        corr = load_correct(fname)
        tids = [t for t in corr if t in items]
        # gate: abstain if resolvable is False; answer if True or unknown
        answered = [t for t in tids if items[t]["resolvable"] is not False]
        abstained = [t for t in tids if items[t]["resolvable"] is False]
        N = len(tids)
        base_err = 1 - sum(corr[t] for t in tids) / N
        ans_err = 1 - sum(corr[t] for t in answered) / len(answered)
        # Fisher 2x2: answered/abstained x wrong/right
        aw = sum(1 for t in answered if not corr[t]); ar = len(answered) - aw
        bw = sum(1 for t in abstained if not corr[t]); br = len(abstained) - bw
        fp = fisher_exact(aw, ar, bw, br)
        fisher_ps.append(fp)
        # permutation vs random at matched coverage
        obs_err, perm_p = perm_test_vs_random(
            {"all": [corr[t] for t in tids],
             "gate_answered": [corr[t] for t in answered]},
            n_abstain=len(abstained))
        # AURC (margin-ranked) vs random
        aurc, aurc_base, _ = aurc_margin([(items[t], corr[t]) for t in tids])

        out[label] = {
            "N": N, "coverage": round(len(answered) / N, 3),
            "base_err": round(base_err, 3), "answered_err": round(ans_err, 3),
            "err_reduction_pp": round((base_err - ans_err) * 100, 1),
            "errors_avoided": bw, "corrects_sacrificed": br,
            "fisher_p": round(fp, 4),
            "perm_p_vs_random": round(perm_p, 4),
            "aurc_gate": round(aurc, 3), "aurc_random": round(aurc_base, 3),
            "aurc_improve": round(aurc_base - aurc, 3),
        }
        print(f"\n### {label}  (N={N})")
        print(f"  base_err={base_err*100:.1f}%  -> answered_err={ans_err*100:.1f}%  "
              f"(coverage {len(answered)/N*100:.1f}%, dE={(base_err-ans_err)*100:+.1f}pp)")
        print(f"  errors_avoided={bw}  corrects_sacrificed={br}")
        print(f"  Fisher p (abstain vs answer x wrong/right) = {fp:.4f}")
        print(f"  Permutation p (gate beats RANDOM abstention) = {perm_p:.4f}")
        print(f"  AURC: gate(margin-ranked)={aurc:.3f}  random={aurc_base:.3f}  "
              f"improvement={aurc_base-aurc:+.3f}")

    adj = bh(fisher_ps)
    print(f"\nFisher p-values: {[round(p,4) for p in fisher_ps]}")
    print(f"BH-adjusted:     {[round(p,4) for p in adj]}  "
          f"(survive 0.05: {sum(1 for p in adj if p<0.05)}/3)")
    for (label, _), p in zip(BACKBONES.items(), adj):
        out[label]["fisher_p_bh"] = round(p, 4)

    # ---- dataset x resolvable confound table (backbone-averaged + Qwen3-VL) ----
    print("\n" + "=" * 72)
    print("CONFOUND TABLE: error rate by dataset x resolvable")
    print("=" * 72)
    confound = {}
    for label, fname in BACKBONES.items():
        corr = load_correct(fname)
        cell = {}
        for t in corr:
            if t not in items:
                continue
            ds = items[t]["dataset"]
            rv = items[t]["resolvable"]
            key = (ds, "resolvable" if rv else ("res-limited" if rv is False else "unknown"))
            cell.setdefault(key, []).append(corr[t])
        confound[label] = {f"{k[0]}|{k[1]}": [round((1-sum(v)/len(v))*100,1), len(v)]
                           for k, v in cell.items()}
    # print a compact view keyed by stratum
    strata = sorted({k for c in confound.values() for k in c})
    print(f"{'stratum':22s} " + "  ".join(f"{l[:10]:>12s}" for l in BACKBONES))
    for s in strata:
        row = f"{s:22s} "
        for l in BACKBONES:
            v = confound[l].get(s)
            row += f"  {f'{v[0]}%(n{v[1]})':>12s}" if v else f"  {'-':>12s}"
        print(row)

    # gate vs dataset collinearity
    abst_ds = {}
    any_corr = load_correct(BACKBONES["Qwen3-VL-8B"])
    for t in any_corr:
        if t in items and items[t]["resolvable"] is False:
            abst_ds[items[t]["dataset"]] = abst_ds.get(items[t]["dataset"], 0) + 1
    print(f"\nAbstained-set composition (gate fires on): {abst_ds}")
    print("(If dominated by one dataset, the gate ~= a dataset indicator.)")

    json.dump({"description": "Rigorous selective-prediction analysis: GSD gate vs random "
                              "abstention (permutation + Fisher/BH), margin-ranked AURC, and "
                              "dataset x resolvable confound. No confidence baseline (no signal "
                              "in stored data).",
               "per_backbone": out, "confound": confound,
               "abstained_dataset_composition": abst_ds},
              open(EXP / "triage_rigor.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\nSaved: experiments/triage_rigor.json")


if __name__ == "__main__":
    main()
