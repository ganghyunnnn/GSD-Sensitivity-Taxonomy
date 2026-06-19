"""
Consolidate matched-resolution taxonomy-routed prompting evidence (confound-free).

The original Table 5 compared full-resolution direct baselines against 1024px
routed M2 inference, confounding the prompt effect with image resolution. This
script rebuilds the routed-prompting evidence with ALL conditions at matched
resolution, per backbone, with McNemar tests and a single BH correction across
the whole family.

Sources (all matched-resolution, paired):
  M1: annotation/gsd_ablation.json   baseline vs few_shot  (N=161, identical res)
  M2: annotation/m2ab_*.json         direct   vs counting  (N=83, 1024px)
  M2: annotation/m2search_*.json     direct   vs grid/cot_count[/systematic] (N=83, 1024px)

Outputs:
  experiments/routed_matched_summary.json
  experiments/routed_matched_summary.md
"""
import io
import json
import sys
from pathlib import Path

from scipy import stats

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent
ANN = ROOT / "annotation"

BACKBONES = {
    "qwen3-vl_8b":          {"label": "Qwen3-VL-8B",   "m2ab": "m2ab_qwen3vl_8b.json",
                             "search": "m2search_qwen3vl_8b.json"},
    "qwen2.5vl_7b":         {"label": "Qwen2.5-VL-7B", "m2ab": "m2ab_qwen25vl_7b.json",
                             "search": "m2search_qwen25vl_7b.json"},
    "blaifa_InternVL3_5_8b":{"label": "InternVL3.5-8B","m2ab": "m2ab_internvl35_8b.json",
                             "search": "m2search_internvl35_8b.json"},
}


def mcnemar(base_map, alt_map):
    """Paired exact-ish McNemar (continuity-corrected chi-square).
    Returns (base_fail%, alt_fail%, effect_pp, p, b, c). effect<0 = alt improves."""
    ids = [t for t in base_map if t in alt_map]
    bf = (1 - sum(base_map[t] for t in ids) / len(ids)) * 100
    af = (1 - sum(alt_map[t] for t in ids) / len(ids)) * 100
    b = sum(1 for t in ids if base_map[t] and not alt_map[t])      # base ok, alt wrong
    c = sum(1 for t in ids if not base_map[t] and alt_map[t])      # base wrong, alt ok
    if b + c > 0:
        chi = (abs(b - c) - 1) ** 2 / (b + c)
        p = float(stats.chi2.sf(chi, 1))
    else:
        p = 1.0
    return round(bf, 1), round(af, 1), round(af - bf, 1), round(p, 4), b, c, len(ids)


def load(fn):
    p = ANN / fn
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    rows = []  # each: backbone,type,prompt,base_fail,alt_fail,effect,p,b,c,N

    # --- M1: ablation baseline vs few_shot ---
    abl = load("gsd_ablation.json")
    for key, meta in BACKBONES.items():
        if not abl or key not in abl:
            continue
        bm = {r["task_id"]: r["correct"] for r in abl[key]["baseline"]}
        fm = {r["task_id"]: r["correct"] for r in abl[key]["few_shot"]}
        bf, af, eff, p, b, c, n = mcnemar(bm, fm)
        rows.append({"backbone": meta["label"], "type": "M1", "prompt": "few_shot",
                     "base_fail": bf, "alt_fail": af, "effect_pp": eff,
                     "p": p, "b": b, "c": c, "N": n})

    # --- M2: A/B direct vs counting ---
    for key, meta in BACKBONES.items():
        ab = load(meta["m2ab"])
        if not ab:
            continue
        dm = {r["task_id"]: r["correct"] for r in ab["direct"]}
        cm = {r["task_id"]: r["correct"] for r in ab["counting"]}
        bf, af, eff, p, b, c, n = mcnemar(dm, cm)
        rows.append({"backbone": meta["label"], "type": "M2", "prompt": "counting",
                     "base_fail": bf, "alt_fail": af, "effect_pp": eff,
                     "p": p, "b": b, "c": c, "N": n})

    # --- M2: prompt search direct vs grid/cot_count/systematic ---
    for key, meta in BACKBONES.items():
        sr = load(meta["search"])
        if not sr:
            continue
        conds = sr["conditions"]
        if "direct" not in conds:
            continue
        dm = {r["task_id"]: r["correct"] for r in conds["direct"]}
        for alt in ("grid", "cot_count", "systematic"):
            if alt not in conds or not conds[alt]:
                continue
            am = {r["task_id"]: r["correct"] for r in conds[alt]}
            # only compare on shared completed ids
            if len(am) < len(dm):
                shared = {t: dm[t] for t in dm if t in am}
                dm_use = shared
            else:
                dm_use = dm
            bf, af, eff, p, b, c, n = mcnemar(dm_use, am)
            rows.append({"backbone": meta["label"], "type": "M2", "prompt": alt,
                         "base_fail": bf, "alt_fail": af, "effect_pp": eff,
                         "p": p, "b": b, "c": c, "N": n})

    # --- BH correction across the whole routed-prompting family ---
    ps = sorted((r["p"], i) for i, r in enumerate(rows))
    m = len(ps)
    bh = {}
    for rank, (p, i) in enumerate(ps, start=1):
        bh[i] = p * m / rank
    # enforce monotonicity
    running = 1.0
    for rank in range(m, 0, -1):
        i = ps[rank - 1][1]
        running = min(running, bh[i])
        bh[i] = round(min(running, 1.0), 4)
    for i, r in enumerate(rows):
        r["p_bh"] = bh.get(i, None)
        r["sig_bh"] = (r["p_bh"] is not None and r["p_bh"] < 0.05)

    # --- report ---
    print(f"=== Matched-resolution taxonomy-routed prompting ({m} paired McNemar tests) ===")
    hdr = f"{'backbone':15s} {'type':4s} {'prompt':11s} {'base':>6s} {'alt':>6s} {'eff':>7s} {'p':>7s} {'p_BH':>7s} {'N':>4s}"
    print(hdr)
    for r in rows:
        star = " *" if r["sig_bh"] else ""
        print(f"{r['backbone']:15s} {r['type']:4s} {r['prompt']:11s} "
              f"{r['base_fail']:5.1f}% {r['alt_fail']:5.1f}% {r['effect_pp']:+6.1f}pp "
              f"{r['p']:7.3f} {r['p_bh']:7.3f} {r['N']:4d}{star}")

    n_sig = sum(1 for r in rows if r["sig_bh"])
    n_improve = sum(1 for r in rows if r["effect_pp"] < 0)
    print(f"\n{n_improve}/{m} tests show nominal improvement; {n_sig}/{m} survive BH (alpha=0.05).")
    print("'*' = survives BH. effect<0 = routed prompt improves over direct.")

    out = {"description": "Matched-resolution taxonomy-routed prompting; "
                          "M1 from ablation (baseline vs few_shot, N=161), "
                          "M2 from A/B + prompt search (direct vs counting/grid/cot_count, N=83); "
                          "all conditions paired at identical resolution; single BH correction.",
           "n_tests": m, "n_nominal_improve": n_improve, "n_survive_bh": n_sig,
           "rows": rows}
    (ROOT / "experiments").mkdir(exist_ok=True)
    with open(ROOT / "experiments/routed_matched_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # markdown
    ml = ["# Matched-resolution taxonomy-routed prompting", "",
          f"{m} paired McNemar tests; {n_improve} nominal improvements; {n_sig} survive BH.", "",
          "| Backbone | Type | Routed prompt | Direct fail | Routed fail | Effect | p | p(BH) | N |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        star = " ✓BH" if r["sig_bh"] else ""
        ml.append(f"| {r['backbone']} | {r['type']} | {r['prompt']} | {r['base_fail']}% | "
                  f"{r['alt_fail']}% | {r['effect_pp']:+.1f}pp | {r['p']:.3f} | {r['p_bh']:.3f}{star} | {r['N']} |")
    with open(ROOT / "experiments/routed_matched_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(ml) + "\n")
    print("\nSaved: experiments/routed_matched_summary.json, .md")


if __name__ == "__main__":
    main()
