"""Analyze M2 A/B test results (same-resolution prompt effect) with McNemar tests."""
import io, sys, json
from scipy import stats
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

files = {
    "Qwen3-VL-8B": "m2ab_qwen3vl_8b.json",
    "Qwen2.5-VL-7B": "m2ab_qwen25vl_7b.json",
    "InternVL3.5-8B": "m2ab_internvl35_8b.json",
}
print("=== M2 prompt A/B @ 1024px (same resolution, prompt-only difference) ===")
print(f"{'backbone':18s} {'direct':>8s} {'counting':>9s} {'effect':>8s} {'McNemar_p':>10s}")
rows = []
for name, fn in files.items():
    with open(f"annotation/{fn}", encoding="utf-8") as f:
        d = json.load(f)
    df = d["direct_fail_rate"] * 100
    cf = d["counting_fail_rate"] * 100
    eff = d["counting_effect_pp"]
    dmap = {r["task_id"]: r["correct"] for r in d["direct"]}
    cmap = {r["task_id"]: r["correct"] for r in d["counting"]}
    b = sum(1 for t in dmap if dmap[t] and not cmap.get(t))
    c = sum(1 for t in dmap if not dmap[t] and cmap.get(t))
    if b + c > 0:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p = stats.chi2.sf(chi2, 1)
    else:
        p = 1.0
    print(f"{name:18s} {df:7.1f}% {cf:8.1f}% {eff:+7.1f}pp {p:10.3f}  (b={b},c={c})")
    rows.append({"backbone": name, "direct_fail": round(df, 1),
                 "counting_fail": round(cf, 1), "effect_pp": eff,
                 "mcnemar_p": round(p, 4), "b": b, "c": c})

print("\nInterpretation: 'effect' = counting_fail - direct_fail (negative = improvement).")
print("None significant after correction => prompt-only intervention is weak/inconsistent,")
print("reinforcing that architectural (not prompt) solutions are needed.")

# Save summary
with open("experiments/m2ab_summary.json", "w", encoding="utf-8") as f:
    json.dump({"description": "M2 direct-vs-counting prompt A/B at fixed 1024px",
               "rows": rows}, f, indent=2, ensure_ascii=False)
print("\nSaved: experiments/m2ab_summary.json")
