"""Check existing multi-backbone data availability for resubmission planning."""
import io, sys, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

with open("annotation/gsd_ablation.json", encoding="utf-8") as f:
    abl = json.load(f)
print("=== gsd_ablation models & conditions ===")
for m in abl.keys():
    conds = {k: len(v) for k, v in abl[m].items()}
    print(f"  {m}: {conds}")

print("\n=== backbone result files ===")
for fn in sorted(os.listdir("annotation")):
    if fn.startswith("backbone_"):
        with open(f"annotation/{fn}", encoding="utf-8") as f:
            d = json.load(f)
        pt = d["direct"]["per_type"]
        print(f"  {fn}: D={pt['D']['total']} M1={pt['M1']['total']} M2={pt['M2']['total']}")

# eval split M1 IDs per backbone for routed prompting
print("\n=== Routed prompting feasibility ===")
print("For each backbone, need: D(baseline reuse) + M1(ablation few_shot reuse) + M2(NEW inference)")
for fn in sorted(os.listdir("annotation")):
    if fn.startswith("backbone_"):
        with open(f"annotation/{fn}", encoding="utf-8") as f:
            d = json.load(f)
        m2_n = d["direct"]["per_type"]["M2"]["total"]
        print(f"  {fn}: M2 inference needed = {m2_n} tasks")
