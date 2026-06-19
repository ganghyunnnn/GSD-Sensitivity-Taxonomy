"""HELD-OUT generalization test for the rule-based classifier.

The 7 pattern extensions were fit to the 19-query edge set (test_classifier_edge.py),
so 19/19 there is IN-SAMPLE. This file is an INDEPENDENT held-out set of colloquial /
unstructured M1/M2/D queries written with different vocabulary and structure, NOT used
to design any pattern. The extended classifier's recall here is the unbiased
generalization measure.
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "src")
from taxonomy.classifier import TaxonomyClassifier

clf = TaxonomyClassifier()

# Held-out: distinct phrasings from the 19-query tuning set.
heldout = [
    # M1 (spatial-metric: distance / area / length, colloquial)
    ("Roughly how far apart are the two control towers?", "M1"),
    ("Give the footprint area of the largest hangar in square metres", "M1"),
    ("What's the width of the river at its narrowest point?", "M1"),
    ("Estimate the perimeter of the cargo yard", "M1"),
    ("How many metres separate the jetty from the moored boat?", "M1"),
    ("Work out the floor area of the warehouse roof", "M1"),
    ("Approximately how tall is the smokestack?", "M1"),
    # M2 (cardinality / counting, colloquial)
    ("Can you tally the boats moored in the harbour?", "M2"),
    ("Give me a headcount of the jets on the tarmac", "M2"),
    ("Roughly how many rooftops are still intact?", "M2"),
    ("Add up the swimming pools you can spot", "M2"),
    ("Which junction looks busier with traffic?", "M2"),
    ("What's the running total of submerged vehicles?", "M2"),
    ("Note down how many silos there are", "M2"),
    # D (descriptive: must NOT be flagged M)
    ("Does this look like an industrial district?", "D"),
    ("What is the overall weather in the scene?", "D"),
    ("Point out where the main road runs", "D"),
    ("Is there any visible smoke or fire?", "D"),
    ("What colour is the largest building?", "D"),
    ("Describe the general land cover", "D"),
]

correct = 0
by_type = {"M1": [0, 0], "M2": [0, 0], "D": [0, 0]}
misses = []
for q, exp in heldout:
    res = clf.classify(q)
    pred = res.all_types[0].value if res.all_types else "D"
    ok = (pred == exp)
    correct += ok
    by_type[exp][1] += 1
    by_type[exp][0] += ok
    if not ok:
        misses.append((q, exp, pred))
    print(f"  [{'OK' if ok else 'MISS'}] [{exp}->{pred}] {q}")

n = len(heldout)
print(f"\nHeld-out accuracy: {correct}/{n} = {correct/n:.0%}")
# recall on M-types (detecting measurement queries)
m_hit = by_type['M1'][0] + by_type['M2'][0]
m_tot = by_type['M1'][1] + by_type['M2'][1]
print(f"M-type recall (held-out): {m_hit}/{m_tot} = {m_hit/m_tot:.0%}  "
      f"(M1 {by_type['M1'][0]}/{by_type['M1'][1]}, M2 {by_type['M2'][0]}/{by_type['M2'][1]})")
print(f"D precision proxy (D not over-flagged): {by_type['D'][0]}/{by_type['D'][1]}")
if misses:
    print("Misses:")
    for q, exp, pred in misses:
        print(f"  exp={exp} pred={pred}: {q}")
