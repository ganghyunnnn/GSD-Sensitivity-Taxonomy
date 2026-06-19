"""Classifier edge-case accuracy test after pattern extension."""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "src")
from taxonomy.classifier import TaxonomyClassifier

clf = TaxonomyClassifier()

cases = [
    # Standard cases (should all pass)
    ("How many cars are in the image?", "M2"),
    ("Count the buildings in this area", "M2"),
    ("What is the distance between A and B? GSD=0.3", "M1"),
    ("What percentage of structures are flooded?", "M2"),
    ("Enumerate all flooded buildings", "M2"),
    # Extended colloquial M2
    ("Identify all flooded structures and give the total", "M2"),
    ("Which parking lot is busier?", "M2"),
    ("List every damaged car you can see", "M2"),
    ("Tell me the number of vehicles visible", "M2"),
    ("Count up all the damaged houses", "M2"),
    # Extended colloquial M1
    ("Measure the runway length", "M1"),
    ("What is the span of the bridge?", "M1"),
    ("How many meters is the road?", "M1"),
    ("How long is the runway?", "M1"),
    ("How wide is this road?", "M1"),
    # D cases (should not be misclassified)
    ("How crowded is the parking area?", "D"),
    ("Is the road flooded?", "D"),
    ("What type of building is on the left?", "D"),
    ("Describe the scene", "D"),
]

correct = 0
misses = []
for q, exp in cases:
    res = clf.classify(q)
    pred = res.all_types[0].value if res.all_types else "D"
    ok = (pred == exp)
    if ok:
        correct += 1
    else:
        misses.append((q, exp, pred))
    status = "OK" if ok else "MISS"
    print(f"  [{status}] [{exp}->{pred}] {q}")

print(f"\nAccuracy: {correct}/{len(cases)} = {correct/len(cases):.0%}")
if misses:
    print("Remaining misses:")
    for q, exp, pred in misses:
        print(f"  exp={exp} pred={pred}: {q}")
