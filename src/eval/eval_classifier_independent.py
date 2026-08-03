"""
C5(a): Evaluate the rule-based classifier against annotator-2's labels.

Annotator 2 labeled 88 stratified ThinkGeo tasks from question text + written
guideline, blind to Algorithm 1's output (iaa_sample.csv was distributed with an
empty type_annotator column and no algorithm-label column). This bounds
anchoring bias; it does NOT remove rule-level circularity (the guideline
transcribes the lexical rules), which must be stated in the manuscript.

Metrics (all reported, pre-specified):
  collapsed   classifier label == precedence-collapse(annotator-2 set), M1>M2>D
  lenient     classifier label in annotator-2 set
  single_only collapsed agreement restricted to single-label annotator-2 items
Plus the same metrics against the curated labels on the identical 88 tasks
(circularity-inflation estimate = curated agreement - annotator2 agreement),
Wilson 95% CIs, and Cohen's kappa (3-class, collapsed).

Output: experiments/classifier_independent_eval.json
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from taxonomy.gsd_classifier import GSDSensitivityClassifier

ROOT = Path(__file__).resolve().parent.parent.parent
ANN2 = ROOT / "annotation/iaa_annotator2.json"
LABELS = ROOT / "annotation/thinkgeo_taxonomy_labels.json"
OUT = ROOT / "experiments/classifier_independent_eval.json"

PRECEDENCE = ["M1", "M2", "D"]


def collapse(types):
    for t in PRECEDENCE:
        if t in types:
            return t
    return None


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(100 * (centre - half), 1), round(100 * (centre + half), 1))


def kappa(pairs):
    """Cohen's kappa over (pred, ref) 3-class pairs."""
    n = len(pairs)
    cats = ["D", "M1", "M2"]
    po = sum(1 for p, r in pairs if p == r) / n
    pe = sum(
        (sum(1 for p, _ in pairs if p == c) / n)
        * (sum(1 for _, r in pairs if r == c) / n)
        for c in cats
    )
    return (po - pe) / (1 - pe)


def main():
    ann2 = json.load(open(ANN2, encoding="utf-8"))
    labels = json.load(open(LABELS, encoding="utf-8"))
    clf = GSDSensitivityClassifier()

    rows = []
    for tid, a2_types in ann2.items():
        rec = labels[tid]
        pred = clf.classify(rec["query"]).task_type.value
        cur_types = rec["annotation"]["types"]
        rows.append({
            "task_id": tid,
            "query": rec["query"],
            "pred": pred,
            "ann2_set": a2_types,
            "ann2_collapsed": collapse(a2_types),
            "curated_set": cur_types,
            "curated_collapsed": collapse(cur_types),
        })

    n = len(rows)
    single = [r for r in rows if len(r["ann2_set"]) == 1]

    def agree(rows_, key_pred, key_ref, lenient=False):
        if lenient:
            k = sum(1 for r in rows_ if r[key_pred] in r[key_ref])
        else:
            k = sum(1 for r in rows_ if r[key_pred] == r[key_ref])
        return {"k": k, "n": len(rows_),
                "pct": round(100 * k / len(rows_), 1),
                "wilson95": wilson(k, len(rows_))}

    result = {
        "n_tasks": n,
        "n_single_label_ann2": len(single),
        "n_multi_label_ann2": n - len(single),
        "vs_annotator2": {
            "collapsed": agree(rows, "pred", "ann2_collapsed"),
            "lenient_in_set": agree(rows, "pred", "ann2_set", lenient=True),
            "single_label_only": agree(single, "pred", "ann2_collapsed"),
            "kappa_collapsed": round(
                kappa([(r["pred"], r["ann2_collapsed"]) for r in rows]), 3),
        },
        "vs_curated_same88": {
            "collapsed": agree(rows, "pred", "curated_collapsed"),
            "kappa_collapsed": round(
                kappa([(r["pred"], r["curated_collapsed"]) for r in rows]), 3),
        },
        "disagreements_vs_ann2": [
            {k: r[k] for k in ("task_id", "query", "pred", "ann2_set")}
            for r in rows if r["pred"] != r["ann2_collapsed"]
        ],
        "notes": [
            "annotator-2 blind to algorithm output (iaa_sample.csv had no label columns)",
            "shared author-written guideline -> bounds anchoring bias, not rule-level circularity",
            "88 tasks stratified from all 436; the 95.2% figure is on the 189-task eval split (different population)",
        ],
    }
    result["circularity_inflation_pp"] = round(
        result["vs_curated_same88"]["collapsed"]["pct"]
        - result["vs_annotator2"]["collapsed"]["pct"], 1)

    json.dump(result, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in result.items() if k != "disagreements_vs_ann2"},
                     indent=2, ensure_ascii=False))
    print("disagreements:", [r["task_id"] for r in result["disagreements_vs_ann2"]])
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
