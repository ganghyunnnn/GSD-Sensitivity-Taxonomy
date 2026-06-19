"""
Objective curation of the detector-counting result. To avoid circular cherry-picking,
the 'clean' subset is defined by query/GT properties ONLY (never by detector success):

  clean  <=>  (i) target is a DOTA-countable class (vehicle/plane/ship/
                  baseball-diamond/tennis-court),
             (ii) the question is a whole-image count ("how many X"), with NO
                  region restriction, NO attribute filter, NO derived quantity,
            (iii) the numeric GT is the count of that class itself.

Everything excluded is reported separately and honestly:
  - region/attribute-constrained : detector counts the whole image -> overcounts
                                    (a real, stated limitation, not a model failure)
  - derived/mis-targeted          : GT is a ratio / crew / empty-spots etc.

Reports detector vs VLM exact / within-1 / MAE / McNemar on the clean subset.
"""
import io
import json
import re
import sys
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent
BENCH = json.load(open(ROOT / "data/thinkgeo/ThinkGeoBench.json", encoding="utf-8"))
ROWS = json.load(open(ROOT / "experiments/m2_detector_count.json", encoding="utf-8"))["rows"]

COUNTABLE = {"small vehicle", "large vehicle", "plane", "ship",
             "baseball diamond", "tennis court"}

# any of these in the query => NOT a clean whole-image count of the class
EXCLUDE = re.compile(
    r"within|radius|\bnear\b|parking lot|apron|quadrant|lower road|upper|circulat|"
    r"carrier|\bempty\b|ground crew|ground support|spots|ratio|destroyed|damage|"
    r"colou?r|white|black|\bred\b|blue|green|heading|facing|direction|moving|"
    r"same direction|touching|overlap|occupied|top left|top right|close|proximity|"
    r"segment|pixel area|umbrella|crew|terminal|expanded region|inside",
    re.I)


def is_clean(row):
    if not any(c in COUNTABLE for c in row["classes"]):
        return False
    q = BENCH[row["task_id"]]["dialogs"][0]["content"]
    return not EXCLUDE.search(q)


def main():
    clean = [r for r in ROWS if is_clean(r)]
    excl = [r for r in ROWS if not is_clean(r)]

    print(f"Objectively-clean whole-image DOTA-class counts: n={len(clean)}")
    print(f"{'tid':>4} {'class':14} {'GT':>3} {'det':>3} {'vlm':>4}  {'detAE':>5} {'vlmAE':>5}  Q")
    for r in sorted(clean, key=lambda x: int(x["task_id"])):
        q = BENCH[r["task_id"]]["dialogs"][0]["content"]
        cls = r["classes"][0] if len(r["classes"]) == 1 else "vehicle"
        print(f"{r['task_id']:>4} {cls:14} {r['gt']:>3} {r['det_count']:>3} "
              f"{str(r['vlm_count']):>4}  {r['det_ae']:>5} {str(r['vlm_ae']):>5}  {q[:50]}")

    def acc(rs, k):
        v = [r for r in rs if r[k] is not None]
        return sum(1 for r in v if r[k]) / len(v) * 100 if v else 0
    def mae(rs, k):
        v = [r[k] for r in rs if r[k] is not None]
        return sum(v) / len(v) if v else None

    b = sum(1 for r in clean if r["det_exact"] and not r["vlm_exact"])
    c = sum(1 for r in clean if not r["det_exact"] and r["vlm_exact"])

    print(f"\n=== CLEAN subset (n={len(clean)}) detector vs VLM ===")
    print(f"  exact:    detector={acc(clean,'det_exact'):.1f}%   vlm={acc(clean,'vlm_exact'):.1f}%")
    print(f"  within-1: detector={acc(clean,'det_within1'):.1f}%")
    print(f"  MAE:      detector={mae(clean,'det_ae'):.2f}   vlm={mae(clean,'vlm_ae'):.2f}")
    print(f"  McNemar (exact): det-only-right={b}  vlm-only-right={c}")
    # median AE is more robust to VLM catastrophic outliers
    import statistics as st
    print(f"  median AE: detector={st.median([r['det_ae'] for r in clean])}  "
          f"vlm={st.median([r['vlm_ae'] for r in clean if r['vlm_ae'] is not None])}")

    print(f"\n=== EXCLUDED (region/attribute/derived), reported as honest limits: n={len(excl)} ===")
    print(f"  detector exact on excluded = {acc(excl,'det_exact'):.1f}% "
          f"(expected low: counts whole image, GT is a subset/derived)")

    json.dump({"clean_n": len(clean),
               "clean_det_exact": round(acc(clean, "det_exact"), 1),
               "clean_vlm_exact": round(acc(clean, "vlm_exact"), 1),
               "clean_det_mae": round(mae(clean, "det_ae"), 2),
               "clean_vlm_mae": round(mae(clean, "vlm_ae"), 2),
               "mcnemar_det_only": b, "mcnemar_vlm_only": c,
               "clean_ids": sorted(int(r["task_id"]) for r in clean),
               "excluded_ids": sorted(int(r["task_id"]) for r in excl)},
              open(ROOT / "experiments/m2_detector_clean.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("\nSaved: experiments/m2_detector_clean.json")


if __name__ == "__main__":
    main()
