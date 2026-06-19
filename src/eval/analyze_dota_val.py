"""
Analyze DOTA-val detector-vs-VLM counting (paired). Pre-reg: dota_val_preregistration.json.
Reports per-class + aggregate: exact/within1/MAE/RMSE/Pearson r/bias for detector vs VLM,
McNemar exact (paired exact-match), Wilcoxon signed-rank (abs-error), cluster-bootstrap CI
(by image) for MAE difference, BH across classes. Headlines detector-vs-VLM only.
"""
import io
import json
import math
import random
import sys
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent
random.seed(0)
ROWS = json.load(open(ROOT / "experiments/dota_val_counting.json", encoding="utf-8"))["rows"]
PRIMARY = ["plane", "ship", "small vehicle", "large vehicle"]


def mcnemar_exact(b, c):
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return min(1.0, p)


def wilcoxon(diffs):
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n < 1:
        return 1.0, 0
    ranks = sorted(range(n), key=lambda i: abs(d[i]))
    rank_val = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[ranks[j + 1]]) == abs(d[ranks[i]]):
            j += 1
        avg = (i + 1 + j + 1) / 2
        for t in range(i, j + 1):
            rank_val[ranks[t]] = avg
        i = j + 1
    W_plus = sum(rank_val[i] for i in range(n) if d[i] > 0)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma == 0:
        return 1.0, n
    z = (W_plus - mu) / sigma
    p = math.erfc(abs(z) / math.sqrt(2))
    return p, n


def bh(ps):
    m = len(ps); order = sorted(range(m), key=lambda i: ps[i]); adj = [0] * m; prev = 1.0
    for rank, i in enumerate(reversed(order), 1):
        k = m - rank + 1; prev = min(prev, ps[i] * m / k); adj[i] = prev
    return adj


def stats(cells, gtkey="gt"):
    det_ae = [abs(r["det"] - r[gtkey]) for r in cells]
    vlm_ae = [abs(r["vlm"] - r[gtkey]) for r in cells if r["vlm"] is not None]
    det_ex = [r["det"] == r[gtkey] for r in cells]
    vlm_ex = [r["vlm"] == r[gtkey] for r in cells if r["vlm"] is not None]
    def m(x): return sum(x) / len(x) if x else 0
    def rmse(rs): return math.sqrt(m([(r["det"] - r[gtkey]) ** 2 for r in rs]))
    def rmse_v(rs): return math.sqrt(m([(r["vlm"] - r[gtkey]) ** 2 for r in rs if r["vlm"] is not None]))
    # pearson
    def pearson(pred):
        xs = [r[gtkey] for r in cells if r["vlm"] is not None or pred == "det"]
        pr = [r[pred] for r in cells if r["vlm"] is not None or pred == "det"]
        xs = [r[gtkey] for r in cells]; pr = [r[pred] if r[pred] is not None else 0 for r in cells]
        n = len(xs); mx = sum(xs) / n; mp = sum(pr) / n
        cov = sum((a - mx) * (b - mp) for a, b in zip(xs, pr))
        sx = math.sqrt(sum((a - mx) ** 2 for a in xs)); sp = math.sqrt(sum((b - mp) ** 2 for b in pr))
        return cov / (sx * sp) if sx and sp else 0
    return {
        "n": len(cells), "n_vlm": len(vlm_ex),
        "det_exact": round(m(det_ex) * 100, 1), "vlm_exact": round(m(vlm_ex) * 100, 1),
        "det_within1": round(m([a <= 1 for a in det_ae]) * 100, 1),
        "vlm_within1": round(m([a <= 1 for a in vlm_ae]) * 100, 1),
        "det_mae": round(m(det_ae), 2), "vlm_mae": round(m(vlm_ae), 2),
        "det_rmse": round(rmse(cells), 2), "vlm_rmse": round(rmse_v(cells), 2),
        "det_bias": round(m([r["det"] - r[gtkey] for r in cells]), 2),
        "vlm_bias": round(m([r["vlm"] - r[gtkey] for r in cells if r["vlm"] is not None]), 2),
        "det_r": round(pearson("det"), 3), "vlm_r": round(pearson("vlm"), 3),
    }


def paired_tests(cells, gtkey="gt"):
    pair = [(r, r) for r in cells if r["vlm"] is not None]
    b = sum(1 for r in cells if r["vlm"] is not None and (r["det"] == r[gtkey]) and not (r["vlm"] == r[gtkey]))
    c = sum(1 for r in cells if r["vlm"] is not None and not (r["det"] == r[gtkey]) and (r["vlm"] == r[gtkey]))
    mcp = mcnemar_exact(b, c)
    diffs = [abs(r["det"] - r[gtkey]) - abs(r["vlm"] - r[gtkey]) for r in cells if r["vlm"] is not None]
    wp, wn = wilcoxon(diffs)
    # cluster bootstrap by image for MAE difference
    byimg = {}
    for r in cells:
        if r["vlm"] is not None:
            byimg.setdefault(r["image"], []).append(abs(r["det"] - r[gtkey]) - abs(r["vlm"] - r[gtkey]))
    keys = list(byimg)
    boots = []
    for _ in range(10000):
        s = [random.choice(keys) for _ in keys]
        vals = [d for k in s for d in byimg[k]]
        boots.append(sum(vals) / len(vals))
    boots.sort()
    ci = (round(boots[250], 3), round(boots[9750], 3))
    return {"mcnemar_b_det_only": b, "mcnemar_c_vlm_only": c, "mcnemar_p": round(mcp, 6),
            "wilcoxon_p": round(wp, 6), "mae_diff_mean": round(sum(diffs) / len(diffs), 3),
            "mae_diff_ci95": ci}


def report(gtkey):
    print(f"\n{'='*70}\nGT = {gtkey}\n{'='*70}")
    ps = []
    perclass = {}
    for c in PRIMARY:
        cells = [r for r in ROWS if r["cls"] == c]
        nz = [r for r in cells if r[gtkey] > 0]
        s = stats(cells, gtkey); t = paired_tests(cells, gtkey)
        perclass[c] = {**s, **t}
        ps.append(t["mcnemar_p"])
        print(f"\n[{c}] n={s['n']} (nonzero={len(nz)})")
        print(f"  exact:  det={s['det_exact']}%  vlm={s['vlm_exact']}%   |  within1: det={s['det_within1']}% vlm={s['vlm_within1']}%")
        print(f"  MAE:    det={s['det_mae']}  vlm={s['vlm_mae']}   RMSE: det={s['det_rmse']} vlm={s['vlm_rmse']}")
        print(f"  bias:   det={s['det_bias']}  vlm={s['vlm_bias']}   r: det={s['det_r']} vlm={s['vlm_r']}")
        print(f"  McNemar b(det-only)={t['mcnemar_b_det_only']} c(vlm-only)={t['mcnemar_c_vlm_only']} p={t['mcnemar_p']}")
        print(f"  Wilcoxon(AE) p={t['wilcoxon_p']}  MAE diff(det-vlm)={t['mae_diff_mean']} CI95={t['mae_diff_ci95']}")
    adj = bh(ps)
    for c, a in zip(PRIMARY, adj):
        perclass[c]["mcnemar_p_bh"] = round(a, 6)
    # aggregate
    allc = [r for r in ROWS]
    s = stats(allc, gtkey); t = paired_tests(allc, gtkey)
    print(f"\n[AGGREGATE] n={s['n']}")
    print(f"  exact: det={s['det_exact']}% vlm={s['vlm_exact']}%  within1: det={s['det_within1']}% vlm={s['vlm_within1']}%")
    print(f"  MAE: det={s['det_mae']} vlm={s['vlm_mae']}  RMSE: det={s['det_rmse']} vlm={s['vlm_rmse']}")
    print(f"  McNemar p={t['mcnemar_p']} (b={t['mcnemar_b_det_only']},c={t['mcnemar_c_vlm_only']})  "
          f"Wilcoxon p={t['wilcoxon_p']}  MAE diff CI95={t['mae_diff_ci95']}")
    print(f"  BH-adj McNemar per class: {[perclass[c]['mcnemar_p_bh'] for c in PRIMARY]}")
    return {"per_class": perclass, "aggregate": {**s, **t}}


def main():
    n_imgs = len(set(r["image"] for r in ROWS))
    print(f"DOTA-val counting analysis: {len(ROWS)} cells, {n_imgs} images")
    out = {"gt_strict": report("gt"), "gt_resolvable": report("gt_resolv")}
    json.dump(out, open(ROOT / "experiments/dota_val_analysis.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("\nSaved: experiments/dota_val_analysis.json")


if __name__ == "__main__":
    main()
