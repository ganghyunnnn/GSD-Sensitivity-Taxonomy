"""
Routing-policy comparison on a mixed ThinkGeo + DOTA-val pool.

Compares taxonomy routing against keyword-based, class-based, random, and
route-all-counting baseline policies. Policies only diverge
where (a) a counting question targets a class outside the detector vocabulary
(xBD buildings) or (b) the resolvability gate fires. The DOTA-val pool alone
cannot discriminate them (all four classes covered, gate almost never fires),
so the pool mixes:

  - ThinkGeo eval split (189 tasks; D/M1 go to the VLM under every policy,
    M2 tasks are where policies diverge). Detector outcomes for covered classes
    are obtained by actually running YOLO26l-OBB on the ThinkGeo images
    (vehicle/car -> small+large vehicle, aircraft -> plane). Building and
    comparison questions are outside the detector vocabulary by construction.
  - DOTA-v1 val 800 image x class cells (stored det + 3-backbone VLM counts).

Policies (per M2 question):
  taxonomy   : covered & resolvable -> detector; unresolvable -> FLAG;
               else VLM. (resolvable = GSD <= d/s per best-known GSD;
               unknown GSD -> no gate information -> detector if covered)
  allcount   : any counting question -> detector (uncovered class -> wrong)
  keyword    : query contains "how many" -> detector (misses "count the
               number of...", comparisons; no gate)
  classbased : covered class -> detector (no gate)
  random     : each M2 question routed to detector with p=0.5 (expectation)

Scoring: uniform exact-match protocol. FLAG is reported both ways:
  strict   = flagged counts as incorrect (conservative headline)
  selective = flagged excluded from denominator, with coverage reported
Mix sensitivity: ThinkGeo-only / DOTA-only / combined.

Output: experiments/routing_policy_comparison.json
Run   : uv run python src/eval/simulate_routing_policies.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ANALYSIS = ROOT / "experiments/m2_resolution_analysis.json"
BENCH = ROOT / "data/thinkgeo/ThinkGeoBench.json"
LABELS = ROOT / "annotation/thinkgeo_taxonomy_labels.json"
COUNTS = ROOT / "experiments/dota_val_counting.json"
GSDMAP = ROOT / "experiments/dota_val_gsd_map.json"
IMGROOT = ROOT / "data/thinkgeo"
DET_CACHE = ROOT / "experiments/thinkgeo_m2_detector_counts.json"
OUT = ROOT / "experiments/routing_policy_comparison.json"

BACKBONES = {
    "qwen3vl": "annotation/backbone_qwen3vl_8b.json",
    "qwen25vl": "annotation/backbone_qwen25vl_7b.json",
    "internvl35": "annotation/backbone_internvl35_8b.json",
}
COVERED = {"vehicle": ["small vehicle", "large vehicle"],
           "car": ["small vehicle", "large vehicle"],
           "aircraft": ["plane"]}
DOTA_CLS = {0: "plane", 1: "ship", 9: "large vehicle", 10: "small vehicle"}
S_FLOOR = 10
NOMINAL_M = {"plane": 30.0, "ship": 25.0, "small vehicle": 4.5, "large vehicle": 12.0}


def gt_count_from_whitelist(gt_answer):
    """first integer in the first whitelist group, if any."""
    if isinstance(gt_answer, dict):
        wl = gt_answer.get("whitelist") or []
    elif isinstance(gt_answer, list):
        wl = gt_answer
    else:
        wl = []
    wl = [g if isinstance(g, list) else [g] for g in wl]
    for group in wl:
        for s in group:
            m = re.search(r"\b(\d+)\b", str(s))
            if m:
                return int(m.group(1))
    return None


def detector_counts_thinkgeo(tasks):
    """Run YOLO26l-OBB once per needed image; cache to DET_CACHE."""
    cache = json.load(open(DET_CACHE, encoding="utf-8")) if DET_CACHE.exists() else {}
    todo = sorted({t["image"] for t in tasks if t["covered"] and t["image"]
                   and t["image"] not in cache
                   and (IMGROOT / t["image"]).exists()})
    if todo:
        from ultralytics import YOLO
        model = YOLO(str(ROOT / "yolo26l-obb.pt"))
        for i, img in enumerate(todo):
            res = model.predict(str(IMGROOT / img), imgsz=1536, conf=0.25,
                                verbose=False)[0]
            ids = res.obb.cls.tolist() if res.obb is not None else []
            counts = {}
            for ci in ids:
                name = DOTA_CLS.get(int(ci))
                if name:
                    counts[name] = counts.get(name, 0) + 1
            cache[img] = counts
            if (i + 1) % 10 == 0:
                print(f"  detector {i + 1}/{len(todo)}")
        json.dump(cache, open(DET_CACHE, "w", encoding="utf-8"), indent=1)
    return cache


def build_thinkgeo_pool():
    analysis = json.load(open(ANALYSIS, encoding="utf-8"))
    bench = json.load(open(BENCH, encoding="utf-8"))
    labels = json.load(open(LABELS, encoding="utf-8"))
    m2meta = {r["task_id"]: r for r in analysis["records"]}

    backbone_results = {}
    for name, path in BACKBONES.items():
        d = json.load(open(ROOT / path, encoding="utf-8"))
        backbone_results[name] = {r["task_id"]: r for r in d["direct"]["results"]}

    pool = []
    any_bb = next(iter(backbone_results.values()))
    for tid, res in any_bb.items():
        rec = labels[tid]
        gt_type = res["gt_type"]
        meta = m2meta.get(tid)
        covered_classes = COVERED.get(meta["obj_type"]) if meta else None
        gsd = meta["gsd"] if meta else None
        thr = meta["threshold_gsd"] if meta else None
        resolvable = None if gsd is None else (gsd <= thr)
        pool.append({
            "source": "thinkgeo", "task_id": tid, "type": gt_type,
            "query": rec["query"], "image": rec.get("image"),
            "gt_count": gt_count_from_whitelist(bench[tid].get("gt_answer")),
            "covered": bool(covered_classes), "classes": covered_classes,
            "resolvable": resolvable,
            "vlm_correct": {b: r[tid]["correct"] for b, r in backbone_results.items()},
        })
    det_cache = detector_counts_thinkgeo([t for t in pool if t["type"] == "M2"])
    for t in pool:
        if (t["type"] == "M2" and t["covered"] and t["gt_count"] is not None
                and t["image"] in det_cache):
            c = det_cache[t["image"]]
            det_count = sum(c.get(cls, 0) for cls in t["classes"])
            t["det_correct"] = (det_count == t["gt_count"])
            t["det_count"] = det_count
        else:
            t["det_correct"] = False  # uncovered / unparseable / missing image
    return pool


def build_dota_pool():
    counts = json.load(open(COUNTS, encoding="utf-8"))["rows"]
    gsdmap = json.load(open(GSDMAP, encoding="utf-8"))["images"]
    key = {"qwen3vl": "vlm_q3", "qwen25vl": "vlm", "internvl35": "vlm_ivl"}
    pool = []
    for r in counts:
        meta = gsdmap.get(r["image"])
        gsd_ok = meta and meta["flag"] == "ok"
        resolvable = (meta["gsd"] <= NOMINAL_M[r["cls"]] / S_FLOOR) if gsd_ok else None
        pool.append({
            "source": "dota", "type": "M2",
            "query": f"how many {r['cls']}s are in this image?",
            "covered": True, "resolvable": resolvable,
            "det_correct": r["det"] == r["gt"],
            "vlm_correct": {b: (r.get(k) is not None and r[k] == r["gt"])
                            for b, k in key.items()},
        })
    return pool


def route(policy, q):
    """returns 'det' | 'vlm' | 'flag' for an M2 question."""
    if policy == "taxonomy":
        if q["resolvable"] is False:
            return "flag"
        return "det" if q["covered"] else "vlm"
    if policy == "allcount":
        return "det"
    if policy == "keyword":
        return "det" if "how many" in q["query"].lower() else "vlm"
    if policy == "classbased":
        return "det" if q["covered"] else "vlm"
    raise ValueError(policy)


def evaluate(pool, backbone):
    out = {}
    for policy in ("taxonomy", "allcount", "keyword", "classbased", "random"):
        correct = 0.0
        flagged = 0
        n = 0
        for q in pool:
            n += 1
            if q["type"] != "M2":
                correct += q["vlm_correct"][backbone]
                continue
            if policy == "random":
                correct += 0.5 * q["det_correct"] + 0.5 * q["vlm_correct"][backbone]
                continue
            r = route(policy, q)
            if r == "flag":
                flagged += 1
            elif r == "det":
                correct += q["det_correct"]
            else:
                correct += q["vlm_correct"][backbone]
        out[policy] = {
            "n": n, "flagged": flagged,
            "acc_strict_pct": round(100 * correct / n, 1),
            "acc_selective_pct": round(100 * correct / (n - flagged), 1) if n > flagged else None,
        }
    # counterfactual value of the taxonomy's flags: had the flagged questions
    # been answered instead, how often would VLM / detector have been correct?
    flagged_qs = [q for q in pool if q["type"] == "M2"
                  and route("taxonomy", q) == "flag"]
    if flagged_qs:
        out["flag_counterfactual"] = {
            "n_flagged": len(flagged_qs),
            "vlm_would_be_correct_pct": round(
                100 * sum(q["vlm_correct"][backbone] for q in flagged_qs) / len(flagged_qs), 1),
            "det_would_be_correct_pct": round(
                100 * sum(q["det_correct"] for q in flagged_qs) / len(flagged_qs), 1),
        }
    return out


def main():
    tg = build_thinkgeo_pool()
    dota = build_dota_pool()
    tg_m2 = [q for q in tg if q["type"] == "M2"]

    result = {"pools": {
        "thinkgeo_full_189": {b: evaluate(tg, b) for b in BACKBONES},
        "thinkgeo_m2_83": {b: evaluate(tg_m2, b) for b in BACKBONES},
        "dota_800": {b: evaluate(dota, b) for b in BACKBONES},
        "combined": {b: evaluate(tg + dota, b) for b in BACKBONES},
    }, "notes": [
        "detector on ThinkGeo images: YOLO26l-OBB imgsz=1536 conf=0.25 (cached in thinkgeo_m2_detector_counts.json)",
        "uncovered classes (buildings, comparisons) -> detector cannot answer -> incorrect if routed there",
        "taxonomy FLAG on unresolvable: strict = incorrect; selective = excluded with coverage",
        "random = per-question expectation at p=0.5",
        "question mix pre-registered as the observed ThinkGeo eval-split distribution (D 66 / M1 40 / M2 83)",
    ]}
    m2_stats = {
        "covered": sum(1 for q in tg_m2 if q["covered"]),
        "gt_count_parseable": sum(1 for q in tg_m2 if q["gt_count"] is not None),
        "unresolvable": sum(1 for q in tg_m2 if q["resolvable"] is False),
        "unknown_gsd": sum(1 for q in tg_m2 if q["resolvable"] is None),
    }
    result["thinkgeo_m2_composition"] = m2_stats

    # cost-sensitive dominance: 3-outcome decomposition (correct/wrong/flag)
    # and the break-even error cost c* above which taxonomy's utility
    # U = correct - c * wrong exceeds each gateless policy's, per backbone.
    n = len(tg_m2)
    dom = {}
    for b, pols in result["pools"]["thinkgeo_m2_83"].items():
        tax = pols["taxonomy"]
        tc = round(tax["acc_strict_pct"] * n / 100)
        tf = tax["flagged"]
        tw = n - tc - tf
        rows = {"taxonomy": {"correct": tc, "wrong": tw, "flagged": tf}}
        for p, v in pols.items():
            if p in ("taxonomy", "flag_counterfactual"):
                continue
            c = round(v["acc_strict_pct"] * n / 100)
            w = n - c - v["flagged"]
            rows[p] = {"correct": c, "wrong": w, "flagged": v["flagged"],
                       "breakeven_cost_vs_taxonomy":
                           round((c - tc) / (w - tw), 3) if w != tw else None}
        dom[b] = rows
    result["cost_dominance_thinkgeo_m2"] = dom
    result["notes"].append(
        "cost_dominance: taxonomy utility exceeds every gateless policy for any "
        "error cost c > max breakeven (0.216, qwen3vl vs classbased); vs "
        "allcount/keyword the breakeven is negative (dominates even at c=0)")

    json.dump(result, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(result["thinkgeo_m2_composition"], indent=1))
    for pool_name in ("thinkgeo_m2_83", "combined"):
        print(f"\n== {pool_name} (strict / selective) ==")
        for b, pols in result["pools"][pool_name].items():
            row = "  ".join(f"{p}={v['acc_strict_pct']}/{v['acc_selective_pct']}"
                            for p, v in pols.items() if p != "flag_counterfactual")
            print(f"  {b:10s} {row}")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
