"""
Two-axis recomputation of the ThinkGeo M2 failure decomposition.

Pre-committed framing (decided before seeing the numbers):

  Axis 1 SENSOR-LIMITED : native object extent d/GSD < s px. A physical
    property of the acquisition; not fixable by engineering -> justifies
    resolution-limit flagging / deferral.
  Axis 2 PIPELINE-LIMITED : native extent >= s but effective extent after the
    model's input downsampling < s. A deployment property; fixable by tiling,
    higher input budgets, or a native-resolution tool (= the detector route).
  Else MODEL-ERROR : object resolvable at the model's input; miscount is a
    genuine model failure.

Effective extent at budget B: eff_px = (d/GSD) * min(1, B / image_long_side).
Reported across B in {448, 672, 896, 1024, 1280, native} because the Ollama
API does not expose per-model effective input resolution.

Also addressed here:
  - corrected DOTA GSD prior: the previous run used 0.135 m/px labeled "DOTA
    median"; the actual DOTA-v1 val metadata median is 0.253 (0.132 = p25).
    dataset_prior_dota records are re-evaluated under {p25, median, p75}.
  - measured object sizes: nominal sizes are validated against measured
    per-instance physical extents from DOTA-v1 val annotations (px * GSD).

Inputs : experiments/m2_resolution_analysis.json (63 failure records)
         annotation/thinkgeo_taxonomy_labels.json (image paths)
         experiments/dota_val_gsd_map.json (GSD stats + measured instances)
         data/thinkgeo/image/*.jpg (native dimensions)
Output : experiments/m2_two_axis_decomposition.json
Run    : uv run python src/eval/two_axis_decomposition.py
"""
import json
import math
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
ANALYSIS = ROOT / "experiments/m2_resolution_analysis.json"
LABELS = ROOT / "annotation/thinkgeo_taxonomy_labels.json"
GSDMAP = ROOT / "experiments/dota_val_gsd_map.json"
IMGROOT = ROOT / "data/thinkgeo"
OUT = ROOT / "experiments/m2_two_axis_decomposition.json"

S_FLOOR = 10
BUDGETS = [448, 672, 896, 1024, 1280, None]  # None = native (no resize)
DOTA_PRIOR_SCENARIOS = {"p25_0.132": 0.132, "median_0.253": 0.253, "p75_0.522": 0.522,
                        "legacy_0.135": 0.135}


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(100 * (centre - half), 1), round(100 * (centre + half), 1)]


def classify(native_px, eff_px, s=S_FLOOR):
    if native_px < s:
        return "sensor_limited"
    if eff_px < s:
        return "pipeline_limited"
    return "model_error"


def decompose(records, budget, s=S_FLOOR, dota_prior=None):
    """records: failures with known gsd (or dota prior substitution)."""
    out = {"sensor_limited": 0, "pipeline_limited": 0, "model_error": 0}
    for r in records:
        gsd = r["gsd"]
        if dota_prior is not None and r["gsd_source"] == "dataset_prior_dota":
            gsd = dota_prior
        native = r["obj_size_m"] / gsd
        scale = 1.0 if budget is None else min(1.0, budget / r["img_long"])
        out[classify(native, native * scale, s)] += 1
    n = sum(out.values())
    res = {k: v for k, v in out.items()}
    res["n"] = n
    for k in ("sensor_limited", "pipeline_limited", "model_error"):
        res[f"{k}_pct"] = round(100 * out[k] / n, 1) if n else None
        res[f"{k}_wilson95"] = wilson(out[k], n)
    return res


def main():
    analysis = json.load(open(ANALYSIS, encoding="utf-8"))
    labels = json.load(open(LABELS, encoding="utf-8"))
    gsdmap = json.load(open(GSDMAP, encoding="utf-8"))

    failures = [r for r in analysis["records"] if r["failed"]]
    known = []
    unknown = 0
    for r in failures:
        if r["gsd"] is None:
            unknown += 1
            continue
        img = labels[r["task_id"]].get("image")
        p = IMGROOT / img if img else None
        if not p or not p.exists():
            unknown += 1
            continue
        w, h = Image.open(p).size
        known.append({**r, "img_long": max(w, h)})

    xbd = [r for r in known if r["dataset"] == "xBD"]
    dota = [r for r in known if r["dataset"] == "DOTA"]

    # measured physical sizes from DOTA val annotations (validation of nominals)
    phys = {}
    for cls in ("plane", "ship", "small vehicle", "large vehicle"):
        v = sorted(inst["px"] * m["gsd"] for m in gsdmap["images"].values()
                   if m["flag"] == "ok" for inst in m["instances_px"][cls])
        n = len(v)
        phys[cls] = {"n": n, "median_m": round(v[n // 2], 1),
                     "p25_m": round(v[n // 4], 1), "p75_m": round(v[3 * n // 4], 1)}

    result = {
        "framing": "pre-committed two-axis decomposition (sensor vs pipeline vs model)",
        "s_floor_px": S_FLOOR,
        "n_failures": len(failures),
        "n_gsd_known_with_image": len(known),
        "n_unclassified": unknown,
        "measured_object_sizes_dota_val": phys,
        "nominal_sizes_used": analysis["object_size_m"],
        "by_budget": {},
        "dota_prior_sensitivity": {},
        "s_sweep_at_1024": {},
    }

    for b in BUDGETS:
        key = "native" if b is None else str(b)
        result["by_budget"][key] = {
            "all": decompose(known, b),
            "xBD": decompose(xbd, b),
            "DOTA": decompose(dota, b),
        }

    # DOTA prior sensitivity at representative budgets (native + 1024 + 448)
    for name, prior in DOTA_PRIOR_SCENARIOS.items():
        result["dota_prior_sensitivity"][name] = {
            "native": decompose(dota, None, dota_prior=prior),
            "1024": decompose(dota, 1024, dota_prior=prior),
            "448": decompose(dota, 448, dota_prior=prior),
        }

    for s in (8, 10, 12, 15):
        result["s_sweep_at_1024"][str(s)] = {
            "xBD": decompose(xbd, 1024, s=s),
            "DOTA": decompose(dota, 1024, s=s),
        }

    json.dump(result, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    # console summary
    print(f"failures={len(failures)} known={len(known)} (xBD {len(xbd)}, DOTA {len(dota)}), unclassified={unknown}")
    print("\nbudget  family  sensor%  pipeline%  model%")
    for b, fam in result["by_budget"].items():
        for f in ("xBD", "DOTA"):
            d = fam[f]
            print(f"{b:>6}  {f:>5}  {d['sensor_limited_pct']:>6}  {d['pipeline_limited_pct']:>8}  {d['model_error_pct']:>6}")
    print("\nDOTA prior sensitivity (native / 1024 / 448) model-error%:")
    for name, sc in result["dota_prior_sensitivity"].items():
        print(f"  {name:>14}: {sc['native']['model_error_pct']:>5} / {sc['1024']['model_error_pct']:>5} / {sc['448']['model_error_pct']:>5}")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
