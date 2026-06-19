"""Detector vs EACH of the 3 paper backbones on DOTA-val counting.
Aggregate exact/MAE/McNemar for detector vs {Qwen2.5-VL, Qwen3-VL, InternVL3.5}."""
import io, json, math, sys, random
from pathlib import Path
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except AttributeError:
    pass
random.seed(0)
ROOT = Path(__file__).resolve().parent.parent.parent
ROWS = json.load(open(ROOT / "experiments/dota_val_counting.json", encoding="utf-8"))["rows"]
VLMS = {"Qwen2.5-VL": "vlm", "Qwen3-VL": "vlm_q3", "InternVL3.5": "vlm_ivl"}


def mcnemar_exact(b, c):
    from math import comb
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, sum(comb(n, i) for i in range(k + 1)) / (2 ** n) * 2)


def main():
    print(f"cells={len(ROWS)}  (det vs each backbone, exact-match McNemar paired)\n")
    print(f"{'backbone':14} {'n':>4} {'det_MAE':>8} {'vlm_MAE':>8} {'det_ex%':>8} {'vlm_ex%':>8} "
          f"{'b(det)':>7} {'c(vlm)':>7} {'McNemar p':>10}")
    out = {}
    for name, key in VLMS.items():
        cells = [r for r in ROWS if r.get(key) is not None]
        if not cells:
            print(f"{name:14}  (not available yet)"); continue
        det_mae = sum(abs(r["det"] - r["gt"]) for r in cells) / len(cells)
        vlm_mae = sum(abs(r[key] - r["gt"]) for r in cells) / len(cells)
        det_ex = sum(r["det"] == r["gt"] for r in cells) / len(cells) * 100
        vlm_ex = sum(r[key] == r["gt"] for r in cells) / len(cells) * 100
        b = sum(1 for r in cells if r["det"] == r["gt"] and r[key] != r["gt"])
        c = sum(1 for r in cells if r["det"] != r["gt"] and r[key] == r["gt"])
        p = mcnemar_exact(b, c)
        out[name] = {"n": len(cells), "det_mae": round(det_mae, 2), "vlm_mae": round(vlm_mae, 2),
                     "det_exact": round(det_ex, 1), "vlm_exact": round(vlm_ex, 1),
                     "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p}
        print(f"{name:14} {len(cells):>4} {det_mae:>8.2f} {vlm_mae:>8.2f} {det_ex:>8.1f} {vlm_ex:>8.1f} "
              f"{b:>7} {c:>7} {p:>10.2e}")
    json.dump(out, open(ROOT / "experiments/dota_val_multi.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("\nSaved: experiments/dota_val_multi.json")


if __name__ == "__main__":
    main()
