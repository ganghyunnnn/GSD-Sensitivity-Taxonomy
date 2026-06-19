"""
Verbosity-robust re-scoring of FloodNet results from STORED answers (no re-inference).

Motivation: the original scorer matched Yes/No via `tok in pred.split()`, which
fails when a backbone answers verbosely ("No, not all of the road is flooded")
because tokenization yields "no," (with punctuation) != "no". This penalizes
verbose backbones (Qwen2.5-VL) while terse backbones (Qwen3-VL: "No") score
correctly---an unfair, format-dependent artifact (Yes/No 100% fail for Qwen2.5-VL).

Fix: extract the leading yes/no polarity with a word-boundary regex, punctuation-
insensitive. Condition_Recognition and counting (M2) scoring are UNCHANGED, so the
published Qwen3-VL baseline must reproduce exactly (validation check below).

Usage:
    python src/eval/rescore_floodnet.py
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
ANN = ROOT / "annotation"

FILES = {
    "Qwen3-VL-8B":   "floodnet_backbone_qwen3vl_8b.json",
    "Qwen2.5-VL-7B": "floodnet_qwen25vl_7b.json",
    "InternVL3.5-8B":"floodnet_internvl35_8b.json",
}

_NUM = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
        "six":"6","seven":"7","eight":"8","nine":"9","ten":"10"}


def rescore(pred, gt, q_type):
    pred = str(pred).strip().lower()
    gt = str(gt).strip().lower()
    for w, n in _NUM.items():
        pred = pred.replace(w, n); gt = gt.replace(w, n)

    if q_type == "Yes_No":
        m = re.search(r"\b(yes|no)\b", pred)      # leading polarity, punctuation-robust
        return bool(m) and m.group(1) == gt

    if q_type == "Condition_Recognition":
        # UNCHANGED semantics: require the exact benchmark phrase to appear,
        # but avoid the "flooded" in "non flooded" false positive.
        if gt == "flooded":
            has_non = any(t in pred for t in ("non flooded", "non-flooded", "not flooded"))
            return ("flooded" in pred or "flood" in pred) and not has_non
        return gt in pred                          # gt == "non flooded"

    # counting / other: first integer match (UNCHANGED)
    if pred == gt:
        return True
    pn = re.findall(r"\b\d+\b", pred); gn = re.findall(r"\b\d+\b", gt)
    return bool(pn and gn and pn[0] == gn[0])


def summarize(results):
    def fr(sub):
        n = len(sub); c = sum(1 for r in sub if r["_new"]); return n, (1 - c/n)*100 if n else 0.0
    D = [r for r in results if r["taxonomy"] == "D"]
    M2 = [r for r in results if r["taxonomy"] == "M2"]
    yn = [r for r in results if r["q_type"] == "Yes_No"]
    cr = [r for r in results if r["q_type"] == "Condition_Recognition"]
    nD, fD = fr(D); nM, fM = fr(M2); nY, fY = fr(yn); nC, fC = fr(cr)
    return {"D_fail": round(fD,1), "M2_fail": round(fM,1), "gap": round(fM-fD,1),
            "YesNo_fail": round(fY,1), "Cond_fail": round(fC,1),
            "YN_M2_gap": round(fM-fY,1), "nD": nD, "nM2": nM}


def main():
    print("=== FloodNet re-scoring (verbosity-robust Yes/No; Condition+counting unchanged) ===")
    out = {}
    for label, fn in FILES.items():
        p = ANN / fn
        if not p.exists():
            print(f"{label:16s}  (not yet available: {fn})")
            continue
        d = json.load(open(p, encoding="utf-8"))
        res = d["results"]
        old_fail = sum(1 for r in res if not r["correct"])
        for r in res:
            r["_new"] = rescore(r["answer"], r["gt"], r["q_type"])
        new_fail = sum(1 for r in res if not r["_new"])
        s = summarize(res)
        out[label] = s
        print(f"{label:16s}  D={s['D_fail']:5.1f}%  M2={s['M2_fail']:5.1f}%  "
              f"gap={s['gap']:+6.1f}pp | Yes/No={s['YesNo_fail']:5.1f}%  "
              f"Cond={s['Cond_fail']:5.1f}%  YN-vs-M2={s['YN_M2_gap']:+6.1f}pp "
              f"(old_fail={old_fail}->{new_fail})")

    # Validation: Qwen3-VL must match published baseline (40.5 / 54.0 / 6.0 / +48.0)
    if "Qwen3-VL-8B" in out:
        s = out["Qwen3-VL-8B"]
        ok = (s["D_fail"] == 40.5 and s["M2_fail"] == 54.0 and
              s["YesNo_fail"] == 6.0 and s["YN_M2_gap"] == 48.0)
        print(f"\n[VALIDATION] Qwen3-VL reproduces published baseline: "
              f"{'OK' if ok else 'MISMATCH -- investigate'}")

    (ROOT / "experiments").mkdir(exist_ok=True)
    with open(ROOT / "experiments/floodnet_multi_rescored.json", "w", encoding="utf-8") as f:
        json.dump({"description": "FloodNet multi-backbone, verbosity-robust Yes/No re-scoring; "
                                  "Condition_Recognition and counting scoring unchanged; "
                                  "Qwen3-VL reproduces the published baseline.",
                   "backbones": out}, f, indent=2, ensure_ascii=False)
    print("Saved: experiments/floodnet_multi_rescored.json")


if __name__ == "__main__":
    main()
