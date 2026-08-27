"""
Extract per-image GSD (and image source) from DOTA-v1 original label headers.

Source: the original DOTA-v1 label headers in labels/{val,train}_original/*.txt,
        read from the extracted tree at DOTA_ROOT, or from the release archive at
        DOTA_ZIP when only the archive is available.
Header format:
    imagesource:GoogleEarth
    gsd:0.125266546447

Output: experiments/dota_val_gsd_map.json
    {stem: {gsd, imagesource, flag}}
Flags:
    ok           numeric, plausible
    null         gsd:null / missing / unparsable
    placeholder  gsd == 1.0 exactly and imagesource GF (Gaofen placeholder)
    implausible  gsd <= 0 or gsd > 10

Also records per-instance OBB long side (native px) per image for the four
paper-relevant M2 classes, so downstream analyses (two-axis decomposition,
matched-pair GT) can use measured instance sizes instead of nominal ones.
"""
import json
import math
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
# Original label headers carry the per-image gsd. They are read from the extracted
# tree the counting scripts already use; the release archive is the fallback for a
# checkout that only has the zip. Both locations are overridable.
DOTA_ROOT = Path(os.environ.get("DOTA_ROOT", ROOT / "data" / "DOTAv1"))
ZIP = Path(os.environ.get("DOTA_ZIP", ROOT / "data" / "DOTAv1.zip"))
OUT = ROOT / "experiments/dota_val_gsd_map.json"

PRIMARY = {"plane", "ship", "small-vehicle", "large-vehicle"}
# DOTA original labels use hyphenated class names
CANON = {"plane": "plane", "ship": "ship",
         "small-vehicle": "small vehicle", "large-vehicle": "large vehicle"}


def iter_labels(split):
    """Yield (stem, text) for each original label file in `split`."""
    d = DOTA_ROOT / "labels" / f"{split}_original"
    if d.is_dir():
        for f in sorted(d.glob("*.txt")):
            yield f.stem, f.read_text(encoding="utf-8", errors="replace")
        return
    prefix = f"DOTAv1/labels/{split}_original/"
    with zipfile.ZipFile(ZIP) as z:
        for name in z.namelist():
            if name.startswith(prefix) and name.endswith(".txt"):
                yield Path(name).stem, z.read(name).decode("utf-8", errors="replace")


def parse_split(split):
    out = {}
    for stem, text in iter_labels(split):
        gsd = None
        src = None
        inst = {c: [] for c in CANON.values()}
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("imagesource:"):
                src = line.split(":", 1)[1].strip()
                continue
            if line.startswith("gsd:"):
                raw = line.split(":", 1)[1].strip()
                try:
                    gsd = float(raw)
                except ValueError:
                    gsd = None
                continue
            parts = line.split()
            if len(parts) >= 10 and parts[8] in PRIMARY:
                pts = [(float(parts[i]), float(parts[i + 1])) for i in range(0, 8, 2)]
                edges = [math.dist(pts[i], pts[(i + 1) % 4]) for i in range(4)]
                longest = max(range(4), key=lambda i: edges[i])
                (x1, y1), (x2, y2) = pts[longest], pts[(longest + 1) % 4]
                angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180
                if angle > 90:
                    angle = 180 - angle  # 0 = horizontal, 90 = vertical
                cx = sum(p[0] for p in pts) / 4
                cy = sum(p[1] for p in pts) / 4
                inst[CANON[parts[8]]].append(
                    {"px": round(max(edges), 1), "angle_deg": round(angle, 1),
                     "cx": round(cx, 1), "cy": round(cy, 1)})
        if gsd is None:
            flag = "null"
        elif gsd == 1.0 and src == "GF":
            flag = "placeholder"
        elif gsd <= 0 or gsd > 10:
            flag = "implausible"
        else:
            flag = "ok"
        out[stem] = {"gsd": gsd, "imagesource": src, "flag": flag,
                     "instances_px": inst}
    return out


def main():
    val = parse_split("val")
    flags = {}
    gsds = []
    for v in val.values():
        flags[v["flag"]] = flags.get(v["flag"], 0) + 1
        if v["flag"] == "ok":
            gsds.append(v["gsd"])
    gsds.sort()
    n = len(gsds)
    summary = {
        "n_images": len(val),
        "flags": flags,
        "gsd_ok_stats": {
            "n": n,
            "min": gsds[0], "max": gsds[-1],
            "median": gsds[n // 2],
            "p25": gsds[n // 4], "p75": gsds[3 * n // 4],
            "p90": gsds[int(n * 0.9)],
        },
    }
    json.dump({"summary": summary, "images": val},
              open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=2))
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
