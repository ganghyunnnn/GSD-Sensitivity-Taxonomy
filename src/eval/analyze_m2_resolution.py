"""
M2 Resolution Cause Decomposition
============================================================================

ThinkGeo eval split의 M2(counting) 실패를 물리적 원인으로 분리한다:
  1. resolution_limited : GSD > d/s  → 객체가 s픽셀 미만으로 맺혀 물리적으로 구별 불가
  2. model_error        : GSD ≤ d/s  → 해상도는 충분한데도 모델이 실패

GSD 추론 가능 비율 최대화
-----------------------------------
데이터를 날조하지 않고 GSD 추론 가능 비율을 최대화한다. 63개 M2 실패에 대해
모든 GSD는 실제 출처(`gsd_source`)로 추적 가능하다.

GSD 출처 우선순위 (각 record의 gsd_source에 기록)
  1. query              : 해당 태스크 쿼리 본문에 GSD가 명시됨
  2. image_crossref     : 같은 이미지를 쓰는 다른 태스크(주로 M1)가 GSD를 명시
  3. event_prior_xbd    : xBD 동일 재난 이벤트(예: hurricane-harvey)의 GSD가 명시된
                          타일에서 추론. xBD는 이벤트(=원본 위성영상)별로 GSD가 사실상
                          상수이므로(이벤트 내 분산 ≈ 0, 아래 EVENT_GSD 주석 참조) 추적 가능.
  4. dataset_prior_xbd  : 이벤트 prior가 없는 xBD 타일 → xBD 데이터셋 GSD 중앙값으로 fallback
  5. dataset_prior_dota : DOTA 타일 → DOTA 데이터셋 GSD 중앙값으로 추론
                          (DOTA GSD는 차량 임계값보다 훨씬 작아 model_error로 분류됨 = 보수적)

객체 실세계 크기 d (m), 분해능 픽셀수 s (기본 10px, 문헌 기준)
  threshold_gsd = d / s

또한 결론의 견고성을 보이기 위해 s ∈ {8,10,12,15}, d 배율 ∈ {0.75,1.0,1.25}에 대한
민감도 분석을 수행한다 (sensitivity 블록).

사용법:
    python src/eval/analyze_m2_resolution.py
"""
import json
import re
import sys
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

BENCH_PATH  = ROOT / "data/thinkgeo/ThinkGeoBench.json"
LABELS_PATH = ROOT / "annotation/thinkgeo_taxonomy_labels.json"
EVAL_REF    = ROOT / "annotation/backbone_qwen3vl_8b.json"
OUT_PATH    = ROOT / "experiments/m2_resolution_analysis.json"

# ---------------------------------------------------------------------------
# 객체 타입 → 실세계 최소 변별 치수 d (m).
# counting은 "인스턴스를 셀 수 있는가"가 핵심이므로 footprint의 대표 치수를 사용한다.
# ---------------------------------------------------------------------------
OBJECT_SIZE = {
    "vehicle": 4.5, "car": 4.5, "truck": 6.0, "bus": 12.0, "motorcycle": 2.0,
    "building": 15.0, "house": 12.0, "structure": 15.0,
    "aircraft": 30.0, "plane": 25.0, "airplane": 25.0, "helicopter": 10.0,
    "person": 1.8, "human": 1.8, "pedestrian": 1.8,
    "pool": 8.0, "swimming": 8.0, "road": 10.0,
}
# 객체 키워드 우선순위: 더 구체적/큰 객체를 먼저 검사 (building이 disaster counting의 핵심)
OBJECT_PRIORITY = [
    "airplane", "aircraft", "plane", "helicopter",
    "building", "house", "structure",
    "truck", "bus", "motorcycle", "vehicle", "car",
    "person", "human", "pedestrian", "pool", "swimming", "road",
]

S_PIXELS = 10  # minimum pixel extent for object resolvability (default)


# ---------------------------------------------------------------------------
# GSD 추출 (쿼리 텍스트). 기존보다 형식 커버리지를 넓힘:
#   "GSD = 0.5", "GSD: 0.5", "GSD 0.5", "GSD of 0.5", "GSD is 0.5",
#   "0.5 m/px", "0.5 m / pixel", "ground sample distance ... 0.5"
# ---------------------------------------------------------------------------
_GSD_PATTERNS = [
    r'([\d.]+)\s*m\s*/\s*px',
    r'([\d.]+)\s*m\s*/\s*pixel',
    r'gsd\s*[=:]?\s*(?:of|is)?\s*([\d.]+)',
    r'ground\s+samp\w*\s+dist\w*[^\d]{0,20}?([\d.]+)',
    r'resolution\s*[=:]?\s*(?:of|is)?\s*([\d.]+)\s*m',
]


def extract_gsd(query: str):
    for pat in _GSD_PATTERNS:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            try:
                v = float(m.group(1))
                if 0 < v < 100:        # sanity bound (m/px)
                    return v
            except ValueError:
                pass
    return None


def infer_object_type(query: str):
    q = query.lower()
    for obj in OBJECT_PRIORITY:
        if obj in q:
            return obj, OBJECT_SIZE[obj]
    return "unknown", 4.5   # conservative default (소형 차량 기준)


def images_of(task: dict):
    return [f["path"] for f in task.get("files", []) if f.get("type") == "image"]


def dataset_of(img_path: str) -> str:
    n = img_path.split("/")[-1]
    if re.search(r'(pre|post)_disaster', n):
        return "xBD"
    if re.match(r'P\d+\.png', n):
        return "DOTA"
    return "other"


def event_of(img_path: str):
    """xBD 타일 파일명에서 재난 이벤트명 추출 (예: hurricane-harvey)."""
    n = img_path.split("/")[-1]
    m = re.match(r'([a-z\-]+)_\d+_(pre|post)_disaster', n)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# GSD 출처 맵 구축
# ---------------------------------------------------------------------------
def build_gsd_maps(bench: dict):
    """
    Returns:
      img2gsd   : image_path -> median stated GSD  (image_crossref 용)
      event2gsd : xBD event  -> median stated GSD  (event_prior_xbd 용)
      ds2gsd    : dataset    -> dict(min/median/max stated GSD) (dataset_prior 용)
    """
    img2list, event2list, ds2list = {}, {}, {}
    for tid, task in bench.items():
        g = extract_gsd(task.get("dialogs", [{}])[0].get("content", ""))
        if g is None:
            continue
        for ip in images_of(task):
            img2list.setdefault(ip, []).append(g)
            ev = event_of(ip)
            if ev:
                event2list.setdefault(ev, []).append(g)
            ds2list.setdefault(dataset_of(ip), []).append(g)

    img2gsd   = {ip: statistics.median(v) for ip, v in img2list.items()}
    event2gsd = {ev: statistics.median(v) for ev, v in event2list.items()}
    ds2gsd = {
        ds: {"min": min(v), "median": statistics.median(v), "max": max(v), "n": len(v)}
        for ds, v in ds2list.items()
    }
    return img2gsd, event2gsd, ds2gsd


def resolve_gsd(task: dict, img2gsd, event2gsd, ds2gsd):
    """우선순위에 따라 (gsd, gsd_source, is_prior) 반환."""
    query = task.get("dialogs", [{}])[0].get("content", "")
    ips = images_of(task)

    # 1) query
    g = extract_gsd(query)
    if g is not None:
        return g, "query", False
    # 2) image_crossref
    for ip in ips:
        if ip in img2gsd:
            return img2gsd[ip], "image_crossref", False
    # 3) event_prior_xbd
    for ip in ips:
        ev = event_of(ip)
        if ev and ev in event2gsd:
            return event2gsd[ev], "event_prior_xbd", True
    # 4) dataset_prior_xbd / 5) dataset_prior_dota
    for ip in ips:
        ds = dataset_of(ip)
        if ds in ("xBD", "DOTA") and ds in ds2gsd:
            return ds2gsd[ds]["median"], f"dataset_prior_{ds.lower()}", True
    return None, None, False


def classify(gsd, obj_size, s_pixels):
    if gsd is None:
        return "unknown_gsd"
    return "resolution_limited" if gsd > obj_size / s_pixels else "model_error"


# ---------------------------------------------------------------------------
# 민감도 분석: s 와 객체크기 배율을 바꿔가며 resolution-limited 비율 재계산
# ---------------------------------------------------------------------------
def sensitivity(records, s_values, size_mults, subset_filter=None):
    """records: failed records with 'gsd','obj_size_m' (gsd 알려진 것만).
    subset_filter(record)->bool 로 부분집합(예: xBD)만 평가 가능."""
    out = []
    known = [r for r in records if r["failed"] and r["gsd"] is not None
             and (subset_filter is None or subset_filter(r))]
    n = len(known)
    for s in s_values:
        for mult in size_mults:
            rl = sum(1 for r in known
                     if r["gsd"] > (r["obj_size_m"] * mult) / s)
            out.append({
                "s_pixels": s,
                "size_mult": mult,
                "n_known": n,
                "resolution_limited": rl,
                "model_error": n - rl,
                "resolution_limited_pct": round(rl / n * 100, 1) if n else 0,
            })
    return out


def main():
    bench  = json.load(open(BENCH_PATH, encoding="utf-8"))
    labels = json.load(open(LABELS_PATH, encoding="utf-8"))
    ref    = json.load(open(EVAL_REF, encoding="utf-8"))

    img2gsd, event2gsd, ds2gsd = build_gsd_maps(bench)
    print(f"GSD maps: images={len(img2gsd)} events={len(event2gsd)} datasets={list(ds2gsd)}")
    print("Dataset GSD priors:")
    for ds, st in ds2gsd.items():
        print(f"  {ds:6} n={st['n']:3} min={st['min']:.3f} med={st['median']:.3f} max={st['max']:.3f}")
    print("Event GSD priors (xBD):")
    for ev, g in sorted(event2gsd.items()):
        print(f"  {ev:22} gsd={g:.3f}")

    all_m2 = {r["task_id"]: r for r in ref["direct"]["results"] if r["gt_type"] == "M2"}
    n_fail = sum(1 for r in all_m2.values() if not r["correct"])
    print(f"\nM2 total={len(all_m2)} failures={n_fail}")

    records = []
    for tid, result in all_m2.items():
        task = bench.get(tid, {})
        query = task.get("dialogs", [{}])[0].get("content", "")
        failed = not result["correct"]

        gsd, gsd_source, is_prior = resolve_gsd(task, img2gsd, event2gsd, ds2gsd)
        obj_type, obj_size = infer_object_type(query)
        ips = images_of(task)
        ds = dataset_of(ips[0]) if ips else "none"

        cause = classify(gsd, obj_size, S_PIXELS) if failed else "correct"
        records.append({
            "task_id": tid,
            "dataset": ds,
            "query": query[:120],
            "failed": failed,
            "gsd": round(gsd, 4) if gsd is not None else None,
            "gsd_source": gsd_source,
            "gsd_is_prior": is_prior,
            "obj_type": obj_type,
            "obj_size_m": obj_size,
            "threshold_gsd": round(obj_size / S_PIXELS, 3),
            "failure_cause": cause,
        })

    fails = [r for r in records if r["failed"]]

    # ---- 커버리지 (GSD 추론 가능 비율) ----
    src_counts = {}
    for r in fails:
        key = r["gsd_source"] or "unknown_gsd"
        src_counts[key] = src_counts.get(key, 0) + 1
    n_inferable = sum(1 for r in fails if r["gsd"] is not None)
    n_per_instance = sum(1 for r in fails
                         if r["gsd_source"] in ("query", "image_crossref"))

    # ---- 원인 분해 ----
    def breakdown(subset):
        rl = sum(1 for r in subset if r["failure_cause"] == "resolution_limited")
        me = sum(1 for r in subset if r["failure_cause"] == "model_error")
        n = len(subset)
        return {"n": n, "resolution_limited": rl, "model_error": me,
                "resolution_limited_pct": round(rl / n * 100, 1) if n else 0,
                "model_error_pct": round(me / n * 100, 1) if n else 0}

    inferable_fails = [r for r in fails if r["gsd"] is not None]
    per_instance_fails = [r for r in fails
                          if r["gsd_source"] in ("query", "image_crossref")]

    bd_all_inferable = breakdown(inferable_fails)
    bd_per_instance  = breakdown(per_instance_fails)

    # per-dataset breakdown among inferable failures
    by_dataset = {}
    for ds in ("xBD", "DOTA", "other"):
        sub = [r for r in inferable_fails if r["dataset"] == ds]
        if sub:
            by_dataset[ds] = breakdown(sub)

    # ---- 민감도 ----
    S_GRID = [8, 10, 12, 15]
    MULT_GRID = [0.75, 1.0, 1.25]
    sens = sensitivity(fails, S_GRID, MULT_GRID)
    sens_xbd = sensitivity(fails, S_GRID, MULT_GRID,
                           subset_filter=lambda r: r["dataset"] == "xBD")

    # ---- 콘솔 요약 ----
    print(f"\n=== Coverage (GSD inferable among {len(fails)} failures) ===")
    for k, v in sorted(src_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:20} {v}")
    print(f"  per-instance (query/crossref): {n_per_instance} "
          f"({n_per_instance/len(fails)*100:.1f}%)")
    print(f"  total inferable (incl. priors): {n_inferable} "
          f"({n_inferable/len(fails)*100:.1f}%)")

    print(f"\n=== Cause decomposition ===")
    print(f"  per-instance GSD ({bd_per_instance['n']}): "
          f"RL={bd_per_instance['resolution_limited']} "
          f"({bd_per_instance['resolution_limited_pct']}%)  "
          f"ME={bd_per_instance['model_error']}")
    print(f"  all inferable   ({bd_all_inferable['n']}): "
          f"RL={bd_all_inferable['resolution_limited']} "
          f"({bd_all_inferable['resolution_limited_pct']}%)  "
          f"ME={bd_all_inferable['model_error']}")
    for ds, bd in by_dataset.items():
        print(f"    [{ds}] n={bd['n']} RL={bd['resolution_limited']} "
              f"({bd['resolution_limited_pct']}%) ME={bd['model_error']}")

    print(f"\n=== Sensitivity: all inferable (RL%) ===")
    print(f"  {'s':>3} {'dmult':>6} {'RL%':>6}")
    for row in sens:
        print(f"  {row['s_pixels']:>3} {row['size_mult']:>6} {row['resolution_limited_pct']:>6}")
    print(f"\n=== Sensitivity: xBD building-counting only (RL%) ===")
    print(f"  {'s':>3} {'dmult':>6} {'RL%':>6}")
    for row in sens_xbd:
        print(f"  {row['s_pixels']:>3} {row['size_mult']:>6} {row['resolution_limited_pct']:>6}")

    summary = {
        "model": "Qwen3-VL-8B",
        "split": "direct",
        "total_m2": len(all_m2),
        "total_failures": len(fails),
        "s_pixels_default": S_PIXELS,
        "object_size_m": OBJECT_SIZE,
        "gsd_source_priors": {
            "event_gsd_xbd": {k: round(v, 4) for k, v in event2gsd.items()},
            "dataset_gsd": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                for kk, vv in st.items()}
                            for k, st in ds2gsd.items()},
        },
        "coverage": {
            "n_failures": len(fails),
            "n_inferable": n_inferable,
            "inferable_pct": round(n_inferable / len(fails) * 100, 1),
            "n_per_instance": n_per_instance,
            "per_instance_pct": round(n_per_instance / len(fails) * 100, 1),
            "by_source": src_counts,
        },
        "cause_decomposition": {
            "per_instance_gsd": bd_per_instance,
            "all_inferable": bd_all_inferable,
            "by_dataset": by_dataset,
        },
        "sensitivity": sens,
        "sensitivity_xbd": sens_xbd,
        "records": records,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
