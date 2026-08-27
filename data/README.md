# Datasets

Benchmarks are not bundled with this repository due to size and licensing constraints.
Download them from the official sources and place them under this directory as below.

## Layout

```
data/
├── thinkgeo/
│   ├── ThinkGeoBench.json
│   └── image/
├── RSVQA-LR/
│   ├── Images_LR/
│   └── LR_split_test_{questions,answers,images}.json
├── FloodNet/
│   └── Track2/            # FloodNet VQA
└── EarthVL/
    └── EarthVQA/
```

## Sources

| Dataset | License | Download |
|---|---|---|
| **ThinkGeo** | Apache-2.0 | [mbzuai-oryx/ThinkGeo](https://github.com/mbzuai-oryx/ThinkGeo) · [MBZUAI/ThinkGeo](https://huggingface.co/datasets/MBZUAI/ThinkGeo) |
| **RSVQA-LR** | CC BY 4.0 | [zenodo.org/records/6344334](https://zenodo.org/records/6344334) |
| **FloodNet** | MIT | [BinaLab/FloodNet-Supervised_v1.0](https://github.com/BinaLab/FloodNet-Supervised_v1.0) |
| **EarthVQA** | Academic use only | [Junjue-Wang/EarthVQA](https://github.com/Junjue-Wang/EarthVQA) |

> **RSVQA-LR, not HR.** Zenodo record `6344334` is the *low resolution* (Sentinel-2,
> ~10 m/px) release used throughout this project. Record `6344367` is RSVQA-HR and is
> not used.

> **DOTA-v1 (detector routing, Table III).** The counting experiments additionally
> need the DOTA-v1 validation split in YOLO-OBB format at `data/DOTAv1/{images,labels}/val`.
> If it lives elsewhere, point at it with the `DOTA_ROOT` environment variable:
> `DOTA_ROOT=/path/to/DOTAv1 python src/eval/run_dota_val_counting.py`.
> `extract_dota_gsd.py` additionally reads the release archive at `data/DOTAv1.zip`
> (override with `DOTA_ZIP`) for the original label headers that carry per-image GSD.

> **EarthVQA** is restricted to academic use (RSIDEA, Wuhan University) and its images
> additionally fall under the Google Earth Terms of Use.

Taxonomy annotations for ThinkGeo tasks ship with this repository at
[`../annotation/thinkgeo_taxonomy_labels.json`](../annotation/thinkgeo_taxonomy_labels.json);
`task_id` is the integer index into `ThinkGeoBench.json`.
