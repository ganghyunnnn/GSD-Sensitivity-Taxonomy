---
license: cc-by-4.0
language:
- en
pretty_name: "GSD-Sensitivity Taxonomy: Task Labels for Remote Sensing VQA"
size_categories:
- 1K<n<10K
task_categories:
- visual-question-answering
- image-classification
tags:
- remote-sensing
- vqa
- taxonomy
- gsd
- earth-observation
- inter-annotator-agreement
configs:
- config_name: thinkgeo_labels
  data_files: thinkgeo_taxonomy_labels.json
- config_name: iaa_sample
  data_files: iaa_sample.csv
- config_name: iaa_annotator2
  data_files: iaa_annotator2.json
---

# GSD-Sensitivity Taxonomy: Task Labels for Remote Sensing VQA

Per-task **D / M1 / M2** taxonomy labels, inter-annotator agreement (IAA) data, and
evaluation traces for four public RS-VQA benchmarks.

> Companion to **G. Park and D.-H. Lee, "Identifying the Measurement Gap in Remote
> Sensing VQA with a GSD-Sensitive Taxonomy," *IEEE Geosci. Remote Sens. Lett.*, 2026**
> — accepted, DOI to follow. Code: [github.com/ganghyunnnn/GSD-Sensitivity-Taxonomy](https://github.com/ganghyunnnn/GSD-Sensitivity-Taxonomy)

> ⚠️ This dataset contains **annotations and evaluation artifacts only**. The
> underlying benchmark images and questions are *not* redistributed — download them
> from the original sources and join on `task_id`.

## Taxonomy

One counterfactual: *if the GSD were doubled, would the answer value change (M1), or
would the question become physically unanswerable while the value stays the same (M2)?*

| | Type | Definition |
|---|---|---|
| **D**  | Descriptive    | GSD-invariant; visual–semantic interpretation only. *"What is the land use type?"* |
| **M1** | Spatial Metric | Value scales with GSD (real-world distance / area). *"Distance between the two hangars? (GSD = 0.3 m/px)"* |
| **M2** | Cardinality    | Counting; the value is GSD-invariant but **answerability is resolution-conditioned** (feasible only when GSD ≤ d/s for target size d, threshold s ≈ 10–15 px). *"How many vehicles are in the parking lot?"* |

**Boundary rules.** Proximity queries are M1 with a numeric distance threshold, D
without. Bounding-box drawing is D (pixel-coordinate output). Comparisons inherit
their operation: counting-based → M2, GSD-based spatial → M1.

**Headline results.** Across 293,607 questions, M-type prevalence ranges 2.9–70.4%.
Measurement tasks fail 19–31 pp more often than descriptive tasks across two agent
baselines and three VLM backbones, robust to Benjamini–Hochberg correction.
IAA: Cohen's κ = 0.95. Rule-based classifier: 95.2% agreement.

## Files

| File | Contents |
|---|---|
| `thinkgeo_taxonomy_labels.json` | Per-task D/M1/M2 labels (multi-label) + verbatim question text and image filename |
| `review_436.csv` | Full 436-task human review with verbatim question text |
| `thinkgeo_taxonomy_summary.json` | Distribution summary over ThinkGeo |
| `iaa_sample.csv`, `iaa_sample_annotator2.csv`, `iaa_annotator2.json`, `iaa_guideline.md`, `iaa_sample_annotator2_rationale_ko.md` | 88-task stratified IAA sample: both annotators, guideline, per-task rationale (Korean) |
| `router_eval_3type.json` | Rule-based classifier metrics on the 189-task split |
| `backbone_*.json`, `rsvqa_*.json`, `floodnet_*.json` | VLM evaluation traces (ThinkGeo / RSVQA-LR / FloodNet Track-2) |
| `task_level_*.json` | ThinkGeo agent baselines (Vanilla ReAct, Direct Prompting) |
| `gsd_ablation.json`, `m2ab_*.json`, `m2search_*.json`, `routed_eval_*.json` | Prompt-level interventions: GSD injection, M2 direct-vs-counting decomposition, counting-prompt search, taxonomy-routed prompting |
| `failure_analysis_by_type.json`, `bootstrap_sensitivity.json` | Failure rates by D/M type; bootstrap CI sensitivity |

`thinkgeo_taxonomy_labels.json` is keyed by `task_id` (integer index into
ThinkGeoBench); `annotation.types` holds the multi-label D/M1/M2 list. IAA CSVs use
`task_id, image, query, type_annotator, notes`. Evaluation traces are per-task
`{task_id, type, prompt, prediction, reference, correct, ...}`.

## Loading

```python
import json, urllib.request
from datasets import load_dataset

url = "https://huggingface.co/datasets/ganghyunnnn/GSD-Sensitivity-Taxonomy-Labels/resolve/main/thinkgeo_taxonomy_labels.json"
labels = json.loads(urllib.request.urlopen(url).read())

ds = load_dataset("ganghyunnnn/GSD-Sensitivity-Taxonomy-Labels", name="iaa_sample", split="train")
```

To reproduce the paper, download the source benchmarks and run
`python src/eval/run_all_experiments.py` from the GitHub repository.

## Source Benchmarks

[ThinkGeo](https://github.com/mbzuai-oryx/ThinkGeo) (Apache-2.0) ·
[RSVQA-LR](https://zenodo.org/records/6344334) (CC BY 4.0) ·
[FloodNet](https://github.com/BinaLab/FloodNet-Supervised_v1.0) (MIT) ·
[EarthVQA](https://github.com/Junjue-Wang/EarthVQA) (academic-only, RSIDEA / Wuhan
University). EarthVQA and FloodNet appear in cross-benchmark distribution counts only;
no content from either is redistributed here.

## Annotation Process

Annotator 1 (lead author) labeled all ThinkGeoBench tasks; Annotator 2 independently
labeled an 88-task stratified sample from question text and a written guideline alone.
A task may carry multiple tags when the answer needs more than one capability
(e.g. `D+M2` = identify + count). IAA is reported as Cohen's κ and macro-F1 per label.

## Limitations

- Multi-label annotation introduces label-set ambiguity; rationale notes document marginal cases.
- M1 is under-represented in the evaluation split; the paper mitigates this with bootstrap analysis.
- The RSVQA-LR replication covers D/M2 only — no M1 questions exist at 10 m/px Sentinel-2 resolution.
- The GSD-injection ablation (N=161) is powered only for effects ≥12 pp.
- ThinkGeo `task_id` indexing must match the upstream JSON release used at annotation time.

## License

Portions authored by this project — the D/M1/M2 labels, IAA rationale, evaluation
outputs, guideline and this card — are released under **CC BY 4.0**.

`thinkgeo_taxonomy_labels.json`, `review_436.csv`, `iaa_sample.csv`,
`iaa_sample_annotator2.csv` and `iaa_sample_annotator2_rationale_ko.md` additionally
embed verbatim ThinkGeoBench question text and/or image filenames, which remain under
**Apache-2.0** (Shabbir et al., MBZUAI Oryx Lab); redistribution must preserve that
attribution. `rsvqa_*.json` retains only upstream `q_id` integers. Full details:
[`NOTICE`](https://github.com/ganghyunnnn/GSD-Sensitivity-Taxonomy/blob/master/NOTICE).

Benchmark images are not redistributed and remain under their original licenses.

## Citation

```bibtex
@article{park2026gsdtaxonomy,
  title   = {Identifying the Measurement Gap in Remote Sensing {VQA} with a {GSD}-Sensitive Taxonomy},
  author  = {Park, Ganghyun and Lee, Dong-Ho},
  journal = {IEEE Geoscience and Remote Sensing Letters},
  year    = {2026},
  note    = {Accepted for publication}
}
```
