---
license: cc-by-4.0
language:
- en
pretty_name: "RS-Taxonomy: GSD-Sensitive Task Labels for Remote Sensing VQA"
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

# RS-Taxonomy: GSD-Sensitive Task Labels for Remote Sensing VQA

This dataset accompanies the paper *"Identifying the Measurement Gap in
Remote Sensing VQA with a GSD-Sensitive Taxonomy"* (IEEE GRSL, under review).
It provides per-task **D / M1 / M2** taxonomy labels, inter-annotator agreement
(IAA) data, and reproducibility artifacts for four public RS-VQA benchmarks.

The taxonomy partitions tasks via a single counterfactual: *if the GSD were
doubled, would the answer value change (M1) or would the question become
physically unanswerable while the value stays the same (M2)?*

| Type | Name | Definition |
|------|------|------------|
| **D**  | Descriptive    | GSD-invariant; visual–semantic interpretation only. *E.g.,* "What is the land use type?", "Is there an airport?" |
| **M1** | Spatial Metric | Output value scales with GSD (real-world distance / area). *E.g.,* "Distance between the two hangars? (GSD = 0.3 m/px)" |
| **M2** | Cardinality    | Counting tasks; values are GSD-invariant but **answerability is resolution-conditioned** (counting feasible only when GSD ≤ d/s for target size d, resolvability threshold s ≈ 10–15 px). *E.g.,* "How many vehicles are in the parking lot?" |

**Boundary rules.** (1) Proximity queries with a numeric distance threshold are M1, otherwise D. (2) Bounding-box drawing is always D (output is pixel coordinates). (3) Comparisons inherit the underlying operation: counting-based comparisons → M2; GSD-based spatial comparisons → M1.

**Headline results from the paper.** Across 293,607 questions in four benchmarks, M-type prevalence ranges from 2.9% to 70.4%. Measurement tasks fail 19–31 pp more often than descriptive tasks across two agent baselines and three VLM backbones, robust to BH multiple-comparison correction. A GSD prompt-injection ablation on 161 M1 tasks shows no significant improvement after correction. Inter-annotator agreement: **Cohen's κ = 0.95**. Rule-based classifier accuracy: **95.2%**.

> **Important**: This dataset contains **only the taxonomy annotations and
> evaluation artifacts**. The underlying RS-VQA benchmark images and questions
> are *not* redistributed here — download them from the original sources and
> join on `task_id`.

## Dataset Summary

| File | Type | Description |
|------|------|-------------|
| `thinkgeo_taxonomy_labels.json`   | labels      | ThinkGeoBench task → list of D/M1/M2 labels (multi-label) |
| `thinkgeo_taxonomy_summary.json`  | aggregate   | Distribution summary over ThinkGeo |
| `iaa_sample.csv`                  | IAA seed    | 50 stratified ThinkGeo tasks (annotator 1) |
| `iaa_sample_annotator2.csv`       | IAA         | Same 50 tasks labeled by annotator 2 |
| `iaa_annotator2.json`             | IAA         | Annotator 2 labels in JSON form |
| `iaa_sample_annotator2_rationale_ko.md` | IAA notes | Per-task rationale (Korean) |
| `iaa_guideline.md`                | docs        | Annotation guideline |
| `router_eval*.json`               | results     | Rule/LLM/hybrid router evaluation |
| `llm_router_preds*.json`          | results     | Per-task LLM router predictions |
| `backbone_*.json`                 | results     | Backbone VLM evaluation traces |
| `rsvqa_*.json`                    | results     | RSVQA-LR evaluation traces |
| `gsd_ablation.json`               | results     | GSD-prompt ablation |
| `bootstrap_sensitivity.json`      | results     | Bootstrap CI sensitivity |
| `failure_analysis_by_type.json`   | results     | Failure rates by D/M type |
| `task_level_*.json`               | results     | Task-level prompting comparisons |

## Source Benchmarks

| Benchmark | License | Where to download |
|-----------|---------|-------------------|
| ThinkGeo  | Apache-2.0 | https://github.com/mbzuai-oryx/ThinkGeo |
| RSVQA-LR  | CC BY 4.0  | https://zenodo.org/records/6344334 |
| FloodNet  | MIT        | https://github.com/BinaLab/FloodNet-Supervised_v1.0 |
| EarthVQA  | Academic only* | https://github.com/Junjue-Wang/EarthVQA |

\* EarthVQA images are restricted to academic use (RSIDEA, Wuhan University).
This dataset does **not** redistribute EarthVQA content; cross-benchmark
distribution figures were computed locally from the academic release.

### Citations for Source Benchmarks

```bibtex
@misc{thinkgeo,
  author    = {Shabbir, Akashah and Munir, Muhammad Akhtar and Dudhane, Akshay and
               Sheikh, Muhammad Umer and Khan, Muhammad Haris and Fraccaro, Paolo and
               Moreno, Juan Bernabe and Khan, Fahad Shahbaz and Khan, Salman},
  title     = {{ThinkGeo}: Evaluating Tool-Augmented Agents for Remote Sensing Tasks},
  year      = {2025},
  eprint    = {2505.23752},
  archivePrefix = {arXiv}
}

@article{rsvqa,
  author  = {Lobry, Sylvain and Marcos, Diego and Murray, Jesse and Tuia, Devis},
  title   = {{RSVQA}: Visual Question Answering for Remote Sensing Data},
  journal = {IEEE Trans. Geosci. Remote Sens.},
  year    = {2020}, volume = {58}, number = {12}, pages = {8555--8566},
  doi     = {10.1109/TGRS.2020.2988782}
}

@article{floodnet,
  author  = {Rahnemoonfar, Maryam and Chowdhury, Tashnim and Sarkar, Argho and
             Varshney, Debvrat and Yari, Masoud and Murphy, Robin R.},
  title   = {{FloodNet}: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding},
  journal = {IEEE Access}, year = {2021}, volume = {9}, pages = {89644--89654},
  doi     = {10.1109/ACCESS.2021.3090981}
}

@inproceedings{earthvqa,
  author    = {Wang, Junjue and Zheng, Zhuo and Chen, Zihang and Ma, Ailong and Zhong, Yanfei},
  title     = {{EarthVQA}: Towards Queryable Earth via Relational Reasoning-Based Remote Sensing Visual Question Answering},
  booktitle = {Proc. AAAI Conf. Artificial Intelligence},
  year      = {2024}, volume = {38}, number = {6}, pages = {5481--5489},
  doi       = {10.1609/aaai.v38i6.28357}
}
```

## File Provenance

| File pattern | Provenance | Upstream license |
|---|---|---|
| `thinkgeo_taxonomy_labels.json`, `thinkgeo_taxonomy_summary.json` | Our labels keyed by upstream `task_id` (no question text) | n/a (our work) |
| `iaa_sample.csv`, `iaa_sample_annotator2.csv` | Our labels + **verbatim ThinkGeoBench question text and image filenames** for the 50-task IAA sample | ThinkGeoBench, Apache-2.0 |
| `iaa_annotator2.json` | Our labels keyed by `task_id` only | n/a (our work) |
| `iaa_sample_annotator2_rationale_ko.md` | Our rationale + Korean translations/quotations of selected questions | quoted text: ThinkGeoBench, Apache-2.0 |
| `iaa_guideline.md` | Our annotation guideline | n/a (our work) |
| `review_436.csv` | Our labels + **verbatim ThinkGeoBench question text** | ThinkGeoBench, Apache-2.0 |
| `router_eval*.json`, `llm_router_preds*.json`, `router_llm_qwen35_9b.json` | Router accuracy and predictions | n/a (our work) |
| `backbone_*.json`, `gsd_ablation.json`, `task_level_*.json` | Model predictions on ThinkGeoBench tasks (model outputs are ours; questions referenced by `task_id`) | n/a (our work) |
| `rsvqa_*.json` | Model predictions on RSVQA-LR (only upstream `q_id` integers retained) | n/a (our work) |
| `bootstrap_sensitivity.json`, `failure_analysis_by_type.json` | Aggregate statistics | n/a (our work) |

## Schema

### `thinkgeo_taxonomy_labels.json`
```json
{ "<task_id>": ["D"], "<task_id>": ["M1", "M2"], ... }
```
Multi-label list per task. `task_id` is the integer index into ThinkGeoBench.

### `iaa_sample.csv` / `iaa_sample_annotator2.csv`
Columns: `task_id, image, query, type_annotator, notes`
where `type_annotator` ∈ `{D, M1, M2, D+M1, D+M2, M1+M2, D+M1+M2, ...}`.

### `router_eval*.json`
Aggregated router accuracy / per-class precision-recall.

### `backbone_*.json`, `rsvqa_*.json`
Per-task records: `{task_id, type, prompt, prediction, reference, correct, ...}`.

## Loading

```python
from datasets import load_dataset
import json, urllib.request

# Single file
url = "https://huggingface.co/datasets/ganghyunnnn/rs-taxonomy-labels/resolve/main/thinkgeo_taxonomy_labels.json"
labels = json.loads(urllib.request.urlopen(url).read())

# IAA CSV via datasets
ds = load_dataset(
    "ganghyunnnn/rs-taxonomy-labels",
    name="iaa_sample",
    split="train",
)
```

## Reproducing Paper Results

Code lives in the companion GitHub repository:
**https://github.com/ganghyunnnn/rs-taxonomy**

After downloading the source benchmarks, run:
```bash
python src/eval/run_all_experiments.py
```

## Annotation Process

- Annotator 1 (lead, paper author): labeled all ThinkGeoBench tasks.
- Annotator 2: independently labeled a 50-task stratified sample for IAA.
- Multi-label scheme: a task may carry multiple D/M tags when the answer
  requires more than one capability (e.g., `D+M2` = identify + count).
- IAA computed as Cohen's κ and macro-F1 per label.

## Limitations

- Multi-label annotation introduces label-set ambiguity; rationale notes
  document marginal cases.
- The eval split's M1 under-representation is mitigated by bootstrap analysis
  in the paper.
- The RSVQA-LR replication is confined to D/M2 (no M1 questions exist in
  that benchmark at 10 m/px Sentinel-2 resolution).
- The GSD injection ablation (N=161) is powered only for effects ≥12 pp;
  smaller improvements would require a larger sample.
- ThinkGeo `task_id` indexing must match the upstream JSON release used
  at the time of annotation; see `data/README.md` in the GitHub repo.

## License

Portions authored by this project — per-task D / M1 / M2 labels, IAA
rationale, evaluation outputs, this dataset card, and the annotation
guideline — are released under
**Creative Commons Attribution 4.0 International (CC BY 4.0)**.

A subset of the published files additionally redistributes verbatim
question text and/or image filenames from upstream RS-VQA benchmarks
(see *File Provenance* above). For those embedded portions the upstream
license takes precedence:

- **ThinkGeoBench question text and image filenames** in
  `iaa_sample.csv`, `iaa_sample_annotator2.csv`, `review_436.csv`, and
  `iaa_sample_annotator2_rationale_ko.md` remain under
  **Apache License 2.0** (Shabbir et al., MBZUAI Oryx Lab).
  Downstream redistribution must preserve the upstream attribution; see
  the `NOTICE` file in the companion GitHub repository for the full text.

The underlying benchmark *images* are not redistributed and remain under
their original licenses.

## Citation

```bibtex
@article{park2026rstaxonomy,
  title   = {Identifying the Measurement Gap in Remote Sensing VQA with a GSD-Sensitive Taxonomy},
  author  = {Park, Ganghyun and Lee, Dongho},
  journal = {IEEE Geoscience and Remote Sensing Letters},
  year    = {2026},
  note    = {Under review}
}
```
