# rs-taxonomy

A **three-category GSD-Sensitivity Taxonomy** (D, M1, M2) for remote sensing
visual question answering (RS-VQA), with a deterministic rule-based classifier
and reproducibility artifacts.

Tasks are partitioned by their physical-measurement dependence on ground sampling
distance (GSD), enabling type-stratified evaluation. Across four RS-VQA benchmarks
(293,607 questions), measurement (M) tasks fail **19–31 pp more often** than
descriptive (D) tasks across two agent baselines and three VLM backbones — robust
to BH multiple-comparison correction and unaffected by prompt-level GSD injection.
Routing the *resolvable* M2 branch to an in-domain oriented-box detector cuts
counting error (MAE 12.8→4.2 on held-out DOTA-v1 val). The bundled rule-based
classifier reaches **95.2% agreement** with human-reviewed labels at zero LLM cost.

## Taxonomy

The descriptive/measurement boundary is a single counterfactual: *if the GSD were
doubled, would the answer value change (M1), or would the question become physically
unanswerable while the value stays the same (M2)?*

| Type | Name | Definition |
|------|------|------------|
| **D**  | Descriptive    | GSD-invariant; answerable from visual–semantic interpretation. *E.g.* "What is the land use type?" |
| **M1** | Spatial Metric | Output value scales with GSD (distance / area). *E.g.* "Distance between the two hangars?" |
| **M2** | Cardinality    | Counting; values are GSD-invariant but **answerability is resolution-conditioned** (feasible only when GSD ≤ d/s, s ≈ 10–15 px). *E.g.* "How many vehicles are in the parking lot?" |

Inter-annotator agreement on an 88-task stratified ThinkGeo sample (20% of 436):
mean Cohen's κ = 0.95. Full boundary rules and the lexical pattern set are in
`src/taxonomy/`.

## Repository Structure

```
rs-taxonomy/
├── src/
│   ├── taxonomy/   # Taxonomy definitions + rule-based classifier
│   ├── router/     # LLM/hybrid router for task-type prediction
│   ├── eval/       # Evaluation, failure analysis, experiments
│   └── models/     # VLM/LLM wrappers (HuggingFace, Ollama)
├── annotation/     # Taxonomy labels + inter-annotator agreement
├── experiments/    # Cross-benchmark distribution + statistical tests
├── hf_dataset/     # Hugging Face dataset card + upload script
└── data/           # Dataset root (not included; see data/README.md)
```

## Installation

Python 3.11+. With [uv](https://docs.astral.sh/uv/) (or pip):

```bash
uv sync --extra ml --extra geo --extra agents   # full evaluation stack
# pip install -e ".[ml,geo,agents]"
```

Copy `.env.example` to `.env` and add API keys before running LLM-dependent scripts.

## Datasets and Models

| Dataset | License | Source |
|---------|---------|--------|
| [ThinkGeo](https://github.com/mbzuai-oryx/ThinkGeo) | Apache-2.0 | Shabbir et al., 2025 |
| [RSVQA-LR](https://zenodo.org/records/6344334) | CC BY 4.0 | Lobry et al., TGRS 2020 |
| [FloodNet](https://github.com/BinaLab/FloodNet-Supervised_v1.0) | MIT | Rahnemoonfar et al., 2021 |
| [EarthVQA](https://github.com/Junjue-Wang/EarthVQA) | Academic only\* | Wang et al., AAAI 2024 |

> \* EarthVQA (RSIDEA, Wuhan University) is restricted to academic use and **not
> distributed** here; images also fall under Google Earth Terms of Use.

Backbones: Qwen3-VL-8B, Qwen2.5-VL-7B, InternVL3.5-8B (Apache-2.0). Detector:
YOLO26l-OBB (`yolo26l-obb.pt`, DOTA-v1.0 pretrained, AGPL-3.0). LLM classifiers:
Qwen3.5-9B, gpt-oss-20b. Download instructions: `data/README.md`.

## Annotations Dataset

Per-task D/M1/M2 labels released on the Hugging Face Hub (CC BY 4.0):
**https://huggingface.co/datasets/ganghyunnnn/rs-taxonomy-labels**
(ThinkGeo full annotation + cross-benchmark distribution + IAA sample + VLM eval
outputs). Original images are **not** redistributed; join on `task_id`.

## Reproducibility

Each paper result maps to a script + an output file. Run the full pipeline with
`python src/eval/run_all_experiments.py` after placing datasets under `data/`.

| Paper artifact | Script |
|---|---|
| Cross-benchmark distribution (Table I) | `src/taxonomy/cross_benchmark_analysis.py` |
| Classifier metrics + IAA (Table IV) | `src/router/eval_router_3type.py`, `annotation/iaa_measurement.py` |
| Backbone VLM failure rates (Table II) | `src/eval/run_backbone_eval.py` |
| GSD-injection ablation | `src/eval/run_gsd_ablation.py` |
| Bootstrap CIs | `src/eval/bootstrap_sensitivity.py` |
| M2 failure-cause decomposition | `src/eval/analyze_m2_resolution.py` |
| Taxonomy-routed prompting | `src/eval/run_routed_eval.py`, `consolidate_routed.py` |
| Resolvable-M2 detector routing (Table III, DOTA-v1 val) | `src/eval/run_dota_val_counting.py`, `update_detector_yolo26.py`, `analyze_dota_val_multi.py` |
| FloodNet / RSVQA-LR replication | `src/eval/run_floodnet_multi.py`, `run_rsvqa_eval.py` |
| Classifier held-out robustness | `src/eval/test_classifier_heldout.py` |

## Citation

A citation will be added here once the associated paper is published.

## License

- **Source code** — [Apache License 2.0](LICENSE).
- **Annotations dataset** — CC BY 4.0 for portions authored by this project; a
  subset redistributes verbatim ThinkGeoBench question text under the upstream
  Apache-2.0 license. See [`NOTICE`](NOTICE) and the dataset card's *File
  Provenance* section.
- **Third-party datasets and models** — original licenses as listed above.
