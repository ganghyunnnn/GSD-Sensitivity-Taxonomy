# GSD-Sensitivity-Taxonomy

A three-category **GSD-Sensitivity Taxonomy** (D / M1 / M2) for remote sensing VQA,
with a deterministic rule-based classifier and the artifacts that reproduce the paper.

> **G. Park and D.-H. Lee, "Identifying the Measurement Gap in Remote Sensing VQA
> with a GSD-Sensitive Taxonomy," *IEEE Geosci. Remote Sens. Lett.*, 2026** — accepted,
> DOI to follow.

Across **293,607 questions** from four RS-VQA benchmarks, measurement (M) tasks fail
**19–31 pp more often** than descriptive (D) tasks — across two agent baselines and
three VLM backbones, robust to Benjamini–Hochberg correction and unmoved by
prompt-level GSD injection. Routing the *resolvable* M2 branch to an in-domain
oriented-box detector cuts counting error (**MAE 12.8 → 4.2** on held-out DOTA-v1 val).
The bundled classifier reaches **95.2% agreement** with human-reviewed labels at zero
LLM cost.

## Taxonomy

One counterfactual: *if the GSD were doubled, would the answer value change (M1), or
would the question become physically unanswerable while the value stays the same (M2)?*

| | Type | Definition |
|---|---|---|
| **D**  | Descriptive    | GSD-invariant; visual–semantic interpretation only. *"What is the land use type?"* |
| **M1** | Spatial Metric | Value scales with GSD (distance / area). *"Distance between the two hangars?"* |
| **M2** | Cardinality    | Counting; the value is GSD-invariant but **answerability is resolution-conditioned** (feasible only when GSD ≤ d/s, s ≈ 10–15 px). *"How many vehicles are in the parking lot?"* |

Boundary rules and the lexical pattern set live in `src/taxonomy/`. Inter-annotator
agreement on an 88-task stratified ThinkGeo sample: mean Cohen's κ = 0.95.

## Quickstart

```bash
uv sync --extra ml --extra geo --extra agents   # or: pip install -e ".[ml,geo,agents]"
cp .env.example .env                            # API keys for LLM-dependent scripts
python src/eval/run_all_experiments.py          # after placing datasets under data/
```

Python 3.11+. Benchmarks are **not bundled** — ThinkGeo (Apache-2.0), RSVQA-LR
(CC BY 4.0), FloodNet (MIT), EarthVQA (academic-only, not redistributed here).
Download instructions: [`data/README.md`](data/README.md).

Models: Qwen3-VL-8B, Qwen2.5-VL-7B, InternVL3.5-8B backbones; YOLO26l-OBB detector
(DOTA-v1.0 pretrained, AGPL-3.0); gpt-oss-20b and Qwen3.5-9B LLM classifier baselines.

## Reproducibility

| Paper artifact | Script |
|---|---|
| Task-type distribution (Table I) | `src/taxonomy/cross_benchmark_analysis.py` |
| Per-type failure rates (Table II) | `src/eval/run_backbone_eval.py` |
| Resolvable-M2 detector routing (Table III) | `src/eval/run_dota_val_counting.py`, `update_detector_yolo26.py`, `analyze_dota_val_multi.py` |
| Classifier agreement + IAA | `src/router/eval_router_3type.py`, `annotation/iaa_measurement.py` |
| Classifier held-out robustness | `src/eval/test_classifier_heldout.py` |
| M2 failure-cause decomposition | `src/eval/analyze_m2_resolution.py` |
| GSD-injection ablation | `src/eval/run_gsd_ablation.py` |
| Taxonomy-routed prompting | `src/eval/run_routed_eval.py`, `consolidate_routed.py` |
| FloodNet / RSVQA-LR replication | `src/eval/run_floodnet_multi.py`, `run_rsvqa_eval.py` |
| Bootstrap CIs | `src/eval/bootstrap_sensitivity.py` |

> The detector-routing scripts need DOTA-v1 val at `data/DOTAv1/`, or set `DOTA_ROOT`
> to point elsewhere. See [`data/README.md`](data/README.md).

## Annotations

Per-task D/M1/M2 labels, IAA data and evaluation traces are released on the Hugging
Face Hub under CC BY 4.0:
**[ganghyunnnn/GSD-Sensitivity-Taxonomy-Labels](https://huggingface.co/datasets/ganghyunnnn/GSD-Sensitivity-Taxonomy-Labels)**.
Benchmark images are not redistributed; join on `task_id`.

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

## License

Code is [Apache-2.0](LICENSE); the annotations dataset is CC BY 4.0. A few published
files embed verbatim ThinkGeoBench question text, which remains under the upstream
Apache-2.0 license — see [`NOTICE`](NOTICE) for the affected files and required
attribution. Third-party datasets and models keep their original licenses.
