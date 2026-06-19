# M2 Failure-Cause Decomposition (supplementary to Table — Section III-B)

Backbone: **Qwen3-VL-8B**, direct prompting. M2 failures: 63 of 83. GSD recovered (inferable): 55 (87%). Resolvability floor s=10 px; GSD<=d/s => resolvable.

| Family | Failures (inferable) | GSD source | Resolution-limited | Model-error |
|---|---|---|---|---|
| xBD | 41 | event/dataset prior + image cross-ref | 36 (88%) | 5 (12%) |
| DOTA | 14 | event/dataset prior + image cross-ref | 1 (7%) | 13 (93%) |
| **All inferable** | 55 | mixed | 37 (67%) | 18 (33%) |

Object-size priors (m): building 15, house 12, vehicle/car 4.5, aircraft 30, plane 25. GSD priors: xBD per disaster event (1.25-3.02 m/px); DOTA median 0.135 m/px. Sensitivity sweep over s in {8,10,12,15} px and object size +-25% in the JSON keeps the xBD resolution-limited share at 83-100%.

Full per-task records (task_id, query, dataset, GSD, source, object, threshold, cause) are in experiments/m2_resolution_analysis.json.