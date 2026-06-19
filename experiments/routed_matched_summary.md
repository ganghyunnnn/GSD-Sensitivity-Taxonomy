# Matched-resolution taxonomy-routed prompting

13 paired McNemar tests; 8 nominal improvements; 0 survive BH.

| Backbone | Type | Routed prompt | Direct fail | Routed fail | Effect | p | p(BH) | N |
|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B | M1 | few_shot | 81.4% | 81.4% | +0.0pp | 0.724 | 0.855 | 161 |
| Qwen2.5-VL-7B | M1 | few_shot | 85.7% | 82.0% | -3.7pp | 0.041 | 0.099 | 161 |
| InternVL3.5-8B | M1 | few_shot | 89.4% | 85.7% | -3.7pp | 0.041 | 0.099 | 161 |
| Qwen3-VL-8B | M2 | counting | 79.5% | 77.1% | -2.4pp | 0.752 | 0.855 | 83 |
| Qwen2.5-VL-7B | M2 | counting | 89.2% | 97.6% | +8.4pp | 0.045 | 0.099 | 83 |
| InternVL3.5-8B | M2 | counting | 83.1% | 85.5% | +2.4pp | 0.789 | 0.855 | 83 |
| Qwen3-VL-8B | M2 | grid | 78.3% | 73.5% | -4.8pp | 0.387 | 0.611 | 83 |
| Qwen3-VL-8B | M2 | cot_count | 78.3% | 67.5% | -10.8pp | 0.016 | 0.099 | 83 |
| Qwen3-VL-8B | M2 | systematic | 78.3% | 77.1% | -1.2pp | 1.000 | 1.000 | 83 |
| Qwen2.5-VL-7B | M2 | grid | 86.7% | 94.0% | +7.2pp | 0.114 | 0.211 | 83 |
| Qwen2.5-VL-7B | M2 | cot_count | 86.7% | 74.7% | -12.0pp | 0.034 | 0.099 | 83 |
| InternVL3.5-8B | M2 | grid | 81.9% | 86.7% | +4.8pp | 0.423 | 0.611 | 83 |
| InternVL3.5-8B | M2 | cot_count | 81.9% | 71.1% | -10.8pp | 0.039 | 0.099 | 83 |
