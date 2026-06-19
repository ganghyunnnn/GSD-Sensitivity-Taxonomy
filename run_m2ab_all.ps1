# Sequential M2 A/B test across all 3 backbones (Ollama single-instance → sequential)
cd A:\rs-taxonomy
chcp 65001 | Out-Null

Write-Output "=== [1/3] qwen3-vl:8b ==="
.\.venv\Scripts\python.exe src/eval/run_m2_ab_test.py --model "qwen3-vl:8b" --out "m2ab_qwen3vl_8b.json"

Write-Output "=== [2/3] qwen2.5vl:7b ==="
.\.venv\Scripts\python.exe src/eval/run_m2_ab_test.py --model "qwen2.5vl:7b" --out "m2ab_qwen25vl_7b.json"

Write-Output "=== [3/3] blaifa/InternVL3_5:8b ==="
.\.venv\Scripts\python.exe src/eval/run_m2_ab_test.py --model "blaifa/InternVL3_5:8b" --out "m2ab_internvl35_8b.json"

Write-Output "=== ALL M2 A/B DONE ==="
