# Remaining Ollama work, STRICTLY SEQUENTIAL (single Ollama instance).
# REQUIRED first (FloodNet multi-backbone), then SUPPLEMENTARY M2 prompt-search.
cd A:\rs-taxonomy
chcp 65001 | Out-Null
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"

Write-Output "=== [1/5] REQUIRED FloodNet multi: qwen2.5vl:7b ==="
& $py src/eval/run_floodnet_multi.py --model "qwen2.5vl:7b" --out "floodnet_qwen25vl_7b.json"

Write-Output "=== [2/5] REQUIRED FloodNet multi: InternVL3_5:8b ==="
& $py src/eval/run_floodnet_multi.py --model "blaifa/InternVL3_5:8b" --out "floodnet_internvl35_8b.json"

Write-Output "=== [3/5] SUPP M2 prompt-search finish: qwen3-vl:8b ==="
& $py src/eval/run_m2_prompt_search.py --model "qwen3-vl:8b" --out "m2search_qwen3vl_8b.json"

Write-Output "=== [4/5] SUPP M2 prompt-search transfer: qwen2.5vl:7b ==="
& $py src/eval/run_m2_prompt_search.py --model "qwen2.5vl:7b" --out "m2search_qwen25vl_7b.json" --conditions direct grid cot_count

Write-Output "=== [5/5] SUPP M2 prompt-search transfer: InternVL3_5:8b ==="
& $py src/eval/run_m2_prompt_search.py --model "blaifa/InternVL3_5:8b" --out "m2search_internvl35_8b.json" --conditions direct grid cot_count

Write-Output "=== ALL REMAINING OLLAMA WORK DONE ==="
