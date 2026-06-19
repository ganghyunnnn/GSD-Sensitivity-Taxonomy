#!/bin/bash
# Full M1 tool-routing run across 3 backbones, STRICTLY SEQUENTIAL
# (Ollama single GPU instance -- never parallel). Checkpointed/resumable.
cd /a/rs-taxonomy
export PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python.exe

for M in "qwen2.5vl:7b" "qwen3-vl:8b" "blaifa/InternVL3_5:8b"; do
  echo "================ $M ================"
  "$PY" src/eval/run_m1_tool_routing.py --model "$M" || { echo "FAILED on $M"; exit 1; }
done
echo "STATUS=DONE all 3 backbones complete"
