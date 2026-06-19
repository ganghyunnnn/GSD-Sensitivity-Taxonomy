"""
Master Scheduler — Experiment B → A 순차 실행

실행 순서:
  1. Experiment B: GSD injection ablation (40 M1 tasks × 5조건 × 3모델)  ~1.7h
  2. Experiment A-1: RSVQA-LR InternVL3.5-8b  전수 10,004개             ~11h
  3. Experiment A-2: RSVQA-LR qwen3-vl:8b     N=3,000                  ~12.5h
  4. Experiment A-3: RSVQA-LR qwen2.5vl:7b    N=3,000                  ~14h

사용법:
    python src/eval/run_all_experiments.py
    python src/eval/run_all_experiments.py --skip-b   # B 완료 후 재시작 시
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_step(label: str, cmd: list[str]) -> int:
    log(f"▶ START  {label}")
    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = (time.monotonic() - t0) / 3600
    if result.returncode == 0:
        log(f"DONE   {label}  ({elapsed:.2f}h)")
    else:
        log(f"FAILED {label}  (rc={result.returncode}, {elapsed:.2f}h)")
    return result.returncode


EXPERIMENT_PLAN = [
    {
        "id":    "B",
        "label": "GSD Ablation (40 M1 × 5조건 × 3모델)",
        "cmd":   [sys.executable, "src/eval/run_gsd_ablation.py"],
        "est_h": 1.7,
    },
    {
        "id":    "A-1",
        "label": "RSVQA-LR InternVL3.5-8b 전수",
        "cmd":   [sys.executable, "src/eval/run_rsvqa_eval.py",
                  "--model", "blaifa/InternVL3_5:8b",
                  "--n-per-type", "0"],
        "est_h": 11.0,
    },
    {
        "id":    "A-2",
        "label": "RSVQA-LR qwen3-vl:8b N=3000",
        "cmd":   [sys.executable, "src/eval/run_rsvqa_eval.py",
                  "--model", "qwen3-vl:8b",
                  "--n-per-type", "3000"],
        "est_h": 12.5,
    },
    {
        "id":    "A-3",
        "label": "RSVQA-LR qwen2.5vl:7b N=3000",
        "cmd":   [sys.executable, "src/eval/run_rsvqa_eval.py",
                  "--model", "qwen2.5vl:7b",
                  "--n-per-type", "3000"],
        "est_h": 14.0,
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-b", action="store_true",
                        help="Experiment B를 건너뜀 (이미 완료 시)")
    parser.add_argument("--only", type=str, default=None,
                        help="특정 실험만 실행 (예: --only A-1)")
    args = parser.parse_args()

    plan = EXPERIMENT_PLAN
    if args.skip_b:
        plan = [s for s in plan if s["id"] != "B"]
    if args.only:
        plan = [s for s in plan if s["id"] == args.only]

    total_est = sum(s["est_h"] for s in plan)
    log(f"실험 계획: {[s['id'] for s in plan]}  예상 총 {total_est:.1f}h")
    log(f"예상 완료: {datetime.now().strftime('%m/%d')} + {total_est:.0f}h")

    t_all = time.monotonic()
    failed = []

    for step in plan:
        rc = run_step(step["label"], step["cmd"])
        if rc != 0:
            failed.append(step["id"])
            log(f"[WARNING] {step['id']} 실패 - 다음 실험으로 계속 진행")

    total_elapsed = (time.monotonic() - t_all) / 3600
    log("=" * 50)
    log(f"전체 완료: {total_elapsed:.2f}h")
    if failed:
        log(f"실패한 실험: {failed}")
        log(f"재실행: python src/eval/run_all_experiments.py --only {failed[0]}")
    else:
        log("모든 실험 성공적으로 완료!")


if __name__ == "__main__":
    main()
