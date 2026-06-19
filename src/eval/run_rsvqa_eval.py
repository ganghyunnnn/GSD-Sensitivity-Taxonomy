"""
Experiment A: RSVQA-LR D/M2 Failure Rate Analysis
세 VLM 백본으로 RSVQA-LR test split을 평가하여 D/M2 실패율 격차를 측정.

사용법:
    # 단일 모델
    python src/eval/run_rsvqa_eval.py --model qwen2.5vl:7b
    python src/eval/run_rsvqa_eval.py --model qwen3-vl:8b --n-per-type 3000
    python src/eval/run_rsvqa_eval.py --model blaifa/InternVL3_5:8b --n-per-type 10004

    # 전체 자동 실행 (master에서 호출)
    python src/eval/run_rsvqa_eval.py --model blaifa/InternVL3_5:8b --n-per-type 10004
"""
import argparse
import base64
import io
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.ollama_vlm import OllamaVLM

RSVQA_DIR   = ROOT / "data/RSVQA-LR"
Q_FILE      = RSVQA_DIR / "LR_split_test_questions.json"
A_FILE      = RSVQA_DIR / "LR_split_test_answers.json"
IMG_DIR     = RSVQA_DIR / "Images_LR"
OUT_DIR     = ROOT / "annotation"

DIRECT_PROMPT = (
    "Look at the image carefully and answer the following question concisely.\n"
    "Question: {query}\n"
    "Answer:"
)

# RSVQA 원본 type → D/M 매핑
TYPE_MAP = {
    "presence":    "D",
    "rural_urban": "D",
    "count":       "M2",
    "comp":        "M2",
}

NUMBER_WORDS = {
    "zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
    "six":"6","seven":"7","eight":"8","nine":"9","ten":"10",
    "eleven":"11","twelve":"12","thirteen":"13","fourteen":"14","fifteen":"15",
    "sixteen":"16","seventeen":"17","eighteen":"18","nineteen":"19","twenty":"20",
    "no":"0","none":"0",
}


# ── 평가기 ────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r'[^\w\s]', ' ', t)
    # 숫자 단어 → 숫자
    for word, num in NUMBER_WORDS.items():
        t = re.sub(rf'\b{word}\b', num, t)
    return ' '.join(t.split())


def evaluate_rsvqa(response: str, gt_answer: str) -> bool:
    """
    RSVQA-LR 답변 평가.
    GT는 단순 문자열("yes", "no", "3", "urban" 등).
    모델 응답의 정규화된 텍스트에 GT가 포함되면 정답.
    """
    resp_norm = normalize(response)
    gt_norm   = normalize(gt_answer)

    # 직접 포함
    if gt_norm in resp_norm:
        return True

    # yes/no 특별 처리: "yes" 답인데 "yes" 또는 "is" 포함
    if gt_norm == "yes":
        return bool(re.search(r'\byes\b|\bthere is\b|\bthere are\b|\bcan see\b', resp_norm))
    if gt_norm == "no":
        # "no" 또는 명확한 부정
        if re.search(r'\bno\b|\bnot\b|\bnone\b|\bcannot\b|\bcan\'t\b', resp_norm):
            # "no" 뒤에 긍정이 오면 제외 (예: "no, there are 3...")
            return bool(re.search(r'^no\b|there (?:is|are) no\b|\bno \w+ (?:visible|present)', resp_norm))

    # 숫자 정확 매칭 (앞뒤 word boundary)
    if re.match(r'^\d+$', gt_norm):
        return bool(re.search(rf'\b{re.escape(gt_norm)}\b', resp_norm))

    return False


# ── 이미지 인코딩 ──────────────────────────────────────────────────────────────

def encode_image(img_id: int) -> str:
    path = IMG_DIR / f"{img_id}.tif"
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


# ── 메인 실행 ──────────────────────────────────────────────────────────────────

def run_eval(model_name: str, n_per_type: int, seed: int = 42):
    model_slug = model_name.replace("/", "_").replace(":", "_")
    out_path = OUT_DIR / f"rsvqa_{model_slug}.json"

    # 데이터 로드
    with open(Q_FILE, encoding="utf-8") as f:
        all_questions = [q for q in json.load(f)["questions"] if q.get("active")]
    with open(A_FILE, encoding="utf-8") as f:
        ans_map = {a["question_id"]: a["answer"]
                   for a in json.load(f)["answers"] if a.get("active")}

    # 타입 그룹핑
    type_groups: dict[str, list] = defaultdict(list)
    for q in all_questions:
        dm_type = TYPE_MAP.get(q.get("type", ""), None)
        if dm_type:
            type_groups[dm_type].append(q)

    # stratified sampling
    rng = random.Random(seed)
    selected = []
    for dm_type, qs in type_groups.items():
        n = min(n_per_type, len(qs)) if n_per_type > 0 else len(qs)
        sampled = rng.sample(qs, n) if n < len(qs) else list(qs)
        for q in sampled:
            selected.append({
                "q_id":      q["id"],
                "img_id":    q["img_id"],
                "question":  q["question"],
                "orig_type": q.get("type", ""),
                "dm_type":   dm_type,
                "gt_answer": ans_map.get(q["id"], ""),
            })

    print(f"모델: {model_name}")
    print(f"평가 대상: {len(selected)}개 "
          f"(D={sum(1 for s in selected if s['dm_type']=='D')}, "
          f"M2={sum(1 for s in selected if s['dm_type']=='M2')})")

    # 체크포인트 로드
    done_ids: set = set()
    results: list[dict] = []
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            ckpt = json.load(f)
        results = ckpt.get("results", [])
        done_ids = {r["q_id"] for r in results}
        print(f"체크포인트: {len(done_ids)}개 이어서 시작")

    remaining = [s for s in selected if s["q_id"] not in done_ids]
    vlm = OllamaVLM(model_name=model_name)
    errors = 0
    t_start = time.monotonic()

    for i, item in enumerate(remaining):
        try:
            img_b64 = encode_image(item["img_id"])
            prompt = DIRECT_PROMPT.format(query=item["question"])
            resp = vlm.generate_b64(prompt, image_b64=img_b64, max_tokens=128)
            answer = resp.text
            correct = evaluate_rsvqa(answer, item["gt_answer"])
        except Exception as e:
            answer = f"ERROR: {e}"
            correct = False
            errors += 1

        results.append({
            "q_id":      item["q_id"],
            "dm_type":   item["dm_type"],
            "orig_type": item["orig_type"],
            "gt_answer": item["gt_answer"],
            "answer":    answer[:150],
            "correct":   correct,
        })

        # 50개마다 저장 + 진행률 출력
        if (i + 1) % 50 == 0 or (i + 1) == len(remaining):
            elapsed = time.monotonic() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_h = (len(remaining) - i - 1) / rate / 3600 if rate > 0 else 0
            _save(out_path, model_name, results)
            correct_so_far = sum(1 for r in results if r["correct"])
            print(f"  [{len(done_ids)+i+1}/{len(selected)}] "
                  f"acc={correct_so_far/len(results)*100:.1f}% "
                  f"errors={errors} "
                  f"speed={rate:.1f}q/s eta={eta_h:.1f}h")

    _save(out_path, model_name, results)
    _print_summary(model_name, results)
    return results


def _save(out_path: Path, model_name: str, results: list[dict]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "results": results}, f,
                  indent=2, ensure_ascii=False)


def _print_summary(model_name: str, results: list[dict]):
    from collections import Counter
    import math

    def wilson_ci(k, n, z=1.96):
        if n == 0:
            return 0, 0
        p = k / n
        center = (p + z**2/(2*n)) / (1 + z**2/n)
        margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / (1 + z**2/n)
        return (center - margin)*100, (center + margin)*100

    print("\n" + "=" * 60)
    print(f"모델: {model_name}")
    print(f"{'Type':8s} {'N':>6} {'Fail%':>7} {'95% CI':>18}")
    print("-" * 60)

    for dm_type in ["D", "M2"]:
        sub = [r for r in results if r["dm_type"] == dm_type]
        n = len(sub)
        fail = sum(1 for r in sub if not r["correct"])
        if n == 0:
            continue
        lo, hi = wilson_ci(fail, n)
        print(f"{dm_type:8s} {n:>6} {fail/n*100:>7.1f}% [{lo:.1f}%, {hi:.1f}%]")

    d_results  = [r for r in results if r["dm_type"] == "D"]
    m2_results = [r for r in results if r["dm_type"] == "M2"]
    if d_results and m2_results:
        d_fail  = sum(1 for r in d_results  if not r["correct"]) / len(d_results)
        m2_fail = sum(1 for r in m2_results if not r["correct"]) / len(m2_results)
        gap = (m2_fail - d_fail) * 100
        print(f"\nD/M2 Gap: {gap:+.1f} pp")

        # z-test
        from scipy.stats import norm
        n_d, k_d   = len(d_results),  sum(1 for r in d_results  if not r["correct"])
        n_m, k_m   = len(m2_results), sum(1 for r in m2_results if not r["correct"])
        p_pool = (k_d + k_m) / (n_d + n_m)
        se = math.sqrt(p_pool*(1-p_pool)*(1/n_d + 1/n_m))
        z_stat = (m2_fail - d_fail) / se if se > 0 else 0
        p_val = 1 - norm.cdf(z_stat)
        print(f"z={z_stat:.2f}, p={p_val:.4f} (one-sided)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-per-type", type=int, default=0,
                        help="타입당 최대 샘플 수 (0=전수)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_eval(args.model, args.n_per_type, args.seed)


if __name__ == "__main__":
    main()
