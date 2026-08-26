"""
수동 Taxonomy 레이블 검토 도구 (CLI).

불일치 케이스를 하나씩 보여주고 사용자가 정답 레이블을 입력한다.
진행 상황은 자동 저장되므로 중간에 종료해도 이어서 할 수 있다.

사용법:
    cd <repo root>
    python src/router/review_tool.py
    python src/router/review_tool.py --show-all   # 전체 현황 보기
    python src/router/review_tool.py --summary    # 완료 후 결과 요약
"""

import json
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DIS_PATH = ROOT / "annotation/router_disagreements.json"
LABEL_DESC = {
    "D":  "Description    — 시각적 해석만으로 답 가능, GSD 무관",
    "M1": "Spatial Metric — 실제 거리/면적 (GSD 필요)",
    "M2": "Cardinality    — 정밀 객체 카운팅",
    "M3": "Geometric      — 방향각·헤딩·형태 계산",
    "M4": "Threshold      — 침수·피해 등 도메인 임계값 판단",
    "M5": "Comparison     — M1-M4 결과 간 비교",
}
VALID = set(LABEL_DESC.keys())


def load() -> list[dict]:
    with open(DIS_PATH, encoding="utf-8") as f:
        return json.load(f)


def save(data: list[dict]):
    with open(DIS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _to_set(label) -> set:
    """manual_label/rule_pred/llm_pred를 set으로 정규화 (str 또는 list 모두 허용)."""
    if isinstance(label, list):
        return set(label)
    if isinstance(label, str):
        return {label}
    return set()


def show_summary(data: list[dict]):
    total = len(data)
    done = sum(1 for d in data if d["manual_label"])
    print(f"\n진행: {done}/{total}  ({done/total:.0%})")
    if done == total:
        print("모든 케이스 검토 완료!")
        # Exact Match 기준 정확도
        rule_ok = sum(1 for d in data if _to_set(d["rule_pred"]) == _to_set(d["manual_label"]))
        llm_ok  = sum(1 for d in data if _to_set(d["llm_pred"])  == _to_set(d["manual_label"]))
        print(f"  Rule-based Exact Match (불일치 케이스): {rule_ok}/{done} = {rule_ok/done:.1%}")
        print(f"  LLM Exact Match        (불일치 케이스): {llm_ok}/{done} = {llm_ok/done:.1%}")


def show_all(data: list[dict]):
    print(f"\n{'ID':>6}  {'Rule':<10}  {'LLM':<10}  {'Current':<12}  {'Manual':<12}  Query")
    print("-" * 90)
    for d in data:
        rule = ",".join(_to_set(d["rule_pred"])) if d.get("rule_pred") else "?"
        llm  = ",".join(_to_set(d["llm_pred"]))  if d.get("llm_pred")  else "?"
        cur  = ",".join(_to_set(d["current_label"])) if d.get("current_label") else "?"
        ml   = ",".join(_to_set(d["manual_label"])) if d.get("manual_label") else "?"
        q    = d["query"][:45]
        print(f"  {d['task_id']:>4}  {rule:<10}  {llm:<10}  {cur:<12}  {ml:<12}  {q}")


def review_loop(data: list[dict]):
    pending = [d for d in data if not d["manual_label"]]
    total   = len(data)
    done    = total - len(pending)

    print(f"\n수동 레이블 검토 시작 (남은 케이스: {len(pending)}개)")
    print("입력: D / M1 / M1,M4 (복수 가능, 쉼표 구분)  |  s=skip  |  q=저장 후 종료\n")

    for i, entry in enumerate(pending):
        idx = done + i + 1
        rule = ",".join(_to_set(entry.get("rule_pred", [])))
        llm  = ",".join(_to_set(entry.get("llm_pred", [])))
        cur  = ",".join(_to_set(entry.get("current_label", [])))
        print(f"\n[{idx}/{total}]  Task ID: {entry['task_id']}")
        print(f"  Query  : {entry['query']}")
        print(f"  Rule   : {rule}   LLM: {llm}   현재 레이블: {cur}")
        print("  레이블 설명:")
        for k, v in LABEL_DESC.items():
            print(f"    {k}  {v}")

        while True:
            ans = input("  → 정답 (복수 시 쉼표 구분, 예: M1,M4): ").strip().upper()
            if ans == "Q":
                save(data)
                print("저장 후 종료.")
                show_summary(data)
                return
            if ans == "S":
                break
            tokens = [t.strip() for t in ans.split(",")]
            if all(t in VALID for t in tokens) and tokens:
                entry["manual_label"] = tokens if len(tokens) > 1 else tokens[0]
                break
            print(f"  유효하지 않은 입력. {'/'.join(VALID)} 중에서 쉼표로 구분해 입력하세요.")

        save(data)  # 매 케이스마다 저장

    show_summary(data)
    _apply_to_labels(data)


def _apply_to_labels(data: list[dict]):
    """수동 레이블을 annotation/thinkgeo_taxonomy_labels.json에 반영."""
    labels_path = ROOT / "annotation/thinkgeo_taxonomy_labels.json"
    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)

    updated = 0
    for entry in data:
        if entry["manual_label"] and entry["task_id"] in labels:
            ml = entry["manual_label"]
            labels[entry["task_id"]]["annotation"]["types"] = ml if isinstance(ml, list) else [ml]
            labels[entry["task_id"]]["annotation"]["reviewed"] = True
            updated += 1

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    print(f"\n레이블 파일 업데이트: {updated}개 반영 → {labels_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-all", action="store_true")
    parser.add_argument("--summary",  action="store_true")
    parser.add_argument("--apply",    action="store_true", help="수동 레이블을 labels 파일에 반영")
    args = parser.parse_args()

    if not DIS_PATH.exists():
        print(f"불일치 파일 없음: {DIS_PATH}")
        print("먼저 eval_router.py --router both 를 실행하세요.")
        sys.exit(1)

    data = load()
    print(f"불일치 케이스 로드: {len(data)}개")

    if args.show_all:
        show_all(data)
        show_summary(data)
        return

    if args.summary:
        show_summary(data)
        return

    if args.apply:
        _apply_to_labels(data)
        return

    review_loop(data)


if __name__ == "__main__":
    main()
