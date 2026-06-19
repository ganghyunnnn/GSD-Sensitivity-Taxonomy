"""
ThinkGeo whitelist/blacklist 기반 답변 평가기.

gt_answer 형식:
    {
      "whitelist": [["term1a", "term1b"], ["term2a"]],  # 각 리스트에서 최소 1개 매칭
      "blacklist": ["bad_term"]  # 없거나 null
    }

evaluation 형식:
    [{"question": "...", "answer": "Yes/No"}]
"""

import re


def normalize(text: str) -> str:
    """소문자 변환 + 구두점 제거."""
    return re.sub(r'[^\w\s]', ' ', text.lower())


def check_whitelist(answer: str, whitelist: list[list[str]]) -> bool:
    """
    whitelist의 모든 그룹에서 적어도 하나의 term이 answer에 포함되어야 True.
    """
    if not whitelist:
        return True
    ans_norm = normalize(answer)
    for group in whitelist:
        group_matched = any(normalize(term) in ans_norm for term in group)
        if not group_matched:
            return False
    return True


def check_blacklist(answer: str, blacklist: list | None) -> bool:
    """blacklist에 포함된 term이 answer에 없어야 True.
    blacklist는 str 목록 또는 str 목록의 목록일 수 있다."""
    if not blacklist:
        return True
    ans_norm = normalize(answer)
    for item in blacklist:
        # item이 list인 경우: 그 중 하나라도 매칭되면 blacklist hit
        if isinstance(item, list):
            if any(normalize(str(t)) in ans_norm for t in item):
                return False
        else:
            if normalize(str(item)) in ans_norm:
                return False
    return True


def evaluate_answer(
    model_answer: str,
    gt_answer: dict | list,
) -> dict:
    """
    returns: {
      "correct": bool,
      "whitelist_pass": bool,
      "blacklist_pass": bool,
    }
    """
    # gt_answer가 list인 경우: 가능한 정답 문자열 목록
    if isinstance(gt_answer, list):
        ans_norm = normalize(model_answer)
        correct = any(normalize(str(gt)) in ans_norm for gt in gt_answer)
        return {"correct": correct, "whitelist_pass": correct, "blacklist_pass": True}

    whitelist = gt_answer.get("whitelist") or []
    blacklist = gt_answer.get("blacklist")

    wl_pass = check_whitelist(model_answer, whitelist)
    bl_pass = check_blacklist(model_answer, blacklist)
    correct = wl_pass and bl_pass

    return {
        "correct": correct,
        "whitelist_pass": wl_pass,
        "blacklist_pass": bl_pass,
    }
