"""
Hybrid Task Router.

전략:
  1. Rule-based Router 먼저 실행
  2. 패턴으로 불확실하면 LLM(arbiter)에 D/M1/M2 분류 의뢰
  3. LLM 결과로 override (LLM이 None이면 Rule 결과만 사용)
"""

import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from taxonomy.definitions import TaskType
from router.rule_router import RuleRouter

# LLM에 최종 분류를 판정하도록 하는 경량 프롬프트
_ARBITER_SYSTEM = """You are a Remote Sensing task classifier.

Taxonomy:
  D  - Description: answerable by visual/semantic interpretation alone. GSD does NOT affect the answer.
       Covers: scene description, object presence/category, bbox output, color/shape, heading/orientation,
               visual yes/no, qualitative comparison, qualitative spatial relation.
  M1 - Spatial Metric: GSD-dependent spatial computation required.
       Covers: real-world distance/area output, or spatial criterion in real-world units evaluated with GSD.
               Comparisons/superlatives based on GSD measurement are M1.
  M2 - Cardinality: precise counting of discrete objects required.
       Covers: counts, percentages, ratios, arithmetic derived from counts.
               Comparisons/superlatives based on counting are M2.

Rules:
  - Classify by output operation, not by subject/filter.
  - D and M types co-occur ONLY when two independent outputs are required.
  - M1+M2 when both GSD computation and counting are independently required.

Answer with one or more labels separated by commas: D, M1, M2. Nothing else."""

_ARBITER_USER = 'Classify this RS query. Output labels only (e.g. "M2" or "M1,M2").\nQuery: "{query}"'


class HybridRouter:
    """
    Rule-based + optional LLM arbiter.

    Parameters
    ----------
    llm : TextLLMBase or None
        None이면 LLM arbiter 없이 Rule Router 결과만 사용.
    """

    def __init__(self, llm=None):
        self._rule = RuleRouter()
        self._llm = llm
        if llm is not None:
            self._orig_system = getattr(llm, "system_prompt", "")

    def route(self, query: str) -> TaskType:
        rule_result = self._rule.route(query)

        if self._llm is not None:
            llm_result = self._ask_llm(query)
            if llm_result is not None:
                return llm_result

        return rule_result

    def _ask_llm(self, query: str) -> TaskType | None:
        self._llm.system_prompt = _ARBITER_SYSTEM
        prompt = _ARBITER_USER.format(query=query)
        try:
            resp = self._llm.generate(prompt, max_tokens=32, temperature=0.0)
            text = resp.text.strip().upper()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            # 첫 번째 유효 레이블 반환 (멀티레이블 지원은 route_multi 사용)
            match = re.search(r"\b(M1|M2|D)\b", text)
            if match:
                return TaskType(match.group(1))
        except Exception:
            pass
        finally:
            self._llm.system_prompt = self._orig_system
        return None

    def route_multi(self, query: str) -> list[TaskType]:
        """멀티레이블 분류 (LLM 사용 시)."""
        if self._llm is None:
            result = self._rule.route(query)
            return [result]

        self._llm.system_prompt = _ARBITER_SYSTEM
        prompt = _ARBITER_USER.format(query=query)
        try:
            resp = self._llm.generate(prompt, max_tokens=32, temperature=0.0)
            text = resp.text.strip().upper()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            labels = re.findall(r"\b(M1|M2|D)\b", text)
            if labels:
                # 중복 제거, 우선순위 순 정렬
                order = [TaskType.M1, TaskType.M2, TaskType.D]
                unique = list(dict.fromkeys(TaskType(l) for l in labels))
                return sorted(unique, key=lambda t: order.index(t) if t in order else 99)
        except Exception:
            pass
        finally:
            self._llm.system_prompt = self._orig_system

        result = self._rule.route(query)
        return [result]
