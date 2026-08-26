"""
GSD-Sensitivity Taxonomy Classifier
ThinkGeo 태스크 쿼리를 D / M1 / M2로 분류하는 규칙 기반 분류기.

분류 기준:
    - 출력 연산 기준으로 분류 (쿼리 주제/필터 무관)
    - 비교·최상급은 M1(GSD 기반) 또는 M2(카운트 기반)에 흡수
    - D와 M은 두 독립 출력이 동시에 요구될 때만 공존
"""

import re
from dataclasses import dataclass
from typing import Optional

try:
    from .definitions import TaskType, TOOL_TO_TAXONOMY_HINT
except ImportError:
    from definitions import TaskType, TOOL_TO_TAXONOMY_HINT


@dataclass
class ClassificationResult:
    all_types: list[TaskType]
    confidence: str          # "high" | "medium" | "low"
    evidence: list[str]
    notes: Optional[str] = None
    reviewed: bool = False


# ── M1: 공간 메트릭 ──────────────────────────────────────────────────────────
# GSD가 명시되거나 실측 단위 공간 조건이 있을 때
_M1_STRONG = [
    # GSD 명시
    r"\bgsd\b",
    r"\bm/px\b",
    r"\bmeters?\s+per\s+pixel\b",
    # 숫자 + 단위
    r"\b\d+\.?\d*\s*(?:meter|metre|km|kilometer|feet|foot|ft)s?\b",
    r"\bwithin\s+\d+\.?\d*\s*(?:m|km|ft|meter|metre)\b",
    # 단위 명시 거리/면적 질문
    r"\bdistance\s+(?:between|from|to)\b.*\bgsd\b",
    r"\bhow\s+(?:far|long|wide|tall|deep|high)\s+(?:is|are|does)\b",
    r"\barea\s+in\s+(?:square\s+)?(?:meter|metre|km|feet)\b",
    r"\bsquare\s+(?:meter|metre|km|feet)\b",
    r"\bin\s+(?:meter|metre|km|feet)s?\b",
]

# GSD 없어도 거리 계산이 명시적으로 요구되는 패턴
_M1_MEDIUM = [
    r"\bdistance\s+between\b",
    r"\bhow\s+far\s+(?:is|are|apart)\b",
    r"\bhow\s+long\s+(?:is|are)\b.*\b(?:road|runway|lane|path|route)\b",
    r"\barea\s+of\s+the\b",
    r"\b(?:length|width)\s+of\s+the\b",
    # Additional colloquial spatial measurement phrases
    r"\bmeasure\s+the\b",  # "measure the runway", "measure the area"
    r"\bspan\s+of\s+(?:the\s+)?(?:bridge|road|runway|river|building)\b",
    r"\bhow\s+(?:long|wide|tall|large|big|small)\s+is\s+(?:the\s+)?\w",
    r"\bhow\s+many\s+(?:meter|metre|km|kilometer|feet|foot|mile)s?\b",  # "how many meters"
]

# ── M2: 카디널리티 ─────────────────────────────────────────────────────────────
# 카운팅, 비율, 비교(카운트 기반), 최상급(카운트 기반) 모두 포함
_M2_PATTERNS = [
    r"\bhow\s+many\b",
    r"\bcount\s+(?:the\s+)?(?:number\s+of\s+)?\w",
    r"\bnumber\s+of\b",
    r"\btotal\s+(?:number|count)\b",
    r"\bhow\s+much\b(?!.*\barea\b)",  # "how much area" 제외
    r"\benumerate\b",
    r"\btally\b",
    r"\bcount\s+(?:all|each|every)\b",
    # 카운트 기반 비교·최상급 (M5에서 흡수)
    r"\bwhich\s+(?:side|half|quadrant|zone|area|region|part|section)\s+has\s+(?:more|less|most|least|greater|fewer)\b",
    r"\bwhich\s+(?:\w+\s+)?has\s+(?:the\s+)?(?:most|fewest|highest\s+count|lowest\s+count)\b",
    r"\bratio\s+of\b",
    r"\bpercentage\s+of\b",
    r"\bproportion\s+of\b",
    r"\bare\s+there\s+(?:more|fewer|less)\b",
    r"\bmore\s+(?:cars?|buildings?|houses?|vehicles?|planes?|aircraft|pools?|trees?)\s+than\b",
    r"\b(?:equal|same)\s+number\s+of\b",
    r"\bnumber\s+of\b.{0,50}\bnumber\s+of\b",  # "number of X ... number of Y"
    # Additional colloquial counting phrases
    r"\bgive\s+(?:me\s+)?(?:the\s+)?total\b",
    r"\blist\s+(?:all|every|each)\b",
    r"\bidentify\s+all\b.{0,30}\b(?:and\s+)?(?:give|state|provide|report)\b",
    r"\bwhich\s+(?:\w+\s+){0,4}(?:is\s+)?(?:busier|more\s+crowded|denser)\b",
    r"\bbusier\b",
]


class TaxonomyClassifier:
    """
    규칙 기반 Taxonomy 분류기.

    분류 우선순위: M1 > M2 > D
    복수 유형이 감지되면 all_types에 모두 기록.
    """

    def classify(
        self,
        query: str,
        tools: Optional[list[str]] = None,
    ) -> ClassificationResult:
        q = query.lower()
        matched: dict[TaskType, list[str]] = {}

        # M1: strong 먼저, medium은 보조
        m1_hits = self._match(q, _M1_STRONG)
        if not m1_hits:
            m1_hits = self._match(q, _M1_MEDIUM)
            if m1_hits:
                m1_hits = [f"(medium){h}" for h in m1_hits]
        if m1_hits:
            matched[TaskType.M1] = m1_hits

        # M2
        m2_hits = self._match(q, _M2_PATTERNS)
        if m2_hits:
            matched[TaskType.M2] = m2_hits

        # 도구 힌트 (보조)
        tool_hints = self._get_tool_hints(tools or [])

        if not matched:
            # 도구 힌트로만 M 유형 보강 (낮은 신뢰도)
            m_hints = [tt for tt in [TaskType.M1, TaskType.M2]
                       if tt in tool_hints]
            if m_hints and ("Calculator" in (tools or []) or "Solver" in (tools or [])):
                return ClassificationResult(
                    all_types=m_hints,
                    confidence="low",
                    evidence=[f"tool:{t}" for t in (tools or []) if t in ("Calculator", "Solver", "CountGivenObject")],
                    notes="키워드 미검출, Calculator/Solver 사용 → 수동 검토 권장",
                )
            return ClassificationResult(
                all_types=[TaskType.D],
                confidence="high",
                evidence=["no_measurement_keyword"],
            )

        # 우선순위 순서로 정렬
        order = [TaskType.M1, TaskType.M2]
        all_types = [tt for tt in order if tt in matched]
        evidence = []
        for tt in all_types:
            evidence.extend(matched[tt])

        n_matches = sum(len(v) for v in matched.values())
        confidence = "high" if n_matches >= 2 else "medium"

        return ClassificationResult(
            all_types=all_types,
            confidence=confidence,
            evidence=evidence[:6],
        )

    def classify_batch(
        self,
        tasks: dict[str, dict],
    ) -> dict[str, ClassificationResult]:
        results = {}
        for task_id, task in tasks.items():
            results[task_id] = self.classify(task.get("query", ""), task.get("tools", []))
        return results

    @staticmethod
    def _match(text: str, patterns: list[str]) -> list[str]:
        hits = []
        for pat in patterns:
            if re.search(pat, text):
                hits.append(pat)
        return hits

    @staticmethod
    def _get_tool_hints(tools: list[str]) -> set[TaskType]:
        hints: set[TaskType] = set()
        for t in tools:
            for hint_type in TOOL_TO_TAXONOMY_HINT.get(t, []):
                hints.add(hint_type)
        return hints
