"""
GSD-Sensitivity Classifier — Algorithm 1 구현체

분류체계를 형식적 결정 규칙의 집합으로 정의.
이 파일이 논문 Algorithm 1 박스에 대응하는 참조 구현.

우선순위:
  R1: M1 — GSD 명시 (결정론적)
  R2: M1 — 실측 단위 공간 임계값 (결정론적)
  R3: M2 — 카디널리티 (결정론적)
  R4: D  — 기본값 (GSD-invariant)

비교·최상급:
  GSD 기반 비교 → M1 (R1/R2에서 처리)
  카운트 기반 비교 → M2 (R3에서 처리)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskType(str, Enum):
    D  = "D"
    M1 = "M1"
    M2 = "M2"


@dataclass
class ClassificationResult:
    task_type: TaskType
    confidence: str           # "deterministic" | "heuristic" | "edge_case"
    matched_rule: str         # 어떤 규칙이 매칭됐는지
    raw_query: str
    rule_id: str = ""         # R1~R4


# ── R1: M1 — GSD 명시 ─────────────────────────────────────────────────────────
_M1_GSD_PATTERNS = [
    (r'\bgsd\b',                                                          "M1_gsd_keyword"),
    (r'\bm\s*/\s*px\b',                                                   "M1_m_per_px"),
    (r'[\d.]+\s*(meter|metre|m)\s*/\s*px',                               "M1_numeric_m_per_px"),
    (r'\bground\s*sampling\s*distance\b',                                 "M1_gsd_full"),
    (r'\bresolution\s+of\s+[\d.]+\s*(m|cm|meter|metre)\b',               "M1_resolution_value"),
    (r'[\d.]+\s*(m|cm|meter|metre)\s*/\s*pixel',                         "M1_m_per_pixel"),
]

# ── R2: M1 — 실측 단위 공간 임계값 ────────────────────────────────────────────
_M1_SPATIAL_PATTERNS = [
    (r'\bwithin\s+[\d.]+\s*(m|km|meter|metre|feet|ft|kilometer)\b',      "M1_within_dist"),
    (r'[\d.]+\s*(m|km|meter|metre|feet|ft)\s+(away|from|radius|buffer)', "M1_dist_away"),
    (r'\bdistance\s+between\b',                                           "M1_distance_between"),
    (r'\bhow\s+(far|wide|long|tall|high|deep)\b',                        "M1_how_far"),
    (r'\barea\s+(in\s+)?(square\s+)?(m|km|meter|metre|hectare|acre)\b',  "M1_area_unit"),
    (r'\b(length|width|height)\s+of\s+the\b',                            "M1_dimension"),
    (r'\bin\s+(meters?|metres?|km|feet|kilometers?)\b',                   "M1_in_unit"),
    (r'\b(square\s+)?(meter|metre|km|hectare|acre)s?\b',                 "M1_area_keyword"),
]

# ── R3: M2 Cardinality ────────────────────────────────────────────────────────
# 카운팅, 비율, 카운트 기반 비교·최상급 포함
_M2_PATTERNS = [
    (r'\bhow\s+many\b',                                                   "M2_how_many"),
    (r'\bcount\s+(the\s+)?(number\s+of\s+)?\w',                         "M2_count"),
    (r'\bnumber\s+of\b',                                                  "M2_number_of"),
    (r'\btotal\s+(number|count)\b',                                       "M2_total_count"),
    (r'\benumerate\b',                                                     "M2_enumerate"),
    (r'\bquantify\b',                                                      "M2_quantify"),
    (r'\btally\b',                                                         "M2_tally"),
    (r'\bhow\s+much\b(?!.{0,20}\barea\b)',                               "M2_how_much"),
    (r'\bcount\s+(all|each|every)\b',                                     "M2_count_all"),
    (r'\bamount\s+of\b',                                                   "M2_amount_of"),
    # 카운트 기반 비교·최상급 (구 M5에서 흡수)
    (r'\bare\s+there\s+(more|less|fewer|greater)\b',                      "M2_are_there_more"),
    (r'\bnumber\s+of\b.{0,50}\bnumber\s+of\b',                          "M2_number_equal"),
    (r'\b(amount|number|count)\s+of\b.{0,40}\b(greater|less|more|fewer|equal)\b', "M2_amount_compare"),
    (r'\bwhich\b.{0,40}\b(more|less|greater|fewer|most|fewest)\b',       "M2_which_more"),
    (r'\bwhich\s+(side|half|zone|region|area|lot|section|part)\b.{0,30}\b(has|have)\b', "M2_which_side"),
    (r'\bratio\s+of\b',                                                    "M2_ratio"),
    (r'\bpercentage\s+of\b',                                               "M2_percentage"),
    (r'\b(more|less)\s+.{0,20}\bthan\b.{0,30}\b(the other|another|the left|the right)', "M2_more_than"),
    (r'\bmore\s+(cars?|buildings?|vehicles?|aircraft|pools?|trees?)\s+than\b', "M2_more_X_than"),
    (r'\b(equal|same|identical)\s+.{0,20}\b(number|count|amount)\b',     "M2_equal_count"),
]


class GSDSensitivityClassifier:
    """
    Algorithm 1: GSD-Sensitivity Classification Rule Cascade.

    우선순위 (논문 기준):
      R1 M1(GSD) → R2 M1(spatial) → R3 M2 → R4 D
    """

    def classify(self, query: str) -> ClassificationResult:
        q = query.lower().strip()

        # R1: M1 — GSD 명시 (결정론적)
        for pat, rule_id in _M1_GSD_PATTERNS:
            if re.search(pat, q):
                return ClassificationResult(TaskType.M1, "deterministic", pat, query, rule_id)

        # R2: M1 — 실측 단위 공간 임계값 (결정론적)
        for pat, rule_id in _M1_SPATIAL_PATTERNS:
            if re.search(pat, q):
                return ClassificationResult(TaskType.M1, "deterministic", pat, query, rule_id)

        # R3: M2
        for pat, rule_id in _M2_PATTERNS:
            if re.search(pat, q):
                return ClassificationResult(TaskType.M2, "deterministic", pat, query, rule_id)

        # R4: D (default)
        return ClassificationResult(TaskType.D, "heuristic", "D:default", query, "R4_default")

    def classify_batch(self, queries: list[str]) -> list[ClassificationResult]:
        return [self.classify(q) for q in queries]

    def coverage_stats(self, queries: list[str]) -> dict:
        """결정론적 분류 커버리지 통계."""
        results = self.classify_batch(queries)
        deterministic = sum(1 for r in results if r.confidence == "deterministic")
        from collections import Counter
        type_counts = Counter(r.task_type for r in results)
        return {
            "total": len(results),
            "deterministic": deterministic,
            "coverage": deterministic / len(results) if results else 0,
            "type_distribution": dict(type_counts),
            "results": results,
        }
