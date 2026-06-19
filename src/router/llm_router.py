"""
LLM 기반 Task Router (zero-shot).
taxonomy 정의를 프롬프트에 주입하고 D / M1 / M2 를 반환한다.
멀티레이블 지원: route_multi()는 list[TaskType]을 반환한다.
"""

import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from taxonomy.definitions import TaskType

SYSTEM_PROMPT = """You are a Remote Sensing VQA task classifier. Classify the given query into one or more of:

D  - Description: answerable by visual/semantic interpretation alone. GSD does NOT affect the answer.
     Covers: scene description, object presence/category, bounding box output, color/shape,
             heading/orientation/direction, visual yes/no, qualitative comparison, qualitative spatial relation.
     Examples: "What land type is this?", "Draw bbox around the cars.", "What is the heading of the aircraft?"

M1 - Spatial Metric: GSD-dependent spatial computation required at any step.
     (a) Output is a real-world measurement (distance, area, length in real-world units), OR
     (b) A spatial criterion in real-world units must be evaluated using GSD.
     Comparisons/superlatives based on GSD measurement are M1.
     Examples: "Distance between A and B? GSD=0.3m/px", "Is there a house within 50m? GSD=x",
               "Which parking lot is larger? GSD=x"

M2 - Cardinality: precise counting of discrete objects required.
     Covers: counts, percentages, ratios, arithmetic derived from counts.
     Comparisons/superlatives based on counting are M2.
     Examples: "How many cars?", "Which side has more buildings?", "Ratio of damaged to undamaged buildings?"

Rules:
- Classify by OUTPUT OPERATION, not by subject or filter.
- D and M co-occur ONLY when two independent outputs are explicitly required.
- M1+M2 when both GSD computation AND counting are independently required.
- Output labels only, separated by commas if multiple: e.g. "M2" or "M1,M2" or "D,M2"
- Valid labels: D, M1, M2. Nothing else."""

USER_TEMPLATE = 'Classify this RS query. Output label(s) only (D/M1/M2, comma-separated if multiple).\nQuery: "{query}"'


class LLMRouter:
    def __init__(self, llm, think: bool = False):
        """
        llm: TextLLMBase 인스턴스 (OllamaLLM 등)
        think: qwen3의 thinking mode 사용 여부
        """
        self._llm = llm
        self._think = think
        self._llm.system_prompt = SYSTEM_PROMPT

    def route(self, query: str) -> TaskType:
        """단일 레이블 반환 (우선순위: M1 > M2 > D)."""
        types = self.route_multi(query)
        order = [TaskType.M1, TaskType.M2, TaskType.D]
        for t in order:
            if t in types:
                return t
        return TaskType.D

    def route_multi(self, query: str) -> list[TaskType]:
        """멀티레이블 반환."""
        prompt = USER_TEMPLATE.format(query=query)
        resp = self._llm.generate(prompt, max_tokens=512, temperature=0.0)
        return self._parse_multi(resp.text)

    def route_batch(self, queries: list[str]) -> list[TaskType]:
        return [self.route(q) for q in queries]

    def route_batch_multi(self, queries: list[str]) -> list[list[TaskType]]:
        return [self.route_multi(q) for q in queries]

    @staticmethod
    def _parse_multi(text: str) -> list[TaskType]:
        text = text.strip().upper()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        labels = re.findall(r"\b(M1|M2|D)\b", text)
        if not labels:
            return [TaskType.D]
        order = [TaskType.M1, TaskType.M2, TaskType.D]
        unique = list(dict.fromkeys(TaskType(l) for l in labels))
        return sorted(unique, key=lambda t: order.index(t) if t in order else 99)

    @staticmethod
    def _parse(text: str) -> TaskType:
        """단일 레이블 파싱 (하위 호환)."""
        labels = LLMRouter._parse_multi(text)
        return labels[0] if labels else TaskType.D
