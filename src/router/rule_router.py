"""기존 규칙 기반 분류기를 Router 인터페이스로 감싼 래퍼."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from taxonomy.classifier import TaxonomyClassifier, ClassificationResult
from taxonomy.definitions import TaskType


class RuleRouter:
    def __init__(self):
        self._clf = TaxonomyClassifier()

    def route(self, query: str) -> list[TaskType]:
        result: ClassificationResult = self._clf.classify(query)
        return result.all_types

    def route_with_detail(self, query: str) -> ClassificationResult:
        return self._clf.classify(query)
