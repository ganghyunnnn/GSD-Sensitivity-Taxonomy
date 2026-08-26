"""
GSD-Sensitivity-Taxonomy 모델 인터페이스 추상 기반 클래스.
Ollama / HuggingFace 백엔드를 동일하게 사용할 수 있도록 추상화.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VLMResponse:
    text: str
    model: str
    latency_ms: int


class VLMBase(ABC):
    """비전-언어 모델 (이미지 + 텍스트 → 텍스트)"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        image_path: Optional[str | Path] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> VLMResponse:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={getattr(self, 'model_name', '?')})"


class TextLLMBase(ABC):
    """텍스트 전용 LLM (텍스트 → 텍스트)"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> VLMResponse:
        ...
