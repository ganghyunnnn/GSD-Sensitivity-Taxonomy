"""
Ollama 기반 VLM 래퍼.
지원 모델: qwen2.5vl:7b, llama3.2-vision:11b, qwen3.5:9b (thinking 모드 포함)
"""

import time
import base64
from pathlib import Path
from typing import Optional

import requests

from .base import VLMBase, VLMResponse

OLLAMA_BASE = "http://localhost:11434"


def _encode_image(image_path: str | Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class OllamaVLM(VLMBase):
    def __init__(self, model_name: str, base_url: str = OLLAMA_BASE):
        self.model_name = model_name
        self.base_url = base_url

    def generate(
        self,
        prompt: str,
        image_path: Optional[str | Path] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> VLMResponse:
        payload: dict = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_gpu": 99,   # 모든 레이어를 GPU에 올림
            },
        }
        if image_path is not None:
            payload["images"] = [_encode_image(image_path)]

        t0 = time.monotonic()
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=600,
        )
        resp.raise_for_status()
        latency_ms = int((time.monotonic() - t0) * 1000)

        data = resp.json()
        text = data.get("response", "").strip()

        # thinking 모델(qwen3.5 등) 대응: response가 비어있으면 chat API로 재시도
        if not text:
            chat_payload: dict = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": payload["prompt"]}],
                "stream": False,
                "options": {**payload.get("options", {}), "num_predict": 2048},
            }
            if "images" in payload:
                chat_payload["messages"][0]["images"] = payload["images"]
            chat_resp = requests.post(
                f"{self.base_url}/api/chat",
                json=chat_payload,
                timeout=600,
            )
            chat_resp.raise_for_status()
            chat_data = chat_resp.json()
            msg = chat_data.get("message", {})
            text = msg.get("content", "").strip()
            # thinking 필드에만 있는 경우 fallback
            if not text:
                text = msg.get("thinking", "").strip()

        return VLMResponse(
            text=text,
            model=self.model_name,
            latency_ms=latency_ms,
        )

    def generate_b64(
        self,
        prompt: str,
        image_b64: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> VLMResponse:
        """이미 base64로 인코딩된 이미지를 받아 생성. JPEG 사전변환 시 사용."""
        payload: dict = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_gpu": 99,
            },
        }
        t0 = time.monotonic()
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=600,
        )
        resp.raise_for_status()
        latency_ms = int((time.monotonic() - t0) * 1000)
        data = resp.json()
        text = data.get("response", "").strip()

        if not text:
            chat_payload: dict = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
                "stream": False,
                "options": {**payload.get("options", {}), "num_predict": 2048},
            }
            chat_resp = requests.post(
                f"{self.base_url}/api/chat",
                json=chat_payload,
                timeout=600,
            )
            chat_resp.raise_for_status()
            chat_data = chat_resp.json()
            msg = chat_data.get("message", {})
            text = msg.get("content", "").strip() or msg.get("thinking", "").strip()

        return VLMResponse(text=text, model=self.model_name, latency_ms=latency_ms)

    @classmethod
    def available_models(cls, base_url: str = OLLAMA_BASE) -> list[str]:
        resp = requests.get(f"{base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
