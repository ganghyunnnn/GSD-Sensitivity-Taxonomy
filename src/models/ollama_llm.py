"""
Ollama 기반 텍스트 LLM 래퍼.
Task Router 전용: qwen3:8b
"""

import time
import requests
from .base import TextLLMBase, VLMResponse

OLLAMA_BASE = "http://localhost:11434"


class OllamaLLM(TextLLMBase):
    def __init__(self, model_name: str = "qwen3:8b", base_url: str = OLLAMA_BASE,
                 system_prompt: str = "", think: bool = False):
        self.model_name = model_name
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.think = think

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> VLMResponse:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "think": self.think,     # thinking 모드 (gpt-oss:20b는 True 필요)
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        t0 = time.monotonic()
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        latency_ms = int((time.monotonic() - t0) * 1000)

        data = resp.json()
        return VLMResponse(
            text=data["message"]["content"].strip(),
            model=self.model_name,
            latency_ms=latency_ms,
        )
