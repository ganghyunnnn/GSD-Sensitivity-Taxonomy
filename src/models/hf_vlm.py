"""
HuggingFace 기반 VLM 래퍼.
지원 모델: InternVL2_5-8B
"""

import time
from pathlib import Path
from typing import Optional

from .base import VLMBase, VLMResponse


class HuggingFaceVLM(VLMBase):
    """
    Lazy-loading HuggingFace VLM.
    첫 generate() 호출 시 모델 로드 (24GB VRAM 고려).
    """

    def __init__(
        self,
        model_name: str,
        load_in_4bit: bool = True,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.device = device
        self._model = None
        self._tokenizer = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer, AutoProcessor

        print(f"[HuggingFaceVLM] Loading {self.model_name} ...")

        if "InternVL" in self.model_name:
            self._load_internvl()
        else:
            raise NotImplementedError(f"HF model not supported: {self.model_name}")

    def _load_internvl(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            kwargs["device_map"] = "auto"

        self._model = AutoModel.from_pretrained(self.model_name, **kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if not self.load_in_4bit:
            self._model = self._model.cuda()
        self._model.eval()
        print(f"[HuggingFaceVLM] {self.model_name} loaded.")

    def generate(
        self,
        prompt: str,
        image_path: Optional[str | Path] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> VLMResponse:
        self._load()

        if "InternVL" in self.model_name:
            return self._generate_internvl(prompt, image_path, max_tokens, temperature)
        raise NotImplementedError

    def _generate_internvl(
        self,
        prompt: str,
        image_path: Optional[str | Path],
        max_tokens: int,
        temperature: float,
    ) -> VLMResponse:
        import torch
        from PIL import Image

        generation_config = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
        }
        generation_config = {k: v for k, v in generation_config.items() if v is not None}

        if image_path is not None:
            from torchvision import transforms

            image = Image.open(image_path).convert("RGB")
            # InternVL 전처리
            transform = transforms.Compose([
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            pixel_values = transform(image).unsqueeze(0).to(
                self._model.device, dtype=torch.bfloat16
            )
            full_prompt = f"<image>\n{prompt}"
        else:
            pixel_values = None
            full_prompt = prompt

        t0 = time.monotonic()
        with torch.no_grad():
            response = self._model.chat(
                self._tokenizer,
                pixel_values,
                full_prompt,
                generation_config,
            )
        latency_ms = int((time.monotonic() - t0) * 1000)

        return VLMResponse(
            text=response.strip(),
            model=self.model_name,
            latency_ms=latency_ms,
        )

    def unload(self):
        """VRAM 해제."""
        import torch
        if self._model is not None:
            del self._model
            self._model = None
            self._tokenizer = None
            torch.cuda.empty_cache()
            print(f"[HuggingFaceVLM] {self.model_name} unloaded.")
