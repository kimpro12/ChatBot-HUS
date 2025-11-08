"""Local causal language model utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # pragma: no cover - optional dependency in CPU-only environments
    from bitsandbytes import __version__ as _bitsandbytes_version  # type: ignore  # noqa: F401
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover - bitsandbytes is optional
    BitsAndBytesConfig = None  # type: ignore

from .config import LLMConfig


@dataclass(slots=True)
class LocalCausalLM:
    """Wrapper that loads a causal LM via Transformers with CPU/GPU fallback."""

    config: LLMConfig
    _tokenizer: Optional[AutoTokenizer] = None
    _model: Optional[AutoModelForCausalLM] = None

    def _load_tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def _load_model(self) -> AutoModelForCausalLM:
        if self._model is None:
            kwargs = {}

            if (
                self.config.use_bitsandbytes
                and BitsAndBytesConfig is not None
                and torch.cuda.is_available()
            ):
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                kwargs["device_map"] = self.config.device_map or "auto"
            else:
                if torch.cuda.is_available():
                    kwargs["device_map"] = self.config.device_map or "auto"
                    kwargs["torch_dtype"] = torch.bfloat16
                else:
                    kwargs["device_map"] = "cpu"
                    kwargs["torch_dtype"] = torch.float32

            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **kwargs,
            )
            self._model.eval()
        return self._model

    def generate(self, prompt: str) -> str:
        tokenizer = self._load_tokenizer()
        model = self._load_model()

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        )

        device = getattr(model, "device", None)
        if device is None:
            device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = output_ids[0, inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True)


def format_chat_prompt(messages: Iterable[dict], context: str, question: str) -> str:
    """Build a simple prompt from chat history and retrieved context."""

    history_parts = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        if role == "SYSTEM":
            continue
        history_parts.append(f"{role}: {content}")

    history = "\n".join(history_parts)

    instructions = (
        "SYSTEM: Bạn là trợ lý tuyển sinh của trường Đại học. Chỉ trả lời dựa trên NGỮ CẢNH cung cấp. "
        "Nếu không tìm thấy thông tin hãy nói \"Không tìm thấy trong tài liệu\". "
        "Trả lời bằng tiếng Việt và dẫn nguồn theo định dạng (Tên tài liệu - Trang/Bảng)."
    )

    return (
        f"{instructions}\n\n"
        f"HISTORY:\n{history}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"USER QUESTION: {question}"
    )
