"""本地 HF 模型后端（Qwen2.5 等）。"""
from __future__ import annotations
from typing import List, Optional
from .base import GenerationConfig, Message, ModelBackend

class HFBackend(ModelBackend):
    name = "hf"
    def __init__(self, model_path: str, adapter_path: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self._version = "base"
        if adapter_path:
            self.reload(adapter_path)
    def reload(self, adapter_path: Optional[str]) -> None:
        if adapter_path is None:
            return
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self._version = adapter_path.split("/")[-1]
    def generate(self, messages: List[Message], config: Optional[GenerationConfig] = None) -> str:
        cfg = config or GenerationConfig()
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt")
        out = self.model.generate(**inputs, max_new_tokens=cfg.max_tokens, temperature=cfg.temperature)
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
