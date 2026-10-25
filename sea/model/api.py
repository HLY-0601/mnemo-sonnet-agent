"""远端 API 后端。"""
from __future__ import annotations
import os
from typing import List, Optional
from .base import GenerationConfig, Message, ModelBackend

class APIBackend(ModelBackend):
    name = "api"
    def __init__(self, model: str = "deepseek-chat", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("SEA_API_KEY") or os.getenv("OPENAI_API_KEY")
        self._version = model
    def generate(self, messages: List[Message], config: Optional[GenerationConfig] = None) -> str:
        cfg = config or GenerationConfig()
        import requests
        resp = requests.post("https://api.deepseek.com/v1/chat/completions",
                             headers={"Authorization": f"Bearer {self.api_key}"},
                             json={"model": self.model, "messages": messages,
                                   "temperature": cfg.temperature, "max_tokens": cfg.max_tokens})
        return resp.json()["choices"][0]["message"]["content"]
