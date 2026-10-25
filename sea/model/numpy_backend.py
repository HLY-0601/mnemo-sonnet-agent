"""纯 numpy 模型后端（Mnemo-Sonnet 权重）。"""
from __future__ import annotations
import numpy as np
from typing import List, Optional
from .base import GenerationConfig, Message, ModelBackend

class NumpyBackend(ModelBackend):
    name = "numpy"
    def __init__(self, weights_path: str):
        self.data = np.load(weights_path, allow_pickle=True)
        self._version = weights_path.split("/")[-1].replace(".npz", "")
    def generate(self, messages: List[Message], config: Optional[GenerationConfig] = None) -> str:
        return '{"thought": "numpy 推理", "final": "placeholder"}'
