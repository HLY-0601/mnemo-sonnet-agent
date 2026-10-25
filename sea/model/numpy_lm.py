"""numpy 语言模型（手写 Transformer 前向）。"""
from __future__ import annotations
import numpy as np

class NumpyLM:
    def __init__(self, vocab_size: int = 256, d_model: int = 64, n_layers: int = 3):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((x.shape[0], self.vocab_size))
