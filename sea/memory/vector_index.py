"""向量索引（暴力余弦，万级够用）。"""
from __future__ import annotations
import os
import numpy as np

class VectorIndex:
    def __init__(self, dim: int, path: str):
        self.dim = dim
        self.path = path
        self.ids: List[str] = []
        self.mat = np.zeros((0, dim), dtype=np.float32)
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            self.ids = list(d["ids"])
            self.mat = d["mat"]
    def add(self, vec: np.ndarray, idx: str):
        self.ids.append(idx)
        self.mat = np.vstack([self.mat, vec.reshape(1, -1)])
    def search(self, q: np.ndarray, top_k: int = 5):
        if len(self.ids) == 0:
            return []
        sims = self.mat @ q / (np.linalg.norm(self.mat, axis=1) * np.linalg.norm(q) + 1e-8)
        order = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self.ids[i]) for i in order]
    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        np.savez(self.path, ids=np.array(self.ids), mat=self.mat)
