"""文本向量化：默认零依赖哈希嵌入，可选 sentence-transformers。"""
from __future__ import annotations
import hashlib, re
from typing import List, Optional
import numpy as np

class Embedder:
    name = "base"
    dim = 256
    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError
    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

class HashingEmbedder(Embedder):
    name = "hashing"
    def __init__(self, dim: int = 256, ngram: int = 2):
        self.dim = dim
        self.ngram = ngram
    def _tokens(self, text: str) -> List[str]:
        t = text.lower()
        toks = []
        words = re.findall(r"[a-z0-9]+", t)
        toks.extend(words)
        han = re.sub(r"[^一-鿿]", "", t)
        toks.extend(han[i:i + self.ngram] for i in range(max(0, len(han) - self.ngram + 1)))
        return toks or [t]
    def encode(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            vec = out[i]
            for tok in self._tokens(text):
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 20) & 1 else -1.0
                vec[idx] += sign
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
        return out

class STEmbedder(Embedder):
    name = "st"
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
    def encode(self, texts: List[str]) -> np.ndarray:
        v = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)

def build_embedder(kind: str = "hashing", **kwargs) -> Embedder:
    kind = (kind or "hashing").lower()
    if kind in ("st", "sentence-transformers", "sbert"):
        try:
            return STEmbedder(**kwargs)
        except Exception as e:
            print(f"[memory] sentence-transformers 不可用({e})，回退哈希嵌入")
    return HashingEmbedder(**kwargs)
