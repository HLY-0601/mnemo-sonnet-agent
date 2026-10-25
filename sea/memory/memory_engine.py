"""统一记忆引擎：工作/情景/语义/程序四层。"""
from __future__ import annotations
import os, time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from ..types import MemoryItem, MemoryLayer, TemporalFact
from ..util import read_json, recency_score, truncate, write_json
from .embedder import Embedder, build_embedder
from .skill_library import SkillLibrary
from .temporal_graph import TemporalGraph
from .vector_index import VectorIndex

@dataclass
class Recalled:
    item: MemoryItem
    score: float

class MemoryEngine:
    def __init__(self, root: str = "data/memory", embedder: Optional[Embedder] = None,
                 embedder_kind: str = "hashing", working_capacity: int = 20):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.embedder = embedder or build_embedder(embedder_kind)
        self.dim = self.embedder.dim
        self.working: List[MemoryItem] = []
        self.working_capacity = working_capacity
        self.items: Dict[str, MemoryItem] = {}
        self.episodic = VectorIndex(self.dim, os.path.join(root, "episodic.npz"))
        self.semantic = VectorIndex(self.dim, os.path.join(root, "semantic.npz"))
        self.graph = TemporalGraph(os.path.join(root, "facts.json"))
        self.skills = SkillLibrary(os.path.join(root, "skills"), embedder=self.embedder)
        for d in read_json(os.path.join(root, "items.json"), []):
            it = MemoryItem.from_dict(d)
            self.items[it.id] = it
    def remember(self, text: str, layer: MemoryLayer = MemoryLayer.SEMANTIC,
                 kind: str = "note", salience: float = 0.5, meta: Dict = None) -> MemoryItem:
        it = MemoryItem(layer=layer, kind=kind, text=text, salience=salience,
                        meta=meta or {})
        if layer == MemoryLayer.WORKING:
            self.working.append(it)
            if len(self.working) > self.working_capacity:
                self.working.pop(0)
        else:
            self.items[it.id] = it
            vec = self.embedder.encode_one(text)
            if layer == MemoryLayer.EPISODIC:
                self.episodic.add(vec, it.id)
            elif layer == MemoryLayer.SEMANTIC:
                self.semantic.add(vec, it.id)
        self._save()
        return it
    def recall(self, query: str, top_k: int = 5) -> List[Recalled]:
        qv = self.embedder.encode_one(query)
        out = []
        for vid, idx in self.semantic.search(qv, top_k):
            it = self.items.get(idx)
            if it:
                out.append(Recalled(it, vid))
        return out[:top_k]
    def build_context(self, query: str, k: int = 4) -> str:
        hits = self.recall(query, top_k=k)
        if not hits:
            return ""
        return "\n".join(f"- [{h.item.kind}] {h.item.text[:150]}" for h in hits)
    def _save(self):
        write_json(os.path.join(self.root, "items.json"), [it.to_dict() for it in self.items.values()])
        self.episodic.save()
        self.semantic.save()
        self.graph.save()
