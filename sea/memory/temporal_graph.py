"""双时序知识图谱（Graphiti 风格）。"""
from __future__ import annotations
import os, time
from typing import List, Optional
from ..types import TemporalFact
from ..util import read_json, write_json

class TemporalGraph:
    def __init__(self, path: str):
        self.path = path
        self.facts: List[TemporalFact] = []
        for d in read_json(path, []):
            self.facts.append(TemporalFact.from_dict(d))
    def add(self, subject: str, predicate: str, obj: str, source: str = "") -> TemporalFact:
        for f in self.facts:
            if f.subject == subject and f.predicate == predicate and f.valid_to is None:
                f.valid_to = time.time()
        f = TemporalFact(subject=subject, predicate=predicate, obj=obj, source=source)
        self.facts.append(f)
        self.save()
        return f
    def query(self, subject: str = None, predicate: str = None) -> List[TemporalFact]:
        out = self.facts
        if subject:
            out = [f for f in out if f.subject == subject and f.valid_to is None]
        if predicate:
            out = [f for f in out if f.predicate == predicate and f.valid_to is None]
        return out
    def save(self):
        write_json(self.path, [f.to_dict() for f in self.facts])
