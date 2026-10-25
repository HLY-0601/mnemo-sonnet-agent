"""自动进化闭环：诊断 → 策略 → 训练 → A/B → promote/rollback。"""
from __future__ import annotations
import os, time
from typing import Any, Dict, List, Optional
from ..types import Trajectory, EventType
from ..util import read_json, write_json

class EvolutionLoop:
    def __init__(self, root: str = "data/evolution", min_score: float = 0.75,
                 min_samples: int = 4, promote_min_delta: float = 0.0,
                 circuit_break_delta: float = -0.05):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.min_score = min_score
        self.min_samples = min_samples
        self.promote_min_delta = promote_min_delta
        self.circuit_break_delta = circuit_break_delta
        self.history = read_json(os.path.join(root, "history.json"), [])
    def collect_trajectories(self, trajs: List[Trajectory]) -> List[Trajectory]:
        return [t for t in trajs if t.success and t.score >= self.min_score]
    def build_sft(self, trajs: List[Trajectory]) -> List[Dict]:
        samples = []
        seen_tasks = set()
        for t in trajs:
            if t.task in seen_tasks:
                continue
            seen_tasks.add(t.task)
            msgs = []
            for e in t.events:
                if e.type == EventType.TASK:
                    msgs.append({"role": "user", "content": e.content["task"]})
                elif e.type == EventType.ACTION:
                    msgs.append({"role": "assistant", "content": str(e.content)})
                elif e.type == EventType.OBSERVATION:
                    msgs.append({"role": "user", "content": f"observation: {e.content['result'][:100]}"})
                elif e.type == EventType.FINAL:
                    msgs.append({"role": "assistant", "content": str(e.content)})
            samples.append({"messages": msgs, "meta": {"score": t.score, "model_version": t.model_version}})
        return samples
    def evaluate(self, baseline: float, candidate: float) -> str:
        delta = candidate - baseline
        if delta < self.circuit_break_delta:
            return "circuit_break"
        if delta >= self.promote_min_delta:
            return "promote"
        return "rollback"
    def record(self, entry: Dict):
        self.history.append(entry)
        write_json(os.path.join(self.root, "history.json"), self.history)
