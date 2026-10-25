"""SFT 数据构建器：从轨迹提取训练样本。"""
from __future__ import annotations
import json
from typing import List, Dict
from ..types import Trajectory

class SFTBuilder:
    def __init__(self, min_score: float = 0.75):
        self.min_score = min_score
    def build(self, trajs: List[Trajectory]) -> List[Dict]:
        samples = []
        seen = set()
        for t in trajs:
            if not t.success or t.score < self.min_score:
                continue
            if t.task in seen:
                continue
            seen.add(t.task)
            msgs = []
            for e in t.events:
                if e.type.value == "task":
                    msgs.append({"role": "user", "content": e.content["task"]})
                elif e.type.value == "action":
                    msgs.append({"role": "assistant", "content": json.dumps(e.content, ensure_ascii=False)})
                elif e.type.value == "observation":
                    msgs.append({"role": "user", "content": f"observation: {e.content['result'][:200]}"})
                elif e.type.value == "final":
                    msgs.append({"role": "assistant", "content": json.dumps(e.content, ensure_ascii=False)})
            samples.append({"messages": msgs, "meta": {"score": t.score}})
        return samples
