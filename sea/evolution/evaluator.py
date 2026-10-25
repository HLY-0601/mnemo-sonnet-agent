"""评测器：黄金集 A/B 测试。"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

class Evaluator:
    def __init__(self, golden: Optional[List[Dict]] = None):
        self.golden = golden or []
    def run(self, agent, tasks: List[Dict]) -> Dict[str, Any]:
        results = []
        for t in tasks:
            traj = agent.run(t["task"], expected=t.get("expected"))
            results.append(traj)
        passed = sum(1 for r in results if r.success)
        return {"pass_rate": passed / len(results) if results else 0,
                "avg_score": sum(r.score for r in results) / len(results) if results else 0,
                "n": len(results)}
