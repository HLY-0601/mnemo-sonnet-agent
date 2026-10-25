"""可执行技能库（Voyager 风格）。"""
from __future__ import annotations
import os, re
from typing import List, Tuple, Optional, Dict
from ..types import Skill
from ..util import read_json, write_json, safe_exec
from .embedder import Embedder

class SkillLibrary:
    def __init__(self, root: str, embedder: Optional[Embedder] = None):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.embedder = embedder
        self.skills: Dict[str, Skill] = {}
        for d in read_json(os.path.join(root, "skills.json"), []):
            s = Skill.from_dict(d)
            self.skills[s.name] = s
    def search(self, query: str, top_k: int = 1, min_confidence: float = 0.0) -> List[Tuple[Skill, float]]:
        scored = []
        for s in self.skills.values():
            if s.confidence < min_confidence:
                continue
            score = len(set(query.lower().split()) & set((s.name + " " + s.description).lower().split()))
            scored.append((s, score))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
    def validate(self, skill: Skill) -> Tuple[bool, str]:
        if not skill.tests:
            return False, "无测试"
        try:
            ns, _ = safe_exec(skill.tests)
            test_fn = ns.get("test")
            if test_fn:
                test_fn()
            return True, "通过"
        except Exception as e:
            return False, str(e)
    def execute(self, name: str, **kwargs):
        s = self.skills.get(name)
        if not s:
            return None, "未找到"
        try:
            ns, _ = safe_exec(s.code)
            run = ns.get("run")
            result = run(**kwargs) if run else None
            s.success += 1
            self._save()
            return result, "成功"
        except Exception as e:
            s.failure += 1
            self._save()
            return None, str(e)
    def _save(self):
        write_json(os.path.join(self.root, "skills.json"), [s.to_dict() for s in self.skills.values()])
