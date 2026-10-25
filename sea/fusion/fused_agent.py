"""融合层：模型 × Agent × 记忆 × 进化。"""
from __future__ import annotations
from typing import Optional
from ..agent import AgentConfig, ReActAgent
from ..memory import MemoryEngine
from ..model import ModelBackend, build_backend

class FusedAgent:
    def __init__(self, backend: Optional[ModelBackend] = None,
                 memory: Optional[MemoryEngine] = None,
                 config: Optional[AgentConfig] = None):
        self.backend = backend or build_backend()
        self.memory = memory
        self.agent = ReActAgent(self.backend, self.memory, config=config)
    def chat(self, task: str) -> str:
        traj = self.agent.run(task)
        return traj.final_answer
    def teach(self, subject: str, predicate: str, obj: str) -> None:
        if self.memory:
            self.memory.graph.add(subject, predicate, obj)
    def hot_swap(self, adapter_path: Optional[str]) -> None:
        self.backend.reload(adapter_path)
    def model_card(self) -> str:
        return self.backend.describe()
