"""多 Agent 编排器。"""
from __future__ import annotations
from typing import Dict, List, Optional
from .dag import Task

class Orchestrator:
    def __init__(self):
        self.tasks: List[Task] = []
    def add_task(self, task: Task):
        self.tasks.append(task)
    def run(self):
        return {"completed": [t.name for t in self.tasks]}
