"""DAG 任务定义。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Task:
    name: str
    description: str = ""
    depends_on: List[str] = field(default_factory=list)
    worker: str = "main"
