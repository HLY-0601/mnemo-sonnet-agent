"""核心数据结构：事件流、轨迹、记忆、技能、时序事实。"""
from __future__ import annotations
import hashlib, json, time, uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

class EventType(str, Enum):
    TASK = "task"; PLAN = "plan"; ACTION = "action"; OBSERVATION = "observation"
    REFLECTION = "reflection"; ERROR = "error"; FINAL = "final"; SYSTEM = "system"

class MemoryLayer(str, Enum):
    WORKING = "working"; EPISODIC = "episodic"; SEMANTIC = "semantic"; PROCEDURAL = "procedural"

@dataclass
class Event:
    type: EventType
    session_id: str
    agent_id: str
    content: Dict[str, Any] = field(default_factory=dict)
    parent: Optional[str] = None
    id: str = field(default_factory=lambda: new_id("ev"))
    ts: float = field(default_factory=time.time)

@dataclass
class MemoryItem:
    layer: MemoryLayer
    kind: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)
    salience: float = 0.5
    ts: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: new_id("mem"))

@dataclass
class TemporalFact:
    subject: str
    predicate: str
    obj: str
    event_time: float = field(default_factory=time.time)
    ingest_time: float = field(default_factory=time.time)
    valid_to: Optional[float] = None
    source: str = ""
    id: str = field(default_factory=lambda: new_id("fact"))

@dataclass
class Skill:
    name: str
    description: str
    code: str
    preconditions: str = ""
    postconditions: str = ""
    tests: str = ""
    source_traj: str = ""
    version: int = 1
    success: int = 0
    failure: int = 0
    tags: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: new_id("skill"))
    created_at: float = field(default_factory=time.time)

    @property
    def fingerprint(self) -> str:
        return sha1(self.code + self.tests)

    @property
    def confidence(self) -> float:
        return (self.success + 1.0) / (self.success + self.failure + 2.0)

@dataclass
class Trajectory:
    task: str
    session_id: str
    agent_id: str = "agent"
    events: List[Event] = field(default_factory=list)
    final_answer: str = ""
    success: Optional[bool] = None
    score: float = 0.0
    reflections: List[str] = field(default_factory=list)
    model_version: str = "base"
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("traj"))
    created_at: float = field(default_factory=time.time)

    def add(self, type: EventType, content: Dict[str, Any], parent: Optional[str] = None) -> Event:
        ev = Event(type=type, session_id=self.session_id, agent_id=self.agent_id, content=content, parent=parent)
        self.events.append(ev)
        return ev

    @property
    def n_steps(self) -> int:
        return sum(1 for e in self.events if e.type == EventType.ACTION)

def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
