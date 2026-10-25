"""统一模型层抽象。"""
from __future__ import annotations
import json, os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

Message = Dict[str, str]
MODEL_FAMILY = "Mnemo-Sonnet"
MODEL_TAGLINE = "记忆驱动的自我进化模型（Memory-native Self-evolving Model）"
BASE_VARIANT = "base"
MOCK_VARIANT = "base-mock"

@dataclass
class GenerationConfig:
    temperature: float = 0.2
    max_tokens: int = 512
    stop: List[str] = field(default_factory=list)
    json_mode: bool = False

class ModelBackend(ABC):
    name: str = "backend"
    @abstractmethod
    def generate(self, messages: List[Message], config: Optional[GenerationConfig] = None) -> str:
        ...
    def reload(self, adapter_path: Optional[str]) -> None:
        return None
    @property
    def variant(self) -> str:
        return getattr(self, "_version", self.name)
    @property
    def version(self) -> str:
        return f"{MODEL_FAMILY}/{self.variant}"
    def describe(self) -> str:
        return f"{self.version} [{self.name}] — {MODEL_TAGLINE}"

def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except Exception:
        pass
    depth, start = 0, -1
    in_str, esc = False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    start = -1
    raise ValueError(f"无法从模型输出中解析 JSON：{text[:200]}")

class MockBackend(ModelBackend):
    name = "mock"
    def __init__(self, flaky: Optional[bool] = None):
        self._flaky = flaky if flaky is not None else os.getenv("SEA_MOCK_FLAKY", "0") == "1"
        self._flaky_used = False
        self._version = MOCK_VARIANT
    def _pick_tool(self, task: str) -> Dict[str, Any]:
        import re
        t = task.lower()
        file_hit = re.search(r"[\w\-]+\.(py|md|txt|json|yaml)", task)
        if any(k in t for k in ("写入", "写文件", "保存", "编写", "生成", "文档")) and file_hit:
            path = file_hit.group(0)
            return {"tool": "file_write", "args": {"path": path, "content": f"# {task[:40]}\n"}}
        if any(k in t for k in ("计算", "等于", "多少", "+", "-", "*", "/")) and any(c.isdigit() for c in task):
            m = re.search(r"[\d\.\s\+\-\*/\(\)%]{3,}", task)
            return {"tool": "calculator", "args": {"expression": m.group(0).strip() if m else "0"}}
        if any(k in t for k in ("搜索", "查一下", "是什么")):
            return {"tool": "search", "args": {"query": task}}
        if any(k in t for k in ("运行", "执行代码", "python")):
            return {"tool": "python_exec", "args": {"code": "print(1+1)"}}
        return {"tool": "memory_search", "args": {"query": task}}
    def generate(self, messages: List[Message], config: Optional[GenerationConfig] = None) -> str:
        cfg = config or GenerationConfig()
        last = messages[-1]["content"] if messages else ""
        task_text = " ".join(m["content"] for m in messages if m["role"] == "user") or last
        n_assistant = sum(1 for m in messages if m["role"] == "assistant")
        if ("复盘" in last or "反思" in last) and "失败" in last:
            return json.dumps({"diagnosis": "上一轮直接给出了结论，没有先用工具核实",
                               "lesson": "遇到含数值的问题，必须先调用工具核算",
                               "next_plan": "第一步调用 calculator 核算，第二步作答"}, ensure_ascii=False)
        if last.startswith("observation:") or last.startswith("观察:"):
            obs = last.split(":", 1)[1].strip()
            if self._flaky and not self._flaky_used:
                self._flaky_used = True
                return json.dumps({"thought": "我已经知道答案了", "final": "大概是 42"}, ensure_ascii=False)
            return json.dumps({"thought": "工具已返回结果", "final": f"根据工具结果：{obs[:300]}"}, ensure_ascii=False)
        if n_assistant >= 1:
            return json.dumps({"thought": "已有足够信息", "final": f"综合已有信息：{last[:160]}"}, ensure_ascii=False)
        action = self._pick_tool(task_text)
        return json.dumps({"thought": f"需要先核实信息，调用 {action['tool']}", "action": action}, ensure_ascii=False)
