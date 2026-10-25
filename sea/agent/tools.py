"""工具注册表：Agent 能调用的原子能力。"""
from __future__ import annotations
import json, os, threading
from typing import Any, Callable, Dict, List, Optional
from ..types import MemoryLayer
from ..util import safe_eval, safe_exec

ToolFn = Callable[..., str]
_REGISTRY: Dict[str, Dict[str, Any]] = {}
_LOCAL = threading.local()
DEFAULT_ROOT = "data/workspaces/main"

def get_workspace_root() -> str:
    return getattr(_LOCAL, "root", DEFAULT_ROOT)

def set_workspace_root(path: str) -> None:
    _LOCAL.root = path

def register_tool(name: str, description: str, args: Dict[str, str], schema_only: bool = False):
    def deco(fn: ToolFn) -> ToolFn:
        _REGISTRY[name] = {"fn": fn, "description": description, "args": args}
        return fn
    return deco

def list_tools() -> List[Dict[str, Any]]:
    return [{"name": n, "description": m["description"], "args": m["args"]}
            for n, m in _REGISTRY.items()]

def tools_prompt() -> str:
    return "\n".join(f"- {n}({', '.join(m['args'].keys())}): {m['description']}" for n, m in _REGISTRY.items())

def run_tool(name: str, **kwargs) -> str:
    if name not in _REGISTRY:
        return f"[错误] 未知工具 {name}。可用工具: {', '.join(_REGISTRY)}"
    try:
        return str(_REGISTRY[name]["fn"](**kwargs))
    except Exception as e:
        return f"[工具异常] {type(e).__name__}: {e}"

@register_tool("calculator", "计算算术表达式", {"expression": "算术表达式"})
def calculator(expression: str) -> str:
    try:
        return str(safe_eval(expression))
    except Exception as e:
        return f"[计算失败] {e}"

@register_tool("python_exec", "在受限沙箱中执行 Python 代码片段", {"code": "Python 代码"})
def python_exec(code: str) -> str:
    _, out = safe_exec(code)
    return out.strip() or "(无输出)"

def _resolve_path(path: str, root: Optional[str]) -> tuple:
    raw_root = root or get_workspace_root()
    base = os.path.abspath(raw_root)
    p = (path or "").strip().replace("\\", "/")
    prefixes = []
    for cand in (base, raw_root, os.path.relpath(base, os.getcwd())):
        c = (cand or "").replace("\\", "/").rstrip("/")
        if c and c not in prefixes:
            prefixes.append(c)
    for _ in range(8):
        hit = False
        for pre in prefixes:
            if p == pre:
                p, hit = "", True
                break
            if p.startswith(pre + "/"):
                p, hit = p[len(pre) + 1:], True
                break
        if not hit:
            break
    p = p.lstrip("/") or "."
    ap = os.path.abspath(os.path.join(base, p))
    if not (ap.startswith(base + os.sep) or ap == base):
        ap = os.path.join(base, os.path.basename(p))
    return ap, os.path.relpath(ap, base)

@register_tool("file_write", "把内容写入工作区文件", {"path": "相对路径", "content": "文件内容"})
def file_write(path: str, content: Optional[str] = None, root: Optional[str] = None) -> str:
    full, rel = _resolve_path(path, root)
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    if content is None:
        content = f"# {rel}\n\n（由 Agent 创建，内容待补充）\n"
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已写入 {rel}（{len(content)} 字符）"

@register_tool("file_read", "读取工作区文件内容", {"path": "相对路径"})
def file_read(path: str, root: Optional[str] = None) -> str:
    full, rel = _resolve_path(path, root)
    if not os.path.exists(full):
        return f"[错误] 文件不存在: {rel}"
    with open(full, "r", encoding="utf-8") as f:
        return f.read()

_KB = {
    "transformer": "Transformer 由 Vaswani 等人在 2017 年提出，核心是自注意力机制。",
    "lora": "LoRA 通过低秩矩阵分解微调，只训练 A/B 两个小矩阵。",
    "react": "ReAct 让模型交替输出思考(Thought)与行动(Action)。",
    "reflexion": "Reflexion 让 Agent 对失败轨迹做语言反思。",
    "mem0": "mem0 是面向 Agent 的记忆层。",
    "graphiti": "Graphiti 构建带双时序边的知识图谱。",
    "voyager": "Voyager 用可执行技能库积累能力。",
}

@register_tool("search", "在本地知识库中检索事实", {"query": "检索词"})
def search(query: str) -> str:
    q = query.lower()
    hits = [v for k, v in _KB.items() if k in q or k in query]
    if not hits:
        hits = [v for v in _KB.values() if any(tok in v for tok in q.split()[:3])]
    return "\n".join(f"- {h}" for h in hits) if hits else "未检索到相关内容"

_GLOBAL_MEMORY = None

def bind_memory(memory) -> None:
    global _GLOBAL_MEMORY
    _GLOBAL_MEMORY = memory

@register_tool("memory_search", "在长期记忆中检索", {"query": "检索词"})
def memory_search(query: str, top_k: int = 3) -> str:
    if _GLOBAL_MEMORY is None:
        return "（记忆引擎未绑定）"
    hits = _GLOBAL_MEMORY.recall(query, top_k=top_k)
    if not hits:
        return "未检索到相关记忆"
    return "\n".join(f"- [{h.item.kind}] {h.item.text[:200]}（相关度 {h.score:.2f}）" for h in hits)

@register_tool("memory_remember", "把重要信息写入长期记忆", {"text": "要记住的内容"})
def memory_remember(text: str) -> str:
    if _GLOBAL_MEMORY is None:
        return "（记忆引擎未绑定）"
    _GLOBAL_MEMORY.remember(text, layer=MemoryLayer.SEMANTIC, kind="note", salience=0.7)
    return f"已记住：{text[:60]}"

@register_tool("finish", "结束任务并给出最终答案", {"answer": "最终答案"})
def finish(answer: str) -> str:
    return answer
