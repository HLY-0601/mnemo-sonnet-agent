"""工具注册表：Agent 能调用的原子能力。

新增工具只需 @register_tool 装饰一个 run(**kwargs) -> str 的函数。
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Callable, Dict, List, Optional

from ..types import MemoryLayer
from ..util import safe_eval, safe_exec

ToolFn = Callable[..., str]

_REGISTRY: Dict[str, Dict[str, Any]] = {}

# 工作区根目录用线程局部变量：多 Agent 并行时，每个 worker 写自己隔离的目录，互不串味
_LOCAL = threading.local()
DEFAULT_ROOT = "data/workspaces/main"


def get_workspace_root() -> str:
    return getattr(_LOCAL, "root", DEFAULT_ROOT)


def set_workspace_root(path: str) -> None:
    _LOCAL.root = path


def register_tool(name: str, description: str, args: Dict[str, str], schema_only: bool = False):
    """args: {参数名: 说明}"""
    def deco(fn: ToolFn) -> ToolFn:
        _REGISTRY[name] = {"fn": fn, "description": description, "args": args}
        return fn
    return deco


def list_tools() -> List[Dict[str, Any]]:
    return [{"name": n, "description": m["description"], "args": m["args"]}
            for n, m in _REGISTRY.items()]


def tools_prompt() -> str:
    return "\n".join(
        f"- {n}({', '.join(m['args'].keys())}): {m['description']}"
        for n, m in _REGISTRY.items())


def run_tool(name: str, **kwargs) -> str:
    if name not in _REGISTRY:
        return f"[错误] 未知工具 {name}。可用工具: {', '.join(_REGISTRY)}"
    try:
        return str(_REGISTRY[name]["fn"](**kwargs))
    except Exception as e:
        return f"[工具异常] {type(e).__name__}: {e}"


# ------------------------------ 内置工具 ------------------------------

@register_tool("calculator", "计算算术表达式，如 (12+8)*3/5", {"expression": "算术表达式"})
def calculator(expression: str) -> str:
    try:
        return str(safe_eval(expression))
    except Exception as e:
        return f"[计算失败] {e}"


@register_tool("python_exec", "在受限沙箱中执行 Python 代码片段，用 print 输出结果", {"code": "Python 代码"})
def python_exec(code: str) -> str:
    _, out = safe_exec(code)
    return out.strip() or "(无输出)"


def _resolve_path(path: str, root: Optional[str]) -> tuple:
    """解析工作区内路径，返回 (绝对路径, 工作区内相对路径)。

    坑（实测踩过）：工具把路径写进观察结果，模型下一轮把它抄成 path，
    os.path.join 再拼一次 root → data/ws/data/ws/report.md，每重试多一层。
    所以要做两件事：
    1) 反复剥离重复出现的 root 前缀。注意 root 既可能是绝对路径，也可能是
       相对路径（默认 "data/workspaces/main"），模型抄回来的通常是相对形式，
       两种都要剥 —— 只剥绝对形式是剥不掉的（第一版就错在这）。
    2) 观察里只回传相对路径，避免持续污染上下文。
    """
    raw_root = root or get_workspace_root()
    base = os.path.abspath(raw_root)
    p = (path or "").strip().replace("\\", "/")
    prefixes = []
    for cand in (base, raw_root, os.path.relpath(base, os.getcwd())):
        c = (cand or "").replace("\\", "/").rstrip("/")
        if c and c not in prefixes:
            prefixes.append(c)
    for _ in range(8):                       # 上限防止死循环
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
    # 越界访问（../ 或绝对路径逃出工作区）一律拉回工作区内
    if not (ap.startswith(base + os.sep) or ap == base):
        ap = os.path.join(base, os.path.basename(p))
    return ap, os.path.relpath(ap, base)


@register_tool("file_write", "把内容写入工作区文件", {"path": "相对路径", "content": "文件内容"})
def file_write(path: str, content: Optional[str] = None, root: Optional[str] = None) -> str:
    full, rel = _resolve_path(path, root)
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    if content is None:      # 小模型常常只给 path，这里兜底而不是报错
        content = f"# {rel}\n\n（由 Agent 创建，内容待补充）\n"
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    # 只回传相对路径：绝对路径会被模型抄进下一轮的 path 参数，导致无限嵌套
    return f"已写入 {rel}（{len(content)} 字符）"


@register_tool("file_read", "读取工作区文件内容", {"path": "相对路径"})
def file_read(path: str, root: Optional[str] = None) -> str:
    full, rel = _resolve_path(path, root)
    if not os.path.exists(full):
        return f"[错误] 文件不存在: {rel}"
    with open(full, "r", encoding="utf-8") as f:
        return f.read()


# 极简离线知识库：无网络时也能演示检索；有 SEA_SEARCH_FN 时可替换为真实检索
_KB = {
    "transformer": "Transformer 由 Vaswani 等人在 2017 年提出，核心是自注意力机制，取代了 RNN 的序列依赖。",
    "lora": "LoRA 通过低秩矩阵分解微调，只训练 A/B 两个小矩阵，显存与算力开销远低于全参数微调。",
    "react": "ReAct 让模型交替输出思考(Thought)与行动(Action)，把推理与外部工具调用交织在一起。",
    "reflexion": "Reflexion 让 Agent 对失败轨迹做语言反思，把反思文本作为下一轮尝试的上下文，不改动模型权重。",
    "mem0": "mem0 是面向 Agent 的记忆层，提供跨会话的事实抽取、更新与检索，支持向量与图存储。",
    "graphiti": "Graphiti 构建带双时序边的知识图谱，能回答『某个时间点系统知道什么』这类历史查询。",
    "voyager": "Voyager 用可执行技能库积累能力：技能含代码、描述与验证测试，成功后永久复用。",
}


@register_tool(
    "search",
    "【查外部世界】查百科知识、技术文档、概念含义等**新信息**。"
    "当用户问「什么是X」「查查X」「了解X」时用这个",
    {"query": "检索词"})
def search(query: str) -> str:
    q = query.lower()
    hits = [v for k, v in _KB.items() if k in q or k in query]
    if not hits:
        hits = [v for v in _KB.values() if any(tok in v for tok in q.split()[:3])]
    return "\n".join(f"- {h}" for h in hits) if hits else "未检索到相关内容"


_GLOBAL_MEMORY = None


def bind_memory(memory) -> None:
    """把记忆引擎绑定到工具层，供 memory_search / memory_remember 使用。"""
    global _GLOBAL_MEMORY
    _GLOBAL_MEMORY = memory


@register_tool(
    "memory_search",
    "【查自己的过去】只在你自己过去的对话记录与经验里翻找。"
    "只有用户说「以前」「记得吗」「上次」时才用；查新知识一律用 search",
    {"query": "检索词"})
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
