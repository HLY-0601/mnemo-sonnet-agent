"""两阶段级联后端：推理模型负责「想」，协议层负责「写格式」。"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from sea.model.base import GenerationConfig, Message, ModelBackend
from sea.model.numpy_backend import NumpyBackend

_PLANNER_TEMPLATE = """你是一个任务分析器。从下面的工具清单里选出该用的工具，并给出参数。

可用工具（只能用这些名字，不要自己发明）：
{tools}

按这个格式输出，两行：
工具名：<这里填工具名>
参数：<这里填参数值，必须原样抄自用户的任务描述，不要改写、不要翻译>

示例：
用户：什么是Elasticsearch？
工具名：search
参数：Elasticsearch

用户：查查 Zyzzogeton
工具名：search
参数：Zyzzogeton

用户：计算 (17 + 88) * 4 等于多少？
工具名：calculator
参数：(17 + 88) * 4

用户：上次我们是怎么配置 Nginx 的？
工具名：memory_search
参数：Nginx

⚠️ 三个容易混的地方：
1. search vs memory_search：用户问「什么是X / 查查X / 了解X」= 查新知识 → 用 search；用户说「上次 / 以前 / 记得吗」= 翻自己的历史 → 用 memory_search
2. search vs python_exec：用户说「查查X / 了解X / 讲讲X」= 查资料 → 用 search；只有用户**直接给出了一段代码**（如"执行 print(1+1)"）才用 python_exec
3. calculator vs python_exec：用户说「算一下 / 等于多少」= 算数学 → 用 calculator；用户说「执行/跑一段代码」= 跑代码 → 用 python_exec

不要输出 JSON，不要自己计算，不要解释。只要上面两行。"""


def planner_prompt(tools: Optional[List[str]] = None) -> str:
    try:
        from sea.agent.tools import list_tools
        items = list_tools()
        if tools:
            items = [t for t in items if t["name"] in set(tools)]
        tools_txt = "\n".join(
            f"- {t['name']}({', '.join(t['args'].keys())}): {t['description']}"
            for t in items)
    except Exception:
        tools_txt = ("- calculator(expression): 计算算术表达式\n"
                    "- search(query): 检索资料\n"
                    "- file_write(path): 写文件\n"
                    "- file_read(path): 读文件\n"
                    "- python_exec(code): 执行 Python 代码")
    return _PLANNER_TEMPLATE.format(tools=tools_txt)


_TOOL_HINTS = [
    ("calculator", ("计算", "算", "求值", "等于多少", "得多少", "数值", "加", "减", "乘", "除")),
    ("search", ("什么是", "查", "搜索", "检索", "了解", "讲讲", "介绍", "谁", "查询")),
    ("file_write", ("写入", "写文件", "创建文件", "新建", "导出", "保存", "生成")),
    ("file_read", ("读取", "读文件", "打开", "查看文件", "内容")),
    ("python_exec", ("执行", "运行", "跑", "代码")),
    ("memory_search", ("以前", "之前", "记得", "历史", "经验")),
    ("memory_remember", ("记住", "记下", "保存这条")),
]

_ARG_KEY_OF = {
    "calculator": "expression",
    "search": "query",
    "file_write": "path",
    "file_read": "path",
    "python_exec": "code",
    "memory_search": "query",
    "memory_remember": "text",
}


def _tokenize_for_match(text: str):
    out, buf = [], ""
    for ch in text:
        if ord(ch) < 128 and (ch.isalnum() or ch in "._-"):
            buf += ch
        else:
            if buf:
                out.append(buf)
                buf = ""
            if ord(ch) >= 128 and not ch.isspace():
                out.append(ch)
    if buf:
        out.append(buf)
    return out


def _looks_like_code(text: str) -> bool:
    """判断一段文本是否像 Python 代码。
    诊断用：planner 选了 python_exec，但参数不像代码 → 可能是 search 误判。
    """
    t = text.strip()
    if not t:
        return False
    code_signals = ["print(", "=", "def ", "import ", "(", ")", ".", "[", "]", "{", "}", "lambda", "return"]
    hits = sum(1 for s in code_signals if s in t)
    return hits >= 1 and len(t) >= 3


class CascadeBackend(ModelBackend):
    name = "cascade"

    def __init__(self, planner: ModelBackend, formatter: Optional[ModelBackend] = None,
                 verbose: bool = False, tools: Optional[List[str]] = None):
        self.planner = planner
        self.tools = tools
        self.formatter = formatter
        self.verbose = verbose
        self.model_name_or_path = f"cascade({planner.name}+{formatter.name if formatter else 'rule'})"
        self._version = "cascade"
        self.last_plan = ""
        self.last_source = ""

    @staticmethod
    def _extract(plan: str, task: str) -> Optional[Dict[str, Any]]:
        m_tool = re.search(r"工具名\s*[:：]\s*([A-Za-z_]\w*)", plan)
        tool = m_tool.group(1) if m_tool else None
        if tool is not None and tool not in _ARG_KEY_OF:
            tool = None

        if tool is None:
            for name in _ARG_KEY_OF:
                if re.search(rf"\b{name}\b", plan):
                    tool = name
                    break

        if tool is None:
            for name, kws in _TOOL_HINTS:
                if any(k in plan for k in kws):
                    tool = name
                    break

        if tool is None:
            return None

        arg_key = _ARG_KEY_OF[tool]

        if tool == "calculator":
            m = re.search(r"\(?\s*\d[\d\s+\-*/().]*\d\s*\)?", task)
            if not m:
                return None
            return {"tool": tool, "args": {arg_key: m.group(0).strip()}}

        if tool == "python_exec":
            body = task
            for verb in ("执行代码", "运行代码", "执行", "运行", "跑一下", "跑"):
                if body.startswith(verb):
                    body = body[len(verb):]
                    break
            body = body.strip().strip("：: ")
            if not body:
                return None
            # 诊断（不纠正）：planner 选了 python_exec，但任务里没有代码特征
            if not _looks_like_code(body):
                print(f"⚠️ [诊断] planner 选了 python_exec，但参数看起来不像代码: {body[:60]!r}")
            return {"tool": tool, "args": {arg_key: body}}

        m_arg = re.search(r"参数\s*[:：]\s*(.+)", plan)
        if m_arg:
            raw = m_arg.group(1).strip().strip("。.、,，\"'“”")
            words = _tokenize_for_match(task)
            if raw in words:
                return {"tool": tool, "args": {arg_key: raw}}
            for w in sorted(words, key=len, reverse=True):
                if w in raw and len(w) > 1:
                    return {"tool": tool, "args": {arg_key: w}}

        words = _tokenize_for_match(task)
        cands = re.findall(r'["\'“"]([^"\'“”]{1,120})["\'“”]', plan)
        cands += re.findall(r"[A-Za-z_][\w.\-]*", plan)
        cands += re.findall(r"[\u4e00-\u9fa5]{2,8}", plan)
        best = ""
        for c in cands:
            c = c.strip()
            if len(c) > len(best) and c in words:
                best = c
        if not best:
            return None
        return {"tool": tool, "args": {arg_key: best}}

    def generate(self, messages: List[Message],
                 config: Optional[GenerationConfig] = None) -> str:
        task = self._task_of(messages)
        if not task:
            return "{}"

        plan = self._call_planner(task)
        self.last_plan = plan
        if self.verbose:
            print(f"[cascade] 阶段1 推理:\n{plan[:400]}")

        hit = self._extract(plan, task)
        if hit:
            self.last_source = "rule"
            out = json.dumps(
                {"thought": plan.strip()[:120] or "已分析任务",
                 "action": {"tool": hit["tool"], "args": hit["args"]}},
                ensure_ascii=False)
            if self.verbose:
                print(f"[cascade] 阶段2 规则提取 → {out}")
            return out

        if self.formatter is not None:
            try:
                out = self.formatter.generate(messages, config)
                self.last_source = "formatter"
                if self.verbose:
                    print(f"[cascade] 阶段2 回退 formatter → {out}")
                return out
            except Exception as e:
                if self.verbose:
                    print(f"[cascade] formatter 失败: {e}")

        self.last_source = "fallback"
        return json.dumps({"thought": "无法从分析中提取工具调用",
                           "final": ""}, ensure_ascii=False)

    def _call_planner(self, task: str) -> str:
        msgs = [{"role": "system", "content": planner_prompt(self.tools)},
                {"role": "user", "content": task}]
        cfg = GenerationConfig(temperature=0.2, max_tokens=512)
        last = None
        for call in (lambda: self.planner.generate(msgs, config=cfg),
                     lambda: self.planner.generate(msgs)):
            try:
                out = call()
                if out and out.strip():
                    return out
                last = "返回空"
            except TypeError as e:
                last = e
                continue
            except Exception as e:
                self.last_source = "planner_error"
                if self.verbose:
                    print(f"[cascade] planner 故障: {type(e).__name__}: {e}")
                return ""
        self.last_source = "planner_error"
        if self.verbose:
            print(f"[cascade] planner 无有效输出: {last}")
        return ""

    @staticmethod
    def _task_of(messages: List[Message]) -> str:
        users = [m["content"] for m in messages if m.get("role") == "user"]
        if not users:
            return ""
        task = users[0]
        for marker in ("（上一轮失败", "\n你的子任务", "\n总体目标"):
            if marker in task:
                task = task.split(marker)[0]
        return task.strip()

    def describe(self) -> str:
        f = self.formatter.describe() if self.formatter else "（无，纯规则）"
        return f"Cascade: planner={self.planner.describe()} | 协议层={f}"


def build_cascade(planner_kind: str = "mock", api_model: str = "deepseek-chat",
                  formatter_path: str = "models/mnemo_sonnet_evo5.npz",
                  verbose: bool = False) -> CascadeBackend:
    if planner_kind == "api":
        from sea.model.api import APIBackend
        planner = APIBackend(model=api_model)
    elif planner_kind == "local":
        from sea.model.local import HFBackend
        planner = HFBackend(api_model)
    else:
        from sea.model.mock import MockBackend
        planner = MockBackend()

    formatter = None
    if formatter_path:
        try:
            formatter = NumpyBackend(formatter_path, constrained=True)
        except Exception as e:
            print(f"⚠️ 协议层加载失败，退化为纯规则: {e}")
    return CascadeBackend(planner=planner, formatter=formatter, verbose=verbose)
