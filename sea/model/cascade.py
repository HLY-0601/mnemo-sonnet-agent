"""两阶段级联后端：推理模型负责「想」，协议层负责「写格式」。

## 为什么需要它
推理模型（DeepSeek-R1 / R1-Distill / QwQ）的训练目标是「把题想明白」，
不是「按格式输出函数调用」。让它们直接输出工具调用 JSON 是在跟天性对着干：
- 实测 R1-Distill-1.5B 心算 (132+868)*3 得 3000（**算对了**），
  却输出 `\\boxed{3000}` 而不是 JSON —— 它做对了，只是没写成协议要求的格式。
- 加 few-shot、加规则、甚至 LoRA 微调都压不住它的 CoT。

## 思路：别对抗，顺着它
```
阶段1  planner   —— 只要求「想清楚该干什么」，输出自然语言
阶段2  协议层    —— 把结论转成规范 JSON（规则优先，兜底用本地小模型）
```
数学家负责解题，秘书负责填表格。

## 用法
```python
from sea.model.cascade import CascadeBackend
from sea.model.api import APIBackend
from sea.model.numpy_backend import NumpyBackend

# 最强的组合：DeepSeek 官方 API 当大脑（原生支持推理），本地 evo5 当协议层
brain = APIBackend("deepseek-chat")          # 或 deepseek-reasoner
scribe = NumpyBackend("models/mnemo_sonnet_evo5.npz")
agent_backend = CascadeBackend(planner=brain, formatter=scribe)
```
```bash
export SEA_API_KEY=sk-xxx
python scripts/run_demo.py --backend cascade
```
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from sea.model.base import GenerationConfig, Message, ModelBackend
from sea.model.numpy_backend import NumpyBackend


# 阶段1 的提示：明确说「不要输出 JSON」，只要结论。
# 这是级联的关键 —— 不再要求推理模型做它不擅长的事。
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

用户：计算 (17 + 88) * 4 等于多少？
工具名：calculator
参数：(17 + 88) * 4

不要输出 JSON，不要自己计算，不要解释。只要上面两行。"""


def planner_prompt() -> str:
    """生成 planner 的 system prompt：工具清单 + 格式示例。

    两处都是踩过坑才加的：
    1. **工具名清单** —— 第一版只说"应该调用哪个工具"却没给清单，
       真模型（Qwen2.5-0.5B）于是用中文自由描述，规则层吃不到工具名，
       10/13 全部回退。这和 Agent 那边"幻觉工具名靠 schema 修"是同一个教训。
    2. **示例而非规则** —— 本项目已验证：小模型服从示例不服从规则。
       所以给两条示例，而不是只写"按格式输出"。

    格式上刻意用"工具名：/参数："两行，而不是完整 JSON ——
    推理模型的弱项是结构化输出，但填两行标签是它做得到的。
    """
    try:
        from sea.agent.tools import tools_prompt
        tools = tools_prompt()
    except Exception:
        tools = ("- calculator(expression): 计算算术表达式\n"
                 "- search(query): 检索资料\n"
                 "- file_write(path): 写文件\n"
                 "- file_read(path): 读文件\n"
                 "- python_exec(code): 执行 Python 代码")
    return _PLANNER_TEMPLATE.format(tools=tools)


# 工具名 → (识别关键词, 参数键)
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
    """把文本切成候选词：连续的 ASCII 词（含 . _ -）算一个词，中文单独成词。

    关键点：**中文算分隔符**。Python 的 `\\w` 默认匹配中文，
    `re.split(r"[^\\w.\\-]+")` 会把 "什么是Nginx" 当成一整块，
    导致 "Nginx" 永远匹配不上（实测踩过）。
    """
    out, buf = [], ""
    for ch in text:
        if ord(ch) < 128 and (ch.isalnum() or ch in "._-"):
            buf += ch
        else:
            if buf:
                out.append(buf)
                buf = ""
            if ord(ch) >= 128 and not ch.isspace():
                # 连续中文作为整体（用于中文参数），但也可逐字匹配
                out.append(ch)
    if buf:
        out.append(buf)
    return out


class CascadeBackend(ModelBackend):
    """planner（会想） + 协议层（会写）级联。

    阶段2 分两条路：
    - **规则提取**（优先）：planner 说清了工具与参数，且参数原样出现在任务里
      → 直接拼 JSON。零模型调用，100% 忠于输入。
    - **formatter 兜底**：规则没抓到 → 交给本地小模型，用约束解码生成。
    """

    name = "cascade"

    def __init__(self, planner: ModelBackend, formatter: Optional[ModelBackend] = None,
                 verbose: bool = False):
        self.planner = planner
        self.formatter = formatter
        self.verbose = verbose
        self.model_name_or_path = f"cascade({planner.name}+{formatter.name if formatter else 'rule'})"
        self._version = "cascade"
        self.last_plan = ""
        self.last_source = ""     # "rule" | "formatter" | "fallback"

    # ---------------- 阶段2：从推理结论提取协议 ----------------
    @staticmethod
    def _extract(plan: str, task: str) -> Optional[Dict[str, Any]]:
        """从 planner 的自然语言结论里提取工具与参数。

        铁律：参数值必须原样出现在用户任务里。
        这是整个级联的安全边界 —— planner 可以「想」，但不能「编」。

        识别顺序（从明确到模糊）：
        1. `工具名：xxx` 半结构化字段（新 prompt 要求的格式，最可靠）
        2. planner 文本里出现过的合法工具名
        3. 中文关键词兜底
        """
        # 0) 优先：planner 按格式写的「工具名：xxx / 参数：xxx」
        m_tool = re.search(r"工具名\s*[:：]\s*([A-Za-z_]\w*)", plan)
        tool = m_tool.group(1) if m_tool else None
        if tool is not None and tool not in _ARG_KEY_OF:
            tool = None      # 自造的工具名 → 不认，回退兜底

        # 1) 工具名：planner 明确写出的合法工具名
        if tool is None:
            for name in _ARG_KEY_OF:
                if re.search(rf"\b{name}\b", plan):
                    tool = name
                    break

        # 2) 退化：按中文关键词猜
        if tool is None:
            for name, kws in _TOOL_HINTS:
                if any(k in plan for k in kws):
                    tool = name
                    break

        if tool is None:
            return None

        arg_key = _ARG_KEY_OF[tool]

        # calculator / python_exec 的值是表达式或代码，含空格与括号，
        # 不能用「整词匹配」——会把 "(132 + 868) * 3" 拆成 "132"。
        # 所以这两个专用分支必须排在通用的「参数：」提取之前。
        if tool == "calculator":
            # 算术：从任务里抠表达式（允许 planner 算出的结果，但表达式必须来自任务）
            m = re.search(r"\(?\s*\d[\d\s+\-*/().]*\d\s*\)?", task)
            if not m:
                return None
            return {"tool": tool, "args": {arg_key: m.group(0).strip()}}

        if tool == "python_exec":
            # 代码：剥掉开头的动词，剩下的原样作为代码。
            # evo5 在这类上最差（括号复制错位，如 "print(set([1,2,3])))))"），
            # 而代码本身在任务里是完整的，直接切出来即可 —— 仍是任务的子串。
            body = task
            for verb in ("执行代码", "运行代码", "执行", "运行", "跑一下", "跑"):
                if body.startswith(verb):
                    body = body[len(verb):]
                    break
            body = body.strip().strip("：: ")
            if not body:
                return None
            return {"tool": tool, "args": {arg_key: body}}

        # 0b) 参数：planner 按格式写的「参数：xxx」—— 仍必须是任务里的完整词
        #     （只对 search / file_* 等「值是一个词」的工具生效；
        #      calculator / python_exec 已在上面专用分支返回）
        m_arg = re.search(r"参数\s*[:：]\s*(.+)", plan)
        if m_arg:
            raw = m_arg.group(1).strip().strip("。.、,，\"'“”")
            words = _tokenize_for_match(task)
            if raw in words:
                return {"tool": tool, "args": {arg_key: raw}}
            # 参数里含多个词时取最长的那个（如 "report.md 的内容"）
            for w in sorted(words, key=len, reverse=True):
                if w in raw and len(w) > 1:
                    return {"tool": tool, "args": {arg_key: w}}

        # 其它：参数必须是任务里的【完整词】。
        #
        # ⚠️ 两处都踩过坑，都是同一个根因（字符串匹配必须用词边界）：
        #   1. 朴素 `in` 子串判断会放行陷阱 —— planner 说「调用 search 工具」，
        #      任务里是 "什么是Elasticsearch？"，"search" 恰好是其子串，
        #      会被错当成参数。
        #   2. `re.split(r"[^\w.\-]+")` 切不开中文+英文 —— Python 的 \w 默认
        #      匹配中文，"什么是Nginx" 会被当成一整个词，导致 Nginx 匹配不上。
        # 所以这里显式按「ASCII 词 / 中文串」分词，中文视为分隔符。
        words = _tokenize_for_match(task)

        cands = re.findall(r'["\'“”]([^"\'“”]{1,120})["\'“”]', plan)
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

    # ---------------- 主入口 ----------------
    def generate(self, messages: List[Message],
                 config: Optional[GenerationConfig] = None) -> str:
        task = self._task_of(messages)
        if not task:
            return "{}"

        # 阶段1：让推理模型想，明确不要求 JSON
        plan = self._call_planner(task)
        self.last_plan = plan
        if self.verbose:
            print(f"[cascade] 阶段1 推理:\n{plan[:400]}")

        # 阶段2a：规则提取（忠于输入，零幻觉）
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

        # 阶段2b：交给协议层（本地小模型 + 约束解码）
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

        # 兜底：不编造，报无法解析，让 Agent 重试
        self.last_source = "fallback"
        return json.dumps({"thought": "无法从分析中提取工具调用",
                           "final": ""}, ensure_ascii=False)

    def _call_planner(self, task: str) -> str:
        """调 planner，容忍不同的 generate 签名。

        接口约定是 `generate(messages, config=None)`，但第三方后端可能只接受
        `generate(messages)`。带 config 失败时降级重试 —— 否则会静默退化成
        「planner 没起作用」，表面看只是打分低，很难排查（实测踩过）。
        """
        msgs = [{"role": "system", "content": planner_prompt()},
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
            except TypeError as e:      # 签名不兼容 → 降级重试
                last = e
                continue
            except Exception as e:      # 网络/鉴权等真实故障 → 不再重试
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
        """取原始任务（第一条 user 消息），剥掉 Agent 附加的说明。"""
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
    """按名字构造级联后端，供 CLI / demo 使用。"""
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
