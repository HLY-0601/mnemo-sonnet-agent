#!/usr/bin/env python3
"""两阶段级联验证：推理模型负责「想」，协议层负责「写格式」。

⚠️ 本脚本在沙盒里**只跑过模拟 planner**，没跑过真模型 —— 沙盒无 torch、
无外网。真实效果请在本地验证。

## 为什么这样设计
推理模型（R1 / R1-Distill / QwQ）的训练目标是「把题想明白」，不是「按格式
输出函数调用」。让它直接写 JSON 是跟天性对着干。级联的思路是顺着它：

    阶段1  planner  —— 只要求「想清楚该干什么」，输出自然语言
    阶段2  协议层   —— 把结论转成规范 JSON（规则优先，兜底用本地 evo5）

## 本地跑法
```bash
# A. DeepSeek 官方 API 当大脑（推荐，原生支持推理）
export SEA_API_KEY=sk-xxx
python scripts/cascade_demo.py --planner api --model deepseek-chat

# B. 本地模型当大脑（需要 torch + transformers）
python scripts/cascade_demo.py --planner local \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# C. 只用本地小模型（对照组，不级联）
python scripts/cascade_demo.py --planner none
```

## 判读
- 级联 ≥ 协议层单独：架构有效
- 级联随 planner 变强而提升：说明收益来自「想得清楚」
- 级联反而更差：planner 判断有误，看规则的来源分布定位
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sea.model.base import ModelBackend  # noqa: E402
from sea.model.cascade import CascadeBackend  # noqa: E402
from sea.model.numpy_backend import NumpyBackend  # noqa: E402


# 真·分布外用例：生造词，训练语料里绝不可能出现
CASES = [
    ("计算 (132 + 868) * 3 等于多少？", "calculator", {"expression": "(132 + 868) * 3"}),
    ("算一算 (25000 - 12345) * 4", "calculator", {"expression": "(25000 - 12345) * 4"}),
    ("(999 - 111) / 3 得多少？", "calculator", {"expression": "(999 - 111) / 3"}),
    ("什么是Quuxblort？", "search", {"query": "Quuxblort"}),
    ("查查 Zyzzogeton", "search", {"query": "Zyzzogeton"}),
    ("了解 Flibbertigibbet", "search", {"query": "Flibbertigibbet"}),
    ("什么是Nginx？", "search", {"query": "Nginx"}),
    ("给我讲讲 PostgreSQL", "search", {"query": "PostgreSQL"}),
    ("把结果写入 report.md", "file_write", {"path": "report.md"}),
    ("创建文件 Dockerfile.prod", "file_write", {"path": "Dockerfile.prod"}),
    ("导出为 quarterly_summary.xlsx", "file_write", {"path": "quarterly_summary.xlsx"}),
    ("执行 print(1+1)", "python_exec", {"code": "print(1+1)"}),
    ("跑 print(set([1,2,3]))", "python_exec", {"code": "print(set([1,2,3]))"}),
]


class _DemoPlanner(ModelBackend):
    """模拟「想得清楚但不写 JSON」的推理模型 —— R1-Distill 的真实行为。

    ⚠️ 这是**模拟**，不是真模型：它按关键词给出正确分析，代表 planner 的
       「上界」。用它验证的是架构与协议层，不是某个真实模型的能力。
       真模型请用 --planner api / local。
    """

    name = "demo-planner"
    model_name_or_path = "demo"

    def describe(self):
        return "模拟推理模型（想得清楚、不写JSON 的 R1 类行为）"

    def generate(self, messages, config=None):
        t = messages[-1]["content"]
        if any(k in t for k in ("什么是", "查查", "了解", "讲讲", "介绍")):
            w = re.findall(r"[A-Za-z][A-Za-z0-9_.\-]*", t)
            return f"工具名：search\n参数：{w[-1] if w else '未知'}"
        if any(k in t for k in ("计算", "算", "得多少", "等于多少")):
            m = re.search(r"\(?\s*\d[\d\s+\-*/().]*\d\s*\)?", t)
            return f"工具名：calculator\n参数：{m.group(0).strip() if m else ''}"
        if any(k in t for k in ("写入", "创建", "导出", "新建", "生成")):
            f = re.findall(r"[\w\-.]+\.\w+", t)
            return f"工具名：file_write\n参数：{f[0] if f else 'output.txt'}"
        if any(k in t for k in ("执行", "运行", "跑")):
            body = t
            for v in ("执行代码", "运行代码", "执行", "运行", "跑一下", "跑"):
                if body.startswith(v):
                    body = body[len(v):]
                    break
            return f"工具名：python_exec\n参数：{body.strip()}"
        return "工具名：未知"


def build(kind: str, model: str, evo_path: str, verbose: bool):
    evo = NumpyBackend(evo_path, constrained=True) if os.path.exists(evo_path) else None
    if evo is None:
        print(f"⚠️ 找不到协议层权重 {evo_path}，退化为纯规则模式")
    if kind == "none":
        return evo
    if kind == "api":
        from sea.model.api import APIBackend
        planner = APIBackend(model=model)
    elif kind == "local":
        from sea.model.local import HFBackend
        planner = HFBackend(model)
    elif kind == "demo":
        planner = _DemoPlanner()
        print("⚠️ demo planner 是模拟的推理模型（上界），不是真模型")
    elif kind == "mock":
        from sea.model.mock import MockBackend
        planner = MockBackend()
    else:
        raise SystemExit(f"未知 planner: {kind}")
    return CascadeBackend(planner=planner, formatter=evo, verbose=verbose)


def score(backend, label: str, show_plan: bool = False):
    if backend is None:
        print(f"{label}: 无可用后端")
        return 0
    ok, src = 0, {}
    print(f"\n--- {label} ---")
    print(f"  {backend.describe()[:110]}\n")
    for task, tool, args in CASES:
        raw = backend.generate([{"role": "user", "content": task}])
        try:
            g = json.loads(raw)
            hit = g.get("action") == {"tool": tool, "args": args}
        except Exception:
            hit = False
        ok += hit
        s = getattr(backend, "last_source", "-")
        src[s] = src.get(s, 0) + 1
        print(f"  {'✓' if hit else '✗'} [{s:9s}] {task[:34]}")
        if not hit:
            print(f"      期望 {args}")
            print(f"      得到 {raw[:100]}")
        if show_plan and getattr(backend, "last_plan", ""):
            print(f"      planner: {backend.last_plan[:150]}")
    print(f"\n  {ok}/{len(CASES)} = {ok / len(CASES):.0%}   阶段2来源: {src}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", default="mock",
                    choices=["api", "local", "demo", "mock", "none"])
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--evo", default="models/mnemo_sonnet_evo5.npz")
    ap.add_argument("--verbose", action="store_true", help="打印 planner 的原始推理")
    args = ap.parse_args()

    backend = build(args.planner, args.model, args.evo, args.verbose)
    if args.planner != "none":
        score(backend, f"级联（planner={args.planner}/{args.model}）", args.verbose)
        # 对照：只用协议层
        if os.path.exists(args.evo):
            score(NumpyBackend(args.evo, constrained=True), "对照：协议层单独")
    else:
        score(backend, "协议层单独")

    print("\n判读：级联 ≥ 协议层单独 = 架构有效；"
          "看「阶段2来源」里 rule 的占比，越高说明 planner 想得越清楚。")


if __name__ == "__main__":
    main()
