# Mnemo-Sonnet 记忆驱动的自我进化模型

> 记忆驱动的自我进化模型：能力不只在权重里，而在「权重 × 记忆 × 技能」的闭环中生长。

## 快速开始

```bash
python scripts/run_demo.py
```

## 核心模块

- `sea/agent/` — ReAct Agent + 工具调用
- `sea/memory/` — 记忆引擎（工作/情景/语义/程序四层）
- `sea/evolution/` — 自动进化闭环
- `sea/fusion/` — 融合层
- `sea/orch/` — 多 Agent 编排

## 模型卡

详见 [MODEL_CARD.md](MODEL_CARD.md)
