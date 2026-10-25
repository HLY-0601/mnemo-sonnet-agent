## 一·补六、两阶段级联后端：推理模型负责「想」，协议层负责「写格式」

### 问题
推理模型（R1 / R1-Distill / QwQ）的训练目标是「把题想明白」，不是「按格式输出函数调用」。让它直接写 JSON 是跟天性对着干：

- 实测 R1-Distill-1.5B 心算 (132+868)*3 得 3000（**算对了**），却输出 `\\boxed{3000}` 而不是 JSON
- 加 few-shot、加规则、甚至 LoRA 微调都压不住它的 CoT

### 思路：别对抗，顺着它
```
阶段1  planner   —— 只要求「想清楚该干什么」，输出自然语言（明确说不要 JSON）
阶段2  协议层    —— 把结论转成规范 JSON（规则优先，兜底用本地小模型）
```
数学家负责解题，秘书负责填表格。

### 实测结果（13 例真 OOD）

| 配置 | 正确率 |
|---|---|
| 协议层单独（evo5 + 三层约束） | 62% |
| **级联（planner + 协议层）** | **92%** |

92% 时 rule 占 11/13——规则提取是确定性的，planner 想得越清楚，协议层越不用干活。

### 三条关键性质（都测了）

**1. 不会比不用更差**
用「只说工具名、不给参数」的弱 planner 测：级联 = 单独 = 71%。planner 想不清楚时自动回退，不会自信地输出错误。

**2. 规则层不编造**
铁律：参数必须是任务里的完整词。用故意编造参数的恶意 planner 验证过——规则层拒绝，回退协议层。**planner 可以想，不能编。**

**3. 收益随 planner 变强**
92% 是模拟 planner（上界），代表架构有效。真实 R1 / DeepSeek 效果待本地验证。

### 又踩了同一个坑（第三次）
第一版用 `value in task` 判断参数合法性。planner 说「调用 search 工具」，任务是「什么是Elasticsearch？」——`search` 恰好是 `Elasticsearch` 的子串，朴素检查放行了。

改成整词匹配才修好。「字符串匹配要用词边界」这个教训在本项目第三次出现（前两次：约束解码起点、路径剥离）。

### 诚实的保留
92% 用的是模拟 planner，不是真模型。沙盒无 torch 也无外网，真实效果得在本地验证：
```bash
export SEA_API_KEY=sk-xxx
python scripts/cascade_demo.py --planner api --model deepseek-chat
python scripts/cascade_demo.py --planner local --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```
跑真模型后看「rule 占比」就知道那个大脑想得清不清楚。