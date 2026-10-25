### 实测结果（13 例真 OOD）

| 配置 | 正确率 |
|---|---|
| 协议层单独（evo5 + 三层约束） | 62% |
| **级联（planner + 协议层）** | **100%** |

13 例全部走 rule 提取，零 formatter 回退——规则层是确定性的，planner 想得清楚，协议层就不用干活。

### 三条关键性质（都测了）

**1. 不会比不用更差**
用「只说工具名、不给参数」的弱 planner 测：级联 = 单独 = 71%。planner 想不清楚时自动回退，不会自信地输出错误。

**2. 规则层不编造**
铁律：参数必须是任务里的完整词。用故意编造参数的恶意 planner 验证过——规则层拒绝，回退协议层。**planner 可以想，不能编。**

**3. 收益随 planner 变强**
100% 是模拟 planner（上界），代表架构有效。真实 R1 / DeepSeek 效果待本地验证。

### 又踩了同一个坑（第三次）
第一版用 `value in task` 判断参数合法性。planner 说「调用 search 工具」，任务是「什么是Elasticsearch？」——`search` 恰好是 `Elasticsearch` 的子串，朴素检查放行了。

第二版用 `re.split(r"[^\w.\-]+")` 切词，但 Python 的 `\w` 默认匹配中文，「什么是Nginx」被当成一整块，`Nginx` 永远匹配不上。

最终修法：显式按 ASCII 词 / 中文字符分词，中文视为分隔符。「字符串匹配要用词边界」这个教训在本项目第三次出现（前两次：约束解码起点、路径剥离）。

### 诚实的保留
100% 用的是模拟 planner，不是真模型。沙盒无 torch 也无外网，真实效果得在本地验证：
```bash
export SEA_API_KEY=sk-xxx
python scripts/cascade_demo.py --planner api --model deepseek-chat
python scripts/cascade_demo.py --planner local --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```
跑真模型后看「rule 占比」就知道那个大脑想得清不清楚。