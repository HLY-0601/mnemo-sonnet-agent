"""ReAct Agent：思考 → 行动 → 观察 → 反思（Reflexion）→ 复盘沉淀。"""
from __future__ import annotations
import inspect, json, sys, os, re, time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from ..memory import MemoryEngine
from ..model.base import GenerationConfig, Message, ModelBackend, extract_json
from ..types import EventType, MemoryLayer, Trajectory, new_id
from ..util import safe_exec, truncate
from . import prompts
from .tools import run_tool, tools_prompt

@dataclass
class AgentConfig:
    max_steps: int = 6
    max_retries: int = 1
    temperature: float = 0.2
    max_tokens: int = 512
    reflect_on_fail: bool = True
    inject_memory: bool = True
    skill_route_min_confidence: float = 0.7
    skill_route_min_score: float = 0.55
    fresh_workspace: bool = True

class ReActAgent:
    def __init__(self, backend, memory=None, name="agent", config=None):
        self.backend = backend
        self.memory = memory
        self.name = name
        self.cfg = config or AgentConfig()

    def run(self, task, expected=None, session_id=None, metadata=None):
        session_id = session_id or new_id("sess")
        traj = Trajectory(task=task, session_id=session_id, agent_id=self.name,
                          model_version=self.backend.version, metadata=metadata or {})
        traj.add(EventType.TASK, {"task": task})
        if self.cfg.fresh_workspace:
            import shutil
            from .tools import get_workspace_root
            root = get_workspace_root()
            if os.path.isdir(root):
                shutil.rmtree(root, ignore_errors=True)
            os.makedirs(root, exist_ok=True)
        memory_block = ""
        if self.memory and self.cfg.inject_memory:
            self.memory.remember(task, layer=MemoryLayer.WORKING, kind="task", salience=0.4)
            memory_block = self.memory.build_context(task, k=4) or "（暂无相关记忆）"
        routed, result = self._try_skill_route(task)
        if routed:
            traj.add(EventType.ACTION, {"tool": f"skill:{result[0]}", "args": result[2]})
            traj.add(EventType.OBSERVATION, {"result": result[1]})
            traj.final_answer = result[1]
            traj.success = self._judge(traj.final_answer, expected)
            traj.score = 1.0 if traj.success else 0.4
            traj.add(EventType.FINAL, {"answer": traj.final_answer})
            self._persist(traj)
            return traj
        attempt = 0
        while True:
            attempt += 1
            messages = self._build_messages(task, memory_block, traj)
            for step in range(self.cfg.max_steps):
                raw = self.backend.generate(messages, GenerationConfig(
                    temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens))
                try:
                    parsed = extract_json(raw)
                except ValueError:
                    traj.add(EventType.ERROR, {"raw": raw[:300], "reason": "JSON 解析失败"})
                    hint = raw.strip()[:80].replace("\n", "\\n")
                    messages.append({"role": "user", "content":
                                     f"你的输出无法解析为 JSON（收到的是：{hint!r}）。"
                                     f"请只输出一个 JSON 对象，不要有任何前缀、解释或 markdown 代码块。"})
                    continue
                thought = parsed.get("thought", "")
                action = parsed.get("action") or {}
                tool, args = action.get("tool", ""), action.get("args") or {}
                if tool and tool != "finish":
                    ev = traj.add(EventType.ACTION, {"thought": thought, "tool": tool, "args": args})
                    try:
                        obs = run_tool(tool, **args)
                    except Exception as e:
                        obs = f"工具调用失败：{type(e).__name__}: {e}"
                    traj.add(EventType.OBSERVATION, {"tool": tool, "result": obs}, parent=ev.id)
                    messages.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
                    is_last = step >= self.cfg.max_steps - 2
                    nudge = prompts.FINALIZE if is_last else prompts.STEP_HINT
                    messages.append({"role": "user", "content": prompts.OBSERVATION.format(result=obs)})
                    messages.append({"role": "user", "content": nudge})
                    continue
                if tool == "finish":
                    traj.final_answer = str(args.get("answer", ""))
                    traj.add(EventType.FINAL, {"thought": thought, "answer": traj.final_answer})
                    break
                if "final" in parsed:
                    traj.add(EventType.FINAL, {"thought": thought, "answer": parsed["final"]})
                    traj.final_answer = str(parsed["final"])
                    break
                traj.add(EventType.ERROR, {"raw": raw[:300], "reason": "既无 action 也无 final"})
                messages.append({"role": "user", "content": "每条输出必须包含 action（调用工具）或 final（给出最终答案）之一。"})
            else:
                traj.add(EventType.ERROR, {"reason": "达到最大步数仍未作答"})
                if not traj.final_answer:
                    traj.final_answer = self._best_observation(traj)
            traj.success = self._judge(traj.final_answer, expected)
            traj.score = self._score(traj)
            if not traj.success and self.cfg.reflect_on_fail and attempt <= self.cfg.max_retries:
                reason = self._failure_reason(traj, expected)
                reflection = self._reflect(task, traj.final_answer, reason)
                traj.reflections.append(reflection)
                traj.add(EventType.REFLECTION, {"content": reflection})
                memory_block += f"\n【上一次失败的经验】{reflection}"
                traj.metadata["prev_answer"] = traj.final_answer
                traj.final_answer = ""
                continue
            break
        traj.add(EventType.SYSTEM, {"attempts": attempt, "score": traj.score})
        self._persist(traj)
        return traj

    def _try_skill_route(self, task):
        if not self.memory:
            return False, ()
        hits = self.memory.skills.search(task, top_k=1, min_confidence=self.cfg.skill_route_min_confidence)
        if not hits:
            return False, ()
        skill, score = hits[0]
        if score < self.cfg.skill_route_min_score:
            return False, ()
        kwargs = self._infer_skill_args(skill, task)
        if kwargs is None:
            return False, ()
        ok, _ = self.memory.skills.validate(skill)
        if not ok:
            return False, ()
        result, msg = self.memory.skills.execute(skill.name, **kwargs)
        if result is None:
            return False, ()
        return True, (skill.name, f"{msg} → 结果: {result}", kwargs)

    @staticmethod
    def _infer_skill_args(skill, task):
        ns, _ = safe_exec(skill.code)
        run = ns.get("run")
        if run is None:
            return None
        kwargs = {}
        numbers = [float(x) for x in re.findall(r"\d+\.?\d*", task)]
        required = [n for n, p in inspect.signature(run).parameters.items() if p.default is inspect._empty]
        for i, pname in enumerate(required):
            if i < len(numbers):
                kwargs[pname] = numbers[i]
            else:
                return None
        return kwargs

    @staticmethod
    def _judge(answer, expected):
        if not answer:
            return False
        if expected is None:
            return None
        a, e = str(answer), str(expected)
        if e.strip().lower() in a.strip().lower():
            return True
        nums_a = set(re.findall(r"\d+\.?\d*", a))
        nums_e = set(re.findall(r"\d+\.?\d*", e))
        return bool(nums_e) and nums_e.issubset(nums_a)

    @staticmethod
    def _score(traj):
        base = 1.0 if traj.success else 0.2
        if traj.success is None:
            base = 0.6
        eff = max(0.0, 1.0 - 0.08 * max(0, traj.n_steps - 2))
        err = 0.9 if any(e.type == EventType.ERROR for e in traj.events) else 1.0
        return round(max(0.0, min(1.0, base * 0.75 + eff * 0.25) * err), 3)

    @staticmethod
    def _failure_reason(traj, expected):
        if expected:
            return f"标准答案是 {expected}，你的回答是 {truncate(traj.final_answer, 120)}"
        return "未得出可靠结论"

    def _reflect(self, task, answer, reason):
        messages = [{"role": "user", "content": prompts.REFLECT.format(
            task=task, answer=truncate(answer, 300), reason=reason)}]
        raw = self.backend.generate(messages, GenerationConfig(temperature=0.1))
        try:
            d = extract_json(raw)
        except ValueError:
            return raw[:300]
        return "；".join(str(d.get(k, "")) for k in ("diagnosis", "lesson", "next_plan") if d.get(k))

    _INFO_TOOLS = ("calculator", "python_exec", "search", "file_read", "memory_search", "skill")
    _RECEIPT_TOOLS = ("file_write", "memory_remember", "finish")

    def _best_observation(self, traj):
        obs = [e.content for e in traj.events if e.type == EventType.OBSERVATION]
        if not obs:
            return ""
        def rank(c):
            tool = str(c.get("tool", ""))
            if any(t in tool for t in self._INFO_TOOLS):
                return (0, tool)
            if any(t in tool for t in self._RECEIPT_TOOLS):
                return (2, tool)
            return (1, tool)
        for tier in (0, 1, 2):
            hit = [c for c in obs if rank(c)[0] == tier]
            if hit:
                return str(hit[-1].get("result", ""))
        return ""

    def _build_messages(self, task, memory_block, traj):
        system = prompts.SYSTEM.format(tools=tools_prompt(), memory=memory_block or "（无）", fewshot=prompts.fewshot_prompt())
        messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
        if traj.reflections:
            prev = traj.metadata.get("prev_answer") or "（空）"
            messages.append({"role": "user", "content": f"（上一轮失败，答案是：{truncate(prev, 150)}；请吸取上面的经验重新作答）"})
        return messages

    def _persist(self, traj):
        if not self.memory:
            return
        summary = f"任务：{truncate(traj.task, 120)} | 结论：{truncate(traj.final_answer, 200)} | 成功={traj.success} 分数={traj.score} 步数={traj.n_steps}"
        self.memory.remember(summary, layer=MemoryLayer.EPISODIC, kind="episode",
                             meta={"traj_id": traj.id, "score": traj.score},
                             salience=0.3 + 0.6 * traj.score)
        for r in traj.reflections:
            self.memory.remember(r, layer=MemoryLayer.SEMANTIC, kind="reflection",
                                 meta={"traj_id": traj.id}, salience=0.75)
