#!/usr/bin/env python3
"""Mnemo-Sonnet 独立推理脚本（默认 evo.5，三个版本通用） —— 只需要 numpy，不需要 torch / transformers / sea 包。

    python mnemo_load.py                                  # 默认加载 evo.5
    python mnemo_load.py --model mnemo_sonnet_evo1.npz --task "计算 (17 + 88) * 4 等于多少？"
    python mnemo_load.py --model mnemo_sonnet_evo1.npz --batch      # 跑内置样例集看准确率

模型学会了什么：Agent 的决策协议 —— 给定任务，输出结构化的工具调用 JSON。
它自己决定「用哪个工具、为什么用、参数值从输入里抄哪一段」，不是背答案。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

SPECIALS = ["<pad>", "<eos>", "<bos>"]
PAD, EOS, BOS = SPECIALS

DEFAULT_TOKEN_RE = r"\s|[^\s]"
_TOKEN_RE = re.compile(DEFAULT_TOKEN_RE, re.UNICODE)


def tokenize(text: str):
    return _TOKEN_RE.findall(text)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)


def _ln_forward(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    std = np.sqrt((xc ** 2).mean(axis=-1, keepdims=True) + eps)
    xn = xc / std
    return xn * g + b


def _split_heads(x, n_head):
    B, T, D = x.shape
    return x.reshape(B, T, n_head, D // n_head).transpose(0, 2, 1, 3)


def _merge_heads(x):
    B, H, T, Dh = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, H * Dh)


class MnemoModel:
    ARG_KEYS = ("expression", "query", "path", "code", "content", "command")

    def __init__(self, P, cfg, itos):
        self.P, self.cfg, self.itos = P, cfg, itos
        self.stoi = {c: i for i, c in enumerate(itos)}

    @classmethod
    def load(cls, path: str):
        z = np.load(path, allow_pickle=True)
        cfg = json.loads(str(z["__cfg__"]))
        itos = json.loads(str(z["__vocab__"]))
        meta = json.loads(str(z["__meta__"]))
        P = {}
        for k in z.files:
            if k.startswith("P::"):
                P[k[3:]] = z[k].astype(np.float64)
        pat = meta.get("tokenizer_pattern")
        if pat:
            global _TOKEN_RE
            _TOKEN_RE = re.compile(pat, re.UNICODE)
        else:
            print(f"⚠️ 模型文件未存 tokenizer_pattern，回退到兜底分词器 "
                  f"{DEFAULT_TOKEN_RE!r}。若输出异常，请确认该权重的训练分词方式。",
                  file=sys.stderr)
        return cls(P, cfg, itos), meta

    def forward(self, idx: np.ndarray) -> np.ndarray:
        B, T = idx.shape
        D, H = self.cfg["d_model"], self.cfg["n_head"]
        P = self.P
        x = P["Wte"][idx] + P["Wpe"][:T][None, :, :]
        causal = np.tril(np.ones((T, T)))
        for i in range(self.cfg["n_layer"]):
            h = _ln_forward(x, P[f"b{i}.ln1.g"], P[f"b{i}.ln1.b"])
            q = _split_heads(h @ P[f"b{i}.Wq"], H)
            k = _split_heads(h @ P[f"b{i}.Wk"], H)
            v = _split_heads(h @ P[f"b{i}.Wv"], H)
            S = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(D // H)
            S = np.where(causal[None, :, :] > 0, S, -1e9)
            A = softmax(S, axis=-1)
            x = x + _merge_heads(A @ v) @ P[f"b{i}.Wo"]
            h2 = _ln_forward(x, P[f"b{i}.ln2.g"], P[f"b{i}.ln2.b"])
            ff = np.maximum(h2 @ P[f"b{i}.W1"] + P[f"b{i}.b1"], 0.0)
            x = x + ff @ P[f"b{i}.W2"] + P[f"b{i}.b2"]
        hf = _ln_forward(x, P["lnf.g"], P["lnf.b"])
        return hf @ P["Wlm"]

    @staticmethod
    def _is_subseq(needle, hay) -> bool:
        n = len(needle)
        if n == 0:
            return True
        return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))

    LEGAL_KEYS = ("thought", "action", "final", "tool", "args") + ARG_KEYS
    WORD_KEYS = ("query", "path", "content", "command")
    _SEP = set(" \t\n\"',;:")

    @staticmethod
    def _scan_string_state(s):
        i, last_open, in_str, esc = 0, None, False, False
        while i < len(s):
            c = s[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str, last_open = False, None
            elif c == '"':
                in_str, last_open = True, i
            i += 1
        if not in_str or last_open is None:
            return None, ""
        prefix = s[last_open + 1:]
        j = last_open - 1
        while j >= 0 and s[j] in " \t\n":
            j -= 1
        if j < 0:
            return "key", prefix
        return ("value", prefix) if s[j] == ":" else ("key", prefix)

    def _word_start_flags(self, src_ids):
        flags, prev_sep = [False] * len(src_ids), True
        for i, tid in enumerate(src_ids):
            c = self.itos[tid] if tid < len(self.itos) else ""
            flags[i] = bool(prev_sep and c and c not in self._SEP
                            and c.strip() and ord(c) < 128)
            prev_sep = (not c.strip()) or (c in self._SEP) or ord(c) >= 128
        return flags

    def _copy_mask(self, logits, gen_ids, src_ids):
        """约束解码三层：键名白名单 → 词首起点 → 完整词闭合。"""
        generated = "".join(self.itos[i] for i in gen_ids if i < len(self.itos))
        q = self.stoi.get('"')
        kind, prefix = self._scan_string_state(generated)

        if kind == "key":
            cands = [k for k in self.LEGAL_KEYS if k.startswith(prefix)]
            if not cands:
                allowed = {q} if q is not None else set()
            else:
                nxt = {c[len(prefix)] for c in cands if len(c) > len(prefix)}
                allowed = {self.stoi[c] for c in nxt if c in self.stoi}
                if prefix in self.LEGAL_KEYS and q is not None:
                    allowed.add(q)
            return self._apply(logits, allowed)

        if kind != "value":
            return logits

        matches = list(re.finditer(r'"(\w+)"\s*:\s*"', generated))
        if not matches or matches[-1].group(1) not in self.ARG_KEYS:
            return logits
        key = matches[-1].group(1)
        cut = matches[-1].end()
        if '"' in generated[cut:]:
            return logits
        pos, k = 0, len(gen_ids)
        for i, tid in enumerate(gen_ids):
            pos += len(self.itos[tid]) if tid < len(self.itos) else 0
            if pos >= cut:
                k = i + 1
                break
        value = gen_ids[k:]

        if key not in self.WORD_KEYS:
            src_t = tuple(src_ids)
            allowed = set(src_ids) if not value else {
                t for t in set(src_ids) if self._is_subseq(tuple(value) + (t,), src_t)}
            masked = self._apply(logits, allowed)
            if q is not None and value and masked.max() > -1e29:
                masked[q] = logits[q] - 4.0
            return masked

        chars = [self.itos[t] for t in src_ids if t < len(self.itos)]
        flags = self._word_start_flags(src_ids)
        if not value:
            return self._apply(logits, {t for t, f in zip(src_ids, flags) if f})

        base = tuple(value)
        nxt_chars, complete = set(), False
        for st in range(len(src_ids)):
            if not flags[st]:
                continue
            if tuple(src_ids[st:st + len(base)]) != base:
                continue
            j = st + len(base)
            if j < len(src_ids):
                nxt_chars.add(src_ids[j])
                if chars[j] in self._SEP or ord(chars[j]) >= 128:
                    complete = True
            else:
                complete = True
        allowed = set(nxt_chars)
        if complete and q is not None:
            allowed.add(q)
        return self._apply(logits, allowed)

    def _apply(self, logits, allowed):
        if not allowed:
            return logits
        masked = np.full_like(logits, -1e30)
        for t in allowed:
            if 0 <= t < masked.shape[0]:
                masked[t] = logits[t]
        return logits if masked.max() <= -1e29 else masked

    def infer(self, task: str, max_new: int = 0, constrained: bool = True) -> str:
        prompt = f"TASK:{task}|OUT:"
        ids = [self.stoi[BOS]] + self._enc(prompt)
        n_prompt = len(ids)
        src = self._enc(task)
        eos = self.stoi[EOS]
        max_new = max_new or min(160, self.cfg["max_len"])
        for _ in range(max_new):
            ctx = ids[-self.cfg["max_len"]:]
            logits = self.forward(np.array([ctx]))[0, -1]
            if constrained:
                logits = self._copy_mask(logits, ids[n_prompt:], src)
            nxt = int(np.argmax(logits))
            if nxt == eos:
                break
            ids.append(nxt)
        return "".join(self.itos[i] for i in ids[n_prompt:] if i < len(self.itos))

    def _enc(self, s: str):
        return [self.stoi[t] for t in tokenize(s) if t in self.stoi]


SAMPLES = [
    ("计算 (17 + 88) * 4 等于多少？", "calculator"),
    ("请算一下 (92 - 77) * 9", "calculator"),
    ("什么是LoRA？", "search"),
    ("搜索 Graphiti", "search"),
    ("把结果写入 report.md", "file_write"),
    ("生成 README.md 文件", "file_write"),
    ("运行一段 Python 看看结果", "python_exec"),
    ("执行代码 print(1+1)", "python_exec"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "mnemo_sonnet_evo5.npz"),
        help="evo.1~evo.7 任一 .npz 权重文件")
    ap.add_argument("--task", default=None)
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--no-constraint", action="store_true", help="关掉约束解码做对照")
    args = ap.parse_args()

    m, meta = MnemoModel.load(args.model)
    print(f"模型: {meta.get('model_family', 'Mnemo-Sonnet')} / {meta.get('version')}")
    print(f"架构: {meta.get('arch')}")
    print(f"参数: {meta.get('n_params'):,} | 词表: {meta.get('vocab_size')} | "
          f"训练步数: {meta.get('steps')}")
    print(f"训练数据: {meta.get('trained_on')}")
    print(f"生成于: {meta.get('created_at')}\n")

    if args.task:
        out = m.infer(args.task, constrained=not args.no_constraint)
        print(f"任务: {args.task}\n输出: {out}")
        try:
            print("解析: " + json.dumps(json.loads(out), ensure_ascii=False, indent=2))
        except Exception:
            print("（输出不是合法 JSON）")
        return

    print("=== 内置样例 ===")
    ok = 0
    for task, expect_tool in SAMPLES:
        out = m.infer(task, constrained=not args.no_constraint)
        tool = "-"
        try:
            tool = json.loads(out).get("action", {}).get("tool", "-")
            hit = tool == expect_tool
        except Exception:
            hit = False
        ok += int(hit)
        print(f"  {'✓' if hit else '✗'} {task}\n      → {out}")
    print(f"\n工具选择: {ok}/{len(SAMPLES)}")


if __name__ == "__main__":
    main()
