#!/usr/bin/env python3
"""Mnemo-Sonnet ONNX 版独立推理 —— 只需 numpy + onnxruntime，不依赖 sea / torch / transformers。

    pip install numpy onnxruntime
    python mnemo_onnx_load.py                                   # 内置样例
    python mnemo_onnx_load.py --task "计算 (1234 + 5678) * 2 等于多少？"

.onnx 是自包含模型文件：权重 + 计算图 + 词表 + 元数据全在一个文件里，
可被 onnxruntime / TensorRT / 浏览器 / 移动端直接加载。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("需要 onnxruntime：pip install onnxruntime")

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "mnemo_sonnet_evo5.onnx")

DEFAULT_TOKEN_RE = r"\s|[^\s]"


class MnemoONNX:
    ARG_KEYS = ("expression", "query", "path", "code", "content", "command")

    def __init__(self, path: str):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        doc = json.loads(self.sess.get_modelmeta().description)
        self.itos = doc["vocab"]
        self.meta = doc["meta"]
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.T = self.meta["fixed_seq_len"]
        pat = self.meta.get("tokenizer_pattern")
        if pat is None:
            print(f"⚠️ ONNX 文件未存 tokenizer_pattern，回退到 {DEFAULT_TOKEN_RE!r}。"
                  f"若输出异常，请确认该权重的训练分词方式。", file=sys.stderr)
        self.re = re.compile(pat or DEFAULT_TOKEN_RE, re.UNICODE)
        self.eos = self.stoi["<eos>"]
        self.pad = self.stoi["<pad>"]

    def enc(self, s: str):
        return [self.stoi[t] for t in self.re.findall(s) if t in self.stoi]

    @staticmethod
    def _is_subseq(needle: tuple, hay: tuple) -> bool:
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

    def _copy_mask(self, logits: np.ndarray, gen_ids, src_ids) -> np.ndarray:
        """约束解码三层：键名白名单 → 词首起点 → 完整词闭合。
        实测 evo5 真 OOD 参数 35% → 71%。"""
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

        ms = list(re.finditer(r'"(\w+)"\s*:\s*"', generated))
        if not ms or ms[-1].group(1) not in self.ARG_KEYS:
            return logits
        key = ms[-1].group(1)
        cut = ms[-1].end()
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
                t for t in set(src_ids)
                if self._is_subseq(tuple(value) + (t,), src_t)}
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

    def _apply(self, logits, allowed) -> np.ndarray:
        if not allowed:
            return logits
        masked = np.full_like(logits, -1e30)
        for t in allowed:
            if 0 <= t < masked.shape[0]:
                masked[t] = logits[t]
        return logits if masked.max() <= -1e29 else masked

    def _logits(self, ids) -> np.ndarray:
        ctx = (ids + [self.pad] * (self.T - len(ids))) if len(ids) < self.T else ids[-self.T:]
        arr = np.array([ctx], dtype=np.int64)
        out = self.sess.run(["logits"], {"input_ids": arr})[0]
        if out.ndim == 4:
            out = out.reshape(out.shape[1:])
        pos = len(ids) - 1 if len(ids) <= self.T else self.T - 1
        return out[0, pos]

    def infer(self, task: str, max_new: int = 150, constrained: bool = True) -> str:
        prompt = f"TASK:{task}|OUT:"
        ids = [self.stoi["<bos>"]] + self.enc(prompt)
        n_prompt = len(ids)
        src = self.enc(task)
        for _ in range(max_new):
            lg = self._logits(ids)
            if constrained:
                lg = self._copy_mask(lg, ids[n_prompt:], src)
            nxt = int(np.argmax(lg))
            if nxt == self.eos:
                break
            ids.append(nxt)
        return "".join(self.itos[i] for i in ids[n_prompt:] if i < len(self.itos))


SAMPLES = [
    ("计算 (17 + 88) * 4 等于多少？", "calculator"),
    ("算一算 (25000 - 12345) * 4", "calculator"),
    ("请帮我求出 (45 - 12) * 3 的数值", "calculator"),
    ("什么是LoRA？", "search"),
    ("搜索 Graphiti", "search"),
    ("把结果写入 report.md", "file_write"),
    ("创建文件 config.yaml", "file_write"),
    ("执行 print(1+1)", "python_exec"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--task", default=None)
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--no-constraint", action="store_true")
    args = ap.parse_args()

    m = MnemoONNX(args.model)
    mt = m.meta
    print(f"模型: {mt.get('model_family')} / {mt.get('version')}  [ONNX]")
    print(f"架构: {mt.get('arch')} | 参数: {mt.get('n_params'):,} | "
          f"序列长度: {m.T} | 词表: {len(m.itos)}")
    print(f"训练数据: {mt.get('trained_on')}\n")

    if args.task:
        out = m.infer(args.task, constrained=not args.no_constraint)
        print(f"任务: {args.task}\n输出: {out}")
        try:
            print("解析: " + json.dumps(json.loads(out), ensure_ascii=False, indent=2))
        except Exception:
            print("（输出不是合法 JSON）")
        return

    ok = 0
    for task, expect in SAMPLES:
        out = m.infer(task, constrained=not args.no_constraint)
        try:
            tool = json.loads(out).get("action", {}).get("tool", "-")
            hit = tool == expect
        except Exception:
            hit = False
        ok += int(hit)
        print(f"  {'✓' if hit else '✗'} {task}\n      → {out[:100]}")
    print(f"\n工具选择: {ok}/{len(SAMPLES)}")


if __name__ == "__main__":
    main()
