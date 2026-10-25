"""公共工具：受限执行沙箱、文件 IO、时间/文本助手。"""
from __future__ import annotations
import ast, json, math, os, time
from typing import Any, Dict, Optional, Tuple

_ALLOWED_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len, "round": round,
    "sorted": sorted, "range": range, "enumerate": enumerate, "zip": zip,
    "int": int, "float": float, "str": str, "bool": bool, "list": list,
    "dict": dict, "set": set, "tuple": tuple, "print": print, "isinstance": isinstance,
}

def safe_eval(expression: str) -> Any:
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Call, ast.Attribute, ast.Subscript, ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Call):
                raise ValueError("表达式中不允许函数调用")
            raise ValueError(f"不允许的语法: {type(node).__name__}")
    return eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})

def safe_exec(code: str, timeout: float = 5.0, extra_globals: Optional[Dict[str, Any]] = None) -> Tuple[Any, str]:
    import io
    from contextlib import redirect_stdout
    g: Dict[str, Any] = {"__builtins__": dict(_ALLOWED_BUILTINS)}
    if extra_globals:
        g.update(extra_globals)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(code, g)
        return g, buf.getvalue()
    except Exception as e:
        return g, f"[执行错误] {type(e).__name__}: {e}"

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)

def read_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    return out

def append_jsonl(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")

def recency_score(ts: float, half_life_hours: float = 72.0) -> float:
    age_h = max(0.0, (time.time() - ts) / 3600.0)
    return math.pow(0.5, age_h / max(1e-6, half_life_hours))

def truncate(text: str, n: int = 400) -> str:
    text = str(text)
    return text if len(text) <= n else text[:n] + "…"
