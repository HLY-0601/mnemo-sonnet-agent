"""模型层工厂：按配置字符串创建后端。"""
import os
from typing import Optional
from .api import APIBackend
from .base import (BASE_VARIANT, MOCK_VARIANT, MODEL_FAMILY, MODEL_TAGLINE,
                   GenerationConfig, Message, ModelBackend, extract_json)
from .local import HFBackend
from .mock import MockBackend
from .numpy_backend import NumpyBackend

__all__ = ["ModelBackend", "HFBackend", "APIBackend", "MockBackend", "NumpyBackend",
           "build_backend", "GenerationConfig", "Message", "extract_json",
           "MODEL_FAMILY", "MODEL_TAGLINE", "BASE_VARIANT", "MOCK_VARIANT"]

def build_backend(kind: str = "auto", model_name_or_path: Optional[str] = None,
                  adapter_path: Optional[str] = None, **kwargs) -> ModelBackend:
    kind = (kind or "auto").lower()
    if kind == "mock":
        return MockBackend()
    if kind == "numpy":
        if not model_name_or_path:
            raise ValueError("numpy 后端需要 model_name_or_path")
        return NumpyBackend(model_name_or_path, **kwargs)
    if kind == "api":
        return APIBackend(model=model_name_or_path or os.getenv("SEA_MODEL", "deepseek-chat"), **kwargs)
    if kind == "hf":
        if not model_name_or_path:
            raise ValueError("hf 后端需要 model_name_or_path")
        return HFBackend(model_name_or_path, adapter_path=adapter_path, **kwargs)
    if model_name_or_path and os.path.exists(os.path.expanduser(model_name_or_path)):
        try:
            return HFBackend(model_name_or_path, adapter_path=adapter_path, **kwargs)
        except Exception as e:
            print(f"[model] 本地模型加载失败({e})，尝试降级")
    if os.getenv("SEA_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return APIBackend(model=model_name_or_path or "deepseek-chat", **kwargs)
    return MockBackend()
