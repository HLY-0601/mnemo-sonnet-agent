"""工作区隔离。"""
from __future__ import annotations
import os, shutil

class Workspace:
    def __init__(self, root: str = "data/workspaces", clean: bool = False):
        self.root = root
        if clean and os.path.exists(root):
            shutil.rmtree(root)
        os.makedirs(root, exist_ok=True)
    def path(self, name: str) -> str:
        p = os.path.join(self.root, name)
        os.makedirs(p, exist_ok=True)
        return p
