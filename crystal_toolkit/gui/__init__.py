"""
CrystalToolkit GUI — 快捷入口。

用法:
    python -m crystal_toolkit.gui
等价于:
    python -m crystal_toolkit.visualization.gui
"""
from crystal_toolkit.visualization.gui.app import main
__all__ = ["main"]
