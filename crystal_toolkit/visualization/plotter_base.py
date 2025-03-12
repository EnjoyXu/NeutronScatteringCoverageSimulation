from abc import ABC, abstractmethod
import plotly.graph_objs as go

from crystal_toolkit.visualization.visutal_config import VisualConfig


class BasePlotter(ABC):
    def __init__(self, config=None):
        self.config = config or VisualConfig()
        self.fig = go.Figure()

    @abstractmethod
    def plot(self, data: dict) -> go.Figure:
        """主绘图接口（需子类实现）"""
        pass

    def add_trace(self, trace: go.Figure) -> None:
        """添加单个Trace到画布"""
        self.fig.add_trace(trace)

    def add_traces(
        self,
        data,
    ) -> None:
        self.fig.add_traces(data)

    def _apply_layout(self, title: str) -> None:
        """应用统一布局模板"""
        self.fig.update_layout(
            title=title,
            scene=dict(
                # xaxis_title="X (Å)",
                # yaxis_title="Y (Å)",
                # zaxis_title="Z (Å)",
                aspectmode="data",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            xaxis=dict(
                scaleanchor="y",  # 锁定 x 轴比例以匹配 y 轴
                scaleratio=1,  # x 轴和 y 轴比例 1:1
            ),
            yaxis=dict(
                scaleanchor="x",  # 锁定 y 轴比例以匹配 x 轴
                scaleratio=1,  # y 轴和 x 轴比例 1:1
            ),
        )

    def save_html(self, filename: str) -> None:
        """保存为交互式HTML文件"""
        self.fig.write_html(filename)
