from abc import abstractmethod
from traitlets import Bool
from crystal_toolkit.math_utils.math_utils import coordinate_transform
from crystal_toolkit.visualization.plotter_base import BasePlotter
import plotly.graph_objs as go
from numpy import ndarray, min, max


class CompositePlotter(BasePlotter):
    def __init__(self):
        super().__init__()
        self.fig_list = []  # 存储注册的绘图组件
        self.fig = go.Figure()
        self.slider = {}
        self.title = ""

    @abstractmethod
    def _get_title(self) -> str:
        pass

    def add_slider_figure(self, slider_fig: go.Figure, slider_title, slider_label):
        """注册探测器组件"""
        self.slider["figure"] = slider_fig
        self.slider["title"] = slider_title
        self.slider["label"] = slider_label

    def add_figure(self, fig: go.Figure):
        """注册可视化组件"""
        self.fig_list.append(fig)

    def build_plot(self, is_plot_slider=False):
        """组装figures"""
        for figure in self.fig_list:
            self.fig = CompositePlotter.merge(self.fig, figure)

        if self.slider and is_plot_slider:
            self._build_slider_plot()

    def _build_slider_plot(self):
        """组装slider"""
        other_trace_length = len(self.fig.data)
        slider_length = len(self.slider["figure"].data)

        # 第0帧:将非弹slider图像都不可见
        for i in range(1, slider_length):
            self.slider["figure"].data[i].visible = False

        self.fig = CompositePlotter.merge(self.fig, self.slider["figure"])

        steps = []

        for i, title, label in zip(
            range(slider_length), self.slider["title"], self.slider["label"]
        ):
            visible_states = [False] * slider_length
            visible_states[i] = True  # 仅当前轨迹可见

            steps.append(
                dict(
                    method="update",
                    args=[
                        # 设置可见性：保留底图，更新覆盖层
                        {"visible": [True] * other_trace_length + visible_states},
                        {
                            "title": {
                                "text": self.title + title,
                                # ,  # 自定义样式
                            }
                        },
                    ],
                    label=label,
                )
            )

        sliders = [
            dict(
                active=0,
                steps=steps,
                # currentvalue={"prefix": f"dE:"}
            )
        ]

        self.fig.update_layout(sliders=sliders)

    def _apply_unified_layout(
        self,
    ):
        """统一坐标系统和视觉样式(兼容2D/3D)"""
        is_3d = any(
            trace.type in ["scatter3d", "surface", "mesh3d"] for trace in self.fig.data
        )

        layout_args = {
            "title": self.title,
            "xaxis": dict(range=self._calc_axis_range("x"), autorange=False),
            "yaxis": dict(range=self._calc_axis_range("y"), autorange=False),
        }

        if is_3d:
            # 3D图形配置
            layout_args["scene"] = dict(
                xaxis=layout_args.pop("xaxis"),
                yaxis=layout_args.pop("yaxis"),
                zaxis=dict(range=self._calc_axis_range("z"), autorange=False),
                aspectmode="data",
            )
        else:
            # 2D图形配置
            layout_args["xaxis"]["scaleanchor"] = "y"  # 保持宽高比一致
            layout_args["xaxis"]["constrain"] = "domain"

        self.fig.update_layout(**layout_args)

    def _calc_axis_range(self, axis: str) -> list:
        """自动计算所有Trace的坐标范围（兼容空数组）"""
        ranges = []
        for trace in self.fig.data:
            if hasattr(trace, axis):
                data = getattr(trace, axis)
                # 确保是NumPy数组且非空
                if isinstance(data, ndarray) and data.size > 0:
                    ranges.extend([min(data), max(data)])
        return [min(ranges), max(ranges)] if ranges else [-10, 10]

    @staticmethod
    def merge(base_fig: go.Figure, new_fig: go.Figure) -> go.Figure:
        """merge两个figure"""
        merged_fig = go.Figure()

        # 合并数据Trace
        merged_fig.add_traces(list(base_fig.data) + list(new_fig.data))

        # 融合布局设置
        merged_fig.layout.update(
            {**base_fig.layout.to_plotly_json(), **new_fig.layout.to_plotly_json()}
        )

        return merged_fig
