from crystal_toolkit.detector.detector import Detector
from numpy import ndarray
import numpy as np
import plotly.graph_objs as go

from crystal_toolkit.visualization.plot_2d.plotter_2d_base import (
    BasePlotter2D,
)


class Detector2DPlotter(BasePlotter2D):
    def __init__(
        self,
        detector: Detector,
        norm_vector: ndarray,
        plane_point: ndarray,
        thickness: float,
        parallel_new_ex: ndarray,
    ):
        super().__init__(norm_vector, plane_point, thickness, parallel_new_ex)
        self.detector = detector

        self.detector_points_list = self._get_detector_data()

    def _get_detector_data(self):
        result = []
        for points, psi_values in self.detector.detector_points_list:
            # 将psi_values作为标签传递给平面切片，自动同步过滤
            projected, _, filtered_psi = self._get_plane_slice(points, psi_values)
            result.append((projected, filtered_psi))
        return result

    def plot(self) -> go.Figure:
        """绘制探测器空间覆盖区域"""

        traces = []
        for i, (projected, psi_values) in enumerate(self.detector_points_list):
            traces.append(
                go.Scatter(
                    y=projected[:, 1],
                    x=projected[:, 0],
                    opacity=self.config.opacity["detector_2d"],
                    mode="markers",
                    marker=dict(
                        size=self.config.sizes["detector"],
                        color=-np.asarray(psi_values),
                        colorscale=[
                            [0, "rgb(175, 212, 247)"],
                            [0.5, "rgb(100, 168, 228)"],
                            [1, "rgb(10, 95, 190)"],
                        ],
                        showscale=(i == 0),
                        colorbar=dict(
                            title="-Psi (°)",
                            x=0,
                            xanchor="right",
                        ),
                    ),
                    name="detectors",
                )
            )

        self.add_traces(traces)

        self._apply_layout("Detector")

        # 移除 scaleanchor 等比锁定，让 Plot 可以随容器自适应铺满
        self.fig.update_xaxes(scaleanchor=None)
        self.fig.update_yaxes(scaleanchor=None)

        return self.fig
