from crystal_toolkit.detector.detector import Detector
from numpy import ndarray
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

        return [
            self._get_plane_slice(detector_data)[0]
            for detector_data in self.detector.detector_points_list
        ]

    def plot(self) -> go.Figure:
        """绘制探测器空间覆盖区域"""

        self.add_traces(
            [
                go.Scatter(
                    y=detector_data[:, 1],
                    x=detector_data[:, 0],
                    opacity=self.config.opacity["detector_2d"],
                    mode="markers",
                    marker=dict(
                        size=self.config.sizes["detector"],
                        color=self.config.colors["detector"],
                    ),
                    name="detectors",
                )
                for detector_data in self.detector_points_list
            ]
        )

        self._apply_layout("Detector")

        return self.fig
