from crystal_toolkit.detector.detector import Detector
from crystal_toolkit.visualization.plotter_base import BasePlotter
import plotly.graph_objs as go


class Detector3DPlotter(BasePlotter):
    def __init__(
        self,
        detector: Detector,
    ):
        super().__init__()
        self.detector = detector

    def plot(self) -> go.Figure:
        """绘制探测器空间覆盖区域"""

        self.add_traces(
            [
                go.Scatter3d(
                    z=detector_data[:, 2],
                    x=detector_data[:, 0],
                    y=detector_data[:, 1],
                    opacity=0.1,
                    mode="markers",
                    marker=dict(
                        size=self.config.sizes["detector"],
                        color=self.config.colors["detector"],
                    ),
                    name="detectors",
                )
                for detector_data in self.detector.detector_points_list
            ]
        )

        return self.fig

    # def _plot_detector_surface(self) -> None:
    #     """绘制探测器表面（使用Marching Cubes算法生成等值面）"""
    #     vertices, faces = self.detector.generate_surface_mesh()
    #     self.add_trace(
    #         go.Mesh3d(
    #             x=vertices[:, 0],
    #             y=vertices[:, 1],
    #             z=vertices[:, 2],
    #             i=faces[:, 0],
    #             j=faces[:, 1],
    #             k=faces[:, 2],
    #             color=self.config.colors["detector"],
    #             opacity=0.3,
    #             name="Detector Surface",
    #         )
    #     )

    # def _plot_beam_direction(self) -> None:
    #     """绘制入射光束方向指示器"""
    #     beam_vec = self.detector.beam_direction * self.detector.max_range
    #     self.add_trace(
    #         go.Scatter3d(
    #             x=[0, beam_vec],
    #             y=[0, beam_vec],
    #             z=[0, beam_vec],
    #             mode="lines",
    #             line=dict(color="#FFA500", width=10, dash="dot"),
    #             name="Beam Direction",
    #         )
    #     )
