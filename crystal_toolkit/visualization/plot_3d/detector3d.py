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

        traces = []
        for i, (points, psi_values) in enumerate(self.detector.detector_points_list):
            traces.append(
                go.Scatter3d(
                    z=points[:, 2],
                    x=points[:, 0],
                    y=points[:, 1],
                    opacity=0.1,
                    mode="markers",
                    marker=dict(
                        size=self.config.sizes["detector"],
                        color=-psi_values,
                        colorscale=[
                            [0, "rgb(175, 212, 247)"],
                            [0.5, "rgb(120, 185, 235)"],
                            [1, "rgb(50, 150, 220)"],
                        ],
                        showscale=(i == 0),
                        colorbar=dict(
                            title="Psi (°)",
                            x=0,
                            xanchor="right",
                        ),
                    ),
                    name="detectors",
                )
            )

        self.add_traces(traces)

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
