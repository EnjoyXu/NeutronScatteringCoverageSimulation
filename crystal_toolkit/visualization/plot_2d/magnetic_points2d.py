from crystal_toolkit.visualization.plot_2d.plotter_2d_base import BasePlotter2D

import plotly.graph_objs as go
from numpy import ndarray


class MagneticPoints2DPlotter(BasePlotter2D):
    def __init__(
        self,
        magn_3d_points: ndarray,
        magn_label,
        norm_vector: ndarray,
        plane_point: ndarray,
        thickness: float,
        parallel_new_ex: ndarray,
    ):
        super().__init__(norm_vector, plane_point, thickness, parallel_new_ex)

        self.magn_points, _, self.magn_label = self._get_plane_slice(
            magn_3d_points, magn_label
        )

    def plot(self) -> go.Figure:
        """绘制磁峰"""

        self.add_trace(
            go.Scatter(
                x=self.magn_points[:, 0],
                y=self.magn_points[:, 1],
                hovertext=self.magn_label,
                mode="markers",
                marker=dict(
                    size=self.config.sizes["magnetic_points_2d"],
                    color=self.config.colors["magnetic_points_2d"],
                    line=dict(
                        width=self.config.widths["magnetic_points"],
                        color=self.config.colors["magnetic_points_2d"],
                    ),
                ),
                name="magnetic",
            )
        )
        # self._plot_symmetry_points()
        self._apply_layout("Magnetic points 2D")
        return self.fig
