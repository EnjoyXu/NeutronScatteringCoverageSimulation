from numpy import ndarray, vstack

from crystal_toolkit.lattice.brillouin_zone import get_wigner_size_cell_2d
from crystal_toolkit.math_utils.geometry import (
    get_convex_vertice_points_2d,
)
from crystal_toolkit.math_utils.math_utils import get_points_labels
from crystal_toolkit.visualization.plot_2d.plotter_2d_base import BasePlotter2D
import plotly.graph_objs as go


class BrillouinZone2DPlotter(BasePlotter2D):
    def __init__(
        self,
        reciprocal_lattice_matrix: ndarray,
        conv_reciprocal_lattice_matrix: ndarray,
        norm_vector: ndarray,
        plane_point: ndarray,
        thickness: float,
        parallel_new_ex: ndarray,
    ):
        super().__init__(norm_vector, plane_point, thickness, parallel_new_ex)

        self.reciprocal = reciprocal_lattice_matrix
        self.conv_reciprocal = conv_reciprocal_lattice_matrix

        self.edge_2d_points, self.edge_labels = self._get_edge_points()

    def _get_edge_points(
        self,
    ):
        """得到用于绘图的封闭边坐标和label"""

        # 得到筛选过的2d点
        edge_2d_points, edge_points = get_wigner_size_cell_2d(
            self.reciprocal[0],
            self.reciprocal[1],
            self.reciprocal[2],
            self.plane_point,
            self.norm_vector,
            self.thickness,
            self.parallel_new_ex,
        )

        # 根据三维坐标得到label
        edge_labels = get_points_labels(edge_points, new_basis=self.conv_reciprocal)

        # 取凸边界点
        edge_2d_points, edge_labels = get_convex_vertice_points_2d(
            edge_2d_points, edge_labels
        )

        # 顶点封闭
        edge_2d_points = vstack((edge_2d_points, edge_2d_points[0]))

        edge_labels = tuple(list(edge_labels) + [edge_labels[0]])

        return (edge_2d_points, edge_labels)

    def plot(self):
        self.add_trace(
            go.Scatter(
                x=self.edge_2d_points[:, 0],
                y=self.edge_2d_points[:, 1],
                text=self.edge_labels,
                hovertext=self.edge_labels,
                mode="markers",
                marker=dict(
                    size=0.1,
                ),
                fill="toself",
                line=dict(
                    color=self.config.colors["BZ"],
                ),
                name="BZ",
            )
        )

        self._apply_layout("1BZ")
        return self.fig
