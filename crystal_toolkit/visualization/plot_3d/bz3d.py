from crystal_toolkit.lattice.brillouin_zone import get_wigner_size_cell_3d
from crystal_toolkit.math_utils.math_utils import get_points_labels
from crystal_toolkit.visualization.plotter_base import BasePlotter
import plotly.graph_objs as go
from numpy import ndarray


class BrillouinZone3DPlotter(BasePlotter):
    def __init__(
        self, reciprocal_lattice_matrix: ndarray, conv_reciprocal_lattice_matrix
    ):
        super().__init__()
        self.reciprocal = reciprocal_lattice_matrix
        self.conv_reciprocal = conv_reciprocal_lattice_matrix

        self.vertices, self.edges, self.facets = get_wigner_size_cell_3d(
            *self.reciprocal
        )

        self.edge_labels = [
            get_points_labels(edge, new_basis=self.conv_reciprocal, decimal_places=3)
            for edge in self.edges
        ]

    def plot(self) -> go.Figure:
        """绘制布里渊区"""

        self._plot_edges()
        self._plot_facets()
        # self._plot_symmetry_points()
        self._apply_layout("Brillouin Zone 3D")
        return self.fig

    def _plot_edges(
        self,
    ) -> None:
        """绘制所有边界线"""
        start = 0
        for edge, label in zip(self.edges, self.edge_labels):

            x = edge[:, 0]
            y = edge[:, 1]
            z = edge[:, 2]

            is_show_legend = True if start == 0 else False

            self.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    line=dict(
                        color=self.config.colors["BZ"],
                        width=self.config.widths["BZ_edge"],
                    ),
                    text=label,
                    name="BZ",
                    legendgroup="BZ",
                    showlegend=is_show_legend,
                )
            )

            start += 1

    def _plot_facets(
        self,
    ) -> None:
        """绘制所有边界线"""
        for facet in self.facets:
            x = facet[:, 0]
            y = facet[:, 1]
            z = facet[:, 2]
            self.add_trace(
                go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    color=self.config.colors["BZ"],
                    name="BZ Facet",
                    legendgroup="BZ",
                    alphahull=-1,
                )
            )

    def _plot_symmetry_points(self) -> None:
        """标注高对称点"""
        points = self.reciprocal.get_symmetry_points()
        for label, coord in points.items():
            self.add_trace(
                go.Scatter3d(
                    x=[coord],
                    y=[coord],
                    z=[coord],
                    mode="markers+text",
                    marker=dict(size=8, color="#FF1493"),
                    text=label,
                    textposition="top center",
                    name=f"Symmetry Point {label}",
                )
            )
