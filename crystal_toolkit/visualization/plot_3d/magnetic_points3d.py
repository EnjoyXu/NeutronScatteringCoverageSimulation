from crystal_toolkit.lattice.brillouin_zone import get_wigner_size_cell_3d
from crystal_toolkit.math_utils.math_utils import get_points_labels
from crystal_toolkit.visualization.plotter_base import BasePlotter
import plotly.graph_objs as go
from numpy import ndarray


class MagneticPoints3DPlotter(BasePlotter):
    def __init__(self, magn_points: ndarray, magn_label):
        super().__init__()
        self.magn_points = magn_points
        self.magn_label = magn_label

    def plot(self) -> go.Figure:
        """绘制磁峰"""

        self.add_trace(
            go.Scatter3d(
                x=self.magn_points[:, 0],
                y=self.magn_points[:, 1],
                z=self.magn_points[:, 2],
                hovertext=self.magn_label,
                mode="markers",
                marker=dict(
                    size=self.config.sizes["magnetic_points"],
                    color=self.config.colors["magnetic_points"],
                    line=dict(
                        width=self.config.widths["magnetic_points"],
                        color=self.config.colors["magnetic_points"],
                    ),
                ),
                name="magnetic",
            )
        )
        # self._plot_symmetry_points()
        self._apply_layout("Magnetic points 3D")
        return self.fig
