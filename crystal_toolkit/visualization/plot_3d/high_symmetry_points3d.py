from crystal_toolkit.lattice.high_symm_points import get_high_symmetry_path_list
from crystal_toolkit.visualization.plotter_base import BasePlotter
import plotly.graph_objs as go


class HighSymmetryPoints3DPlotter(BasePlotter):
    def __init__(self, hsp_points, hsp_label):
        super().__init__()
        self.points_list, self.label_list = get_high_symmetry_path_list(
            hsp_points, hsp_label
        )

    def plot(
        self,
    ):
        self.add_traces(
            [
                go.Scatter3d(
                    x=point[:, 0],
                    y=point[:, 1],
                    z=point[:, 2],
                    hovertext=label,
                    # text=label,
                    mode="markers+lines",
                    marker=dict(
                        size=self.config.sizes["high_symmetry_points"],
                        color=self.config.colors["high_symmetry_points"],
                        line=dict(
                            width=self.config.widths["high_symmetry_points"],
                            color=self.config.colors["high_symmetry_points"],
                        ),
                    ),
                    # textfont=dict(size=20, color=color),
                    name="HSP",
                    # line=dict(dash="dash"),
                )
                for point, label in zip(self.points_list, self.label_list)
            ]
        )

        self._apply_layout("High symmetry points")

        return self.fig
