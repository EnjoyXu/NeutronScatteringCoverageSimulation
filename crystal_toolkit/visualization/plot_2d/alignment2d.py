import array
from matplotlib.pyplot import step
from crystal_toolkit.detector.detector import Detector
from numpy import linspace, ndarray, meshgrid, rad2deg
import plotly.graph_objs as go

from crystal_toolkit.lattice.lattice import Lattice
from crystal_toolkit.math_utils.math_utils import get_points_labels, wrap_to_interval
from crystal_toolkit.visualization.composite.composite_base import CompositePlotter

from numpy import deg2rad, arange, pi, array, linspace, mod, where


class Alignment2DPlotter(CompositePlotter):
    def __init__(self, detector: Detector, lattice: Lattice, detector_step_deg=1):
        super().__init__()
        self.detector = detector
        self.lattice = lattice
        self.step_deg = detector_step_deg
        self.title = self._get_title()

    def _get_title(self):
        return "Alignment 2D"

    def plot(
        self,
    ):
        self._plot_detector_wall()

        self._plot_k_points()

        self._apply_layout(self.title)

        return self.fig

    def _plot_detector_wall(
        self,
    ):
        for (phi_min, phi_max), (theta_min, theta_max) in zip(
            self.detector.config.phi_ranges, self.detector.config.theta_ranges_direct
        ):
            # 生成角度网格
            phi = linspace(
                phi_min, phi_max, (phi_max - phi_min) // self.step_deg, endpoint=True
            )
            # phi = arange(phi_min, phi_max + 1, self.step_deg)

            theta = linspace(
                theta_min,
                theta_max,
                (theta_max - theta_min) // self.step_deg,
                endpoint=True,
            )
            # theta = arange(theta_min, theta_max + 1, self.step_deg)
            # 创建网格并向量化计算
            theta_grid, phi_grid = meshgrid(theta, phi)

            self.add_trace(
                go.Scatter(
                    x=phi_grid.ravel(),
                    y=theta_grid.ravel(),
                    opacity=self.config.opacity["detector_2d"],
                    mode="markers",
                    marker=dict(
                        size=self.config.sizes["detector"],
                        color=self.config.colors["detector"],
                    ),
                    name="detectors",
                )
            )

    def _plot_k_points(
        self,
    ):
        points_arr, points = self.detector.get_available_points_coordinates_white_beam(
            self.lattice.pri_k_points,
            0.98,
            21.9778 * 1.4,
        )

        label = get_points_labels(
            points, self.lattice.lattice_data.conv_reciprocal_matrix
        )

        phi = wrap_to_interval(rad2deg(points_arr[:, 2]), -180, 180)
        theta = 90 - wrap_to_interval(rad2deg(points_arr[:, 1]), 0, 180)

        # print(points_arr)
        self.add_trace(
            go.Scatter(
                x=phi,
                y=theta,
                mode="markers",
                hovertext=label,
                marker=dict(
                    # size=self.config.sizes["atom"],
                    color="red",
                    # line=dict(width=self.config.widths["atom_marker"], color=color),
                ),
            )
        )
