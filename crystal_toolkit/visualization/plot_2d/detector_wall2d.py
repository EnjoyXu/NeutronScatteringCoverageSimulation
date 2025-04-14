from numpy import linspace, meshgrid, array, append
from crystal_toolkit.detector.detector_config import DetectorConfig
from crystal_toolkit.visualization.plotter_base import BasePlotter
import plotly.graph_objs as go


class DetectorWall2dPlotter(BasePlotter):

    def __init__(self, detector_config: DetectorConfig, detector_step_deg=1):
        super().__init__()

        self.detector_config = detector_config

        self.step_deg = detector_step_deg

    def plot(
        self,
    ):
        """画出二维探测器墙"""

        phi_tot = array([])
        theta_tot = array([])

        for (phi_min, phi_max), (theta_min, theta_max) in zip(
            self.detector_config.phi_ranges, self.detector_config.theta_ranges_direct
        ):
            # 生成角度网格
            phi = linspace(
                phi_min, phi_max, (phi_max - phi_min) // self.step_deg, endpoint=True
            )

            theta = linspace(
                theta_min,
                theta_max,
                (theta_max - theta_min) // self.step_deg,
                endpoint=True,
            )

            # 创建网格并向量化计算
            theta_grid, phi_grid = meshgrid(theta, phi)

            theta_tot = append(theta_tot, theta_grid.ravel())

            phi_tot = append(phi_tot, phi_grid.ravel())

        self.add_trace(
            go.Scatter(
                x=phi_tot,
                y=theta_tot,
                opacity=self.config.opacity["detector_2d"],
                mode="markers",
                marker=dict(
                    size=self.config.sizes["detector"],
                    color=self.config.colors["detector"],
                ),
                name="detectors",
            )
        )
        self._apply_layout("Detector Wall 2d")
        return self.fig
