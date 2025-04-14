from matplotlib.pyplot import step
from crystal_toolkit.detector.detector import Detector
from numpy import ndarray, meshgrid
import plotly.graph_objs as go

from crystal_toolkit.lattice.lattice import Lattice
from crystal_toolkit.math_utils.math_utils import (
    coordinate_transform,
    get_points_labels,
)
from crystal_toolkit.visualization.composite.composite_base import CompositePlotter

from numpy import deg2rad, arange, pi

import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go

from crystal_toolkit.visualization.plot_2d.detector_wall2d import DetectorWall2dPlotter
from crystal_toolkit.math_utils.math_utils import rotation_matrix


class Alignment2DPlotter(CompositePlotter):
    def __init__(self, detector: Detector, lattice: Lattice, step_deg=1):
        super().__init__()
        self.detector = detector
        self.lattice = lattice
        self.step_deg = step_deg
        self.title = self._get_title()

    def _get_title(self):
        return "Alignment 2D"

    def plot(
        self,
    ):
        self.add_figure(
            DetectorWall2dPlotter(self.detector.config, self.step_deg).plot()
        )

        data = self._get_paint_data()

        M = len(data)

        app = dash.Dash(__name__)

        app.layout = html.Div(
            [
                html.Div(
                    [
                        dcc.Input(
                            id="min-input", type="number", value=np.min(data[0]["d"])
                        ),
                        dcc.Input(
                            id="max-input", type="number", value=np.max(data[0]["d"])
                        ),
                    ],
                    style={"margin": "20px"},
                ),
                dcc.Slider(
                    id="array-slider",
                    min=0,
                    max=M,
                    step=1,
                    value=0,
                    marks={i: str(i) for i in range(M)},
                    tooltip={"placement": "bottom"},
                ),
                dcc.Graph(id="3d-plot"),
            ]
        )

        @callback(
            Output("3d-plot", "figure"),
            [
                Input("min-input", "value"),
                Input("max-input", "value"),
                Input("array-slider", "value"),
            ],
        )
        def update_plot(min_val, max_val, slider_val):
            # 筛选当前数组的数据
            current_data = data[slider_val]
            current_d = current_data["d"]

            # 创建背景轨迹（所有数据）
            background_trace = self.fig_list[0].data

            # 创建主轨迹（当前数组）
            mask = (current_d >= min_val) & (current_d <= max_val)
            main_trace = go.Scatter(
                x=current_data["phi"][mask],
                y=current_data["theta"][mask],
                mode="markers",
                # marker=dict(
                #     size=5, color=current_d[mask], colorscale="Viridis", opacity=0.8
                # ),
                name=f"Array {slider_val}",
            )

            return {
                "data": background_trace + background_trace,
                # "layout": go.Layout(
                #     scene=dict(
                #         xaxis_title="phi",
                #         yaxis_title="theta",
                #     ),
                #     # margin=dict(l=0, r=0, b=0, t=30),
                #     # height=800,
                # ),
            }

        # print(points_arr)
        # self.add_trace(
        #     go.Scatter(
        #         x=points_arr[:, 2],
        #         y=points_arr[:, 1],
        #         mode="markers",
        #         hovertext=label,
        #     )
        # )

        # self._apply_layout(self.title)

        return app

    def _get_paint_data(
        self,
    ):
        data = []

        u = self.detector.config.detector_u
        v = self.detector.config.detector_v

        psi_range = sorted([-psi for psi in self.detector.config.psi_range])
        psi_grid = np.rad2deg(np.arange(psi_range[0], psi_range[1], self.step_deg))
        for psi in psi_grid:
            u_new, v_new = coordinate_transform(
                np.vstack((u, v)),
                rotation_matrix(
                    self.detector.config.detector_w,
                    psi,
                ),
                is_positive=True,
            )

            points_arr, points = (
                self.detector.get_available_points_coordinates_white_beam(
                    self.lattice.pri_k_points,
                    self.lattice.lattice_data.a_star_par / 10,
                    self.lattice.lattice_data.a_star_par * 20,
                    u_new,
                    v_new,
                )
            )

            d = 2 * np.pi / np.linalg.norm(points, axis=1)

            label = get_points_labels(
                points, self.lattice.lattice_data.conv_reciprocal_matrix
            )

            points_arr[:, 1] = pi / 2 - points_arr[:, 1]

            data.append(
                {
                    "phi": np.rad2deg(points_arr[:, 2]),
                    "theta": np.rad2deg(points_arr[:, 1]),
                    "label": label,
                    "d": d,
                }
            )
        return data
