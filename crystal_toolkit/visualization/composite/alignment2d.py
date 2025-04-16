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

        background_trace = list(self.fig_list[0].data)

        data = self._get_paint_data()
        combined_d = np.concatenate([item["d"] for item in data])
        M = len(data)

        app = dash.Dash(__name__)

        app.layout = html.Div(
            [
                # 控制栏（输入框）
                html.Div(
                    [
                        html.Span("d min:", style={"marginRight": "10px"}),
                        dcc.Input(
                            id="min-input",
                            type="number",
                            value=np.min(combined_d),
                            style={"marginRight": "10px"},  # 输入框间距
                        ),
                        html.Span("d max:", style={"marginRight": "10px"}),
                        dcc.Input(
                            id="max-input", type="number", value=np.max(combined_d)
                        ),
                    ],
                    style={"margin": "20px"},
                ),
                # 滑块
                dcc.Slider(
                    id="array-slider",
                    min=0,
                    max=M,
                    step=1,
                    value=0,
                    marks={i: f'{data[i]["psi"]:.1f}' for i in range(M)},
                    tooltip={"placement": "bottom"},
                ),
                # 图表区域（最大化）
                dcc.Graph(
                    id="3d-plot",
                    style={
                        "flex": "1",
                        "width": "100%",
                        "height": "calc(100vh - 150px)",  # 动态计算高度（视口高度 - 其他组件高度）
                        "margin": "0",
                    },
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "column",
                "height": "100vh",
                "padding": "0",
            },
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
            if len(current_data["d"]) != 0:
                current_d = current_data["d"]

                # 创建背景轨迹（所有数据）

                # 创建主轨迹（当前数组）
                mask = (current_d >= min_val) & (current_d <= max_val)
                main_trace = [
                    go.Scatter(
                        x=current_data["phi"][mask],
                        y=current_data["theta"][mask],
                        hovertext=current_data["label"][mask],
                        mode="markers",
                        # marker=dict(
                        #     size=5, color=current_d[mask], colorscale="Viridis", opacity=0.8
                        # ),
                        name=f"psi {current_data["psi"]:.1f}",
                    )
                ]
            else:
                main_trace = []

            return {
                "data": background_trace + main_trace,
                "layout": go.Layout(
                    scene=dict(
                        xaxis_title="phi",
                        yaxis_title="theta",
                    ),
                    yaxis=dict(
                        scaleanchor="x",
                        scaleratio=1,
                    ),
                    # margin=dict(l=0, r=0, b=0, t=30),
                    # height=800,
                ),
            }

        return app

    def _get_paint_data(
        self,
    ):

        u = self.detector.config.detector_u
        v = self.detector.config.detector_v

        # psi_range = sorted([-psi for psi in self.detector.config.psi_range])
        psi_grid = np.deg2rad(
            np.arange(
                self.detector.config.psi_range[0],
                self.detector.config.psi_range[1],
                self.step_deg,
            )
        )

        # 对应每一个psi
        data = []
        for psi in psi_grid:
            u_new, v_new = coordinate_transform(
                np.vstack((u, v)),
                rotation_matrix(
                    self.detector.config.detector_w,
                    -psi,  # 因为这里转的是u，所以和晶体转动方向相反
                ),
                is_positive=True,
            )
            # 因为是Qi-Qf，所以这里要乘-1
            points_sphere, points = (
                self.detector.get_available_points_coordinates_white_beam(
                    -1 * self.lattice.pri_k_points,
                    0.98,
                    21.9,
                    u_new,
                    v_new,
                )
            )

            d = 2 * np.pi / np.linalg.norm(points, axis=1)

            label = np.array(
                get_points_labels(
                    -1 * points, self.lattice.lattice_data.conv_reciprocal_matrix
                )
            )

            points_sphere[:, 1] = pi / 2 - points_sphere[:, 1]

            data.append(
                {
                    "phi": np.rad2deg(points_sphere[:, 2]),
                    "theta": np.rad2deg(points_sphere[:, 1]),
                    "label": label,
                    "d": d,
                    "psi": np.rad2deg(psi),
                }
            )
        return data
