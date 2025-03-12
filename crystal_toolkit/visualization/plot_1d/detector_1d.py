from crystal_toolkit.detector.detector import Detector
from crystal_toolkit.math_utils.geometry import points_along_line
from crystal_toolkit.visualization.plotter_base import BasePlotter
from numpy import append, vstack, repeat, column_stack, sum, array
from numpy.linalg import norm
import plotly.graph_objs as go


class Detector1DPlotter(BasePlotter):
    def __init__(self, detector: Detector, k_points_list, width: float, label_list=[]):
        super().__init__()

        self._detector = detector
        self.k_points_list = k_points_list
        self.label_list = (
            label_list
            if len(label_list) != 0
            else [
                f"{point[0]:.3f},{point[1]:.3f},{point[2]:.3f}"
                for point in self.k_points_list
            ]
        )
        self.width = width

        self.detector_points_list = self._get_detector_coverage()

    def _get_detector_coverage(
        self,
    ):
        paint_points_list = []

        detector_points_all = vstack(self._detector.detector_points_list)

        dE_all = repeat(self._detector.dE, len(self._detector.detector_points_list[0]))

        N = len(self.k_points_list) - 1

        self.__distance_list = array(
            [norm(self.k_points_list[i + 1] - self.k_points_list[i]) for i in range(N)]
        )

        self.__distances_tot = sum(self.__distance_list)

        for i in range(N):

            start = self.k_points_list[i]
            end = self.k_points_list[i + 1]

            distance = self.__distance_list[i]

            x, _, y = points_along_line(
                detector_points_all, start, end, self.width, dE_all
            )
            paint_points_list.append(
                column_stack((x * distance * N / self.__distances_tot, y))
            )
        # 绘图起点
        for i in range(1, N):
            paint_points_list[i][:, 0] += paint_points_list[i - 1][-1, 0]

        return paint_points_list

    def plot(
        self,
    ):

        self.add_traces(
            [
                go.Scatter(x=point[:, 0], y=point[:, 1], mode="markers")
                for point in self.detector_points_list
            ]
        )

        self._apply_layout("Detector Coverage")

        return self.fig

    def _apply_layout(self, title):
        self.fig.update_layout(
            title=title,
            scene=dict(
                # xaxis_title="X (Å)",
                # yaxis_title="Y (Å)",
                # zaxis_title="Z (Å)",
                aspectmode="data",
            ),
            # margin=dict(l=0, r=0, b=0, t=40),
            xaxis=dict(
                range=[-0.2, len(self.k_points_list) + 0.2],
                tickvals=self.label_list,
            ),
            yaxis=dict(title=r"$\Delta\text{E(meV)}$"),
        )

        self.fig.update_xaxes(
            tickvals=[min(point[:, 0]) for point in self.detector_points_list]
            + [max(self.detector_points_list[-1][:, 0])],
            ticktext=self.label_list,  # 设置刻度值  # 设置刻度文本
        )
