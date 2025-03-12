from numpy import ndarray

from crystal_toolkit.math_utils.geometry import points_in_plane
from crystal_toolkit.visualization.plotter_base import BasePlotter


class BasePlotter2D(BasePlotter):
    def __init__(
        self,
        norm_vector: ndarray,
        plane_point: ndarray,
        thickness: float,
        parallel_new_ex: ndarray,
    ):
        super().__init__()
        self.norm_vector = norm_vector
        self.plane_point = plane_point
        self.thickness = thickness
        self.parallel_new_ex = parallel_new_ex

    def _get_plane_slice(self, points, *labels_list):
        """
        得到三维点在平面的投影
        """
        return points_in_plane(
            points,
            self.plane_point,
            self.norm_vector,
            self.thickness,
            self.parallel_new_ex,
            *labels_list,
        )
