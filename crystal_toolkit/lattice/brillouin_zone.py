from math import e
from typing import Tuple
from matplotlib.pyplot import scatter
import numpy as np

from crystal_toolkit.math_utils.geometry import generate_line_points, points_in_plane


def get_wigner_size_cell_3d(a, b, c):
    cell = np.vstack((a, b, c))
    # assert cell.shape == (3, 3)

    px, py, pz = np.tensordot(cell, np.mgrid[-1:2, -1:2, -1:2], axes=[0, 0])
    points = np.c_[px.ravel(), py.ravel(), pz.ravel()]

    from scipy.spatial import Voronoi

    vor = Voronoi(points)

    bz_facets = []
    bz_ridges = []
    bz_vertices = []

    for pid, rid in zip(vor.ridge_points, vor.ridge_vertices):
        if pid[0] == 13 or pid[1] == 13:
            bz_ridges.append(vor.vertices[np.r_[rid, [rid[0]]]])
            bz_facets.append(vor.vertices[rid])
            bz_vertices += rid

    bz_vertices = vor.vertices[list(set(bz_vertices))]

    return bz_vertices, bz_ridges, bz_facets


def get_wigner_size_cell_2d(
    a, b, c, plane_point, norm_vector, width, parallel_new_ex
) -> np.ndarray:
    """得到1BZ边轮廓的点"""

    def _collect_edge_points(
        edges_3d: list,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        收集所有三维边界的线段点
        """
        points_all = np.empty((0, 3))

        for edge in edges_3d:
            # 生成闭合多边形的所有顶点对

            start_points = edge
            end_points = np.roll(start_points, shift=-1, axis=0)

            for start, end in zip(start_points, end_points):
                line_points = generate_line_points(start, end)

                points_all = np.vstack((points_all, line_points))

        return points_all

    _, edges_3d, _ = get_wigner_size_cell_3d(a, b, c)

    edge_points = _collect_edge_points(edges_3d)

    return points_in_plane(
        edge_points, plane_point, norm_vector, width, parallel_new_ex
    )
