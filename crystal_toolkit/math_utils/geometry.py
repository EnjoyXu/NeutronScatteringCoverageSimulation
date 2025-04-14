from matplotlib.pylab import norm
import numpy as np
from pkg_resources import parse_requirements

from crystal_toolkit.math_utils.math_utils import (
    coordinate_transform,
    normalize_vector,
)


def points_along_line(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    width: float,
    *labels_list,
) -> tuple[np.ndarray, np.ndarray]:
    """
    线段内的点筛选

    Parameters
    ----------

    points: 3xN 或者 2xN的ndarray,数据点.

    start: 1x3 或者 1x2 的ndarray,线段起点.

    end: 1x3 或者 1x2 的ndarray,线段终点.

    width: 离线段的距离.

    labels_list: 一系列数据标签，与数据点一一对应.

    Returns
    --------
    (0-1之间的坐标,筛选出的数据点,对应的标签元组(若有输入))
    """

    if not all(len(label) == len(points) for label in labels_list):
        raise ValueError("所有标签数组必须与坐标数组长度一致")

    vec = end - start
    unit_vec = normalize_vector(vec)

    # 计算点到线的距离
    diffs = points - start
    cross = np.cross(diffs, unit_vec.reshape(1, 3))
    distances = np.linalg.norm(cross, axis=1)

    # 参数化坐标
    t = np.dot(diffs, unit_vec) / np.linalg.norm(vec)

    mask = (distances < width) & (t >= 0) & (t <= 1)

    filtered_labels = tuple(
        [[item for item, keep in zip(label, mask) if keep] for label in labels_list]
    )

    # return t[mask], points[mask], tuple(filtered_labels)
    return (
        (t[mask], points[mask], *filtered_labels)
        if labels_list
        else (t[mask], points[mask])
    )


def points_in_plane(
    points: np.ndarray,
    plane_point: np.ndarray,
    normal: np.ndarray,
    thickness: float,
    parallel_new_ex: np.ndarray,
    *labels_list,
) -> np.ndarray:
    """
    带法向的平面点筛选.

    Parameters
    ----------
    points: Nx3 或者 Nx2 的ndarray,数据点.

    plane_point: 面上的一点坐标.

    normal: 平面法向量.

    thickness: 点到平面的最大距离.

    parallel_new_ex: 确定平面新的坐标的x方向基矢,新的基矢为该向量在面内的投影.

    labels_list: 一系列数据标签，与数据点一一对应.

    Returns
    --------
    (筛选出的数据点二维坐标,原本数据点,对应的标签元组(若有输入))
    """
    normal = normalize_vector(normal)

    distances = np.abs(np.dot(points - plane_point, normal))
    mask = distances < thickness

    # return points[np.abs(offsets) < thickness]

    filtered_labels = tuple(
        [[item for item, keep in zip(label, mask) if keep] for label in labels_list]
    )

    filter_points = points[mask]

    new_coords = _get_plane_proj_coordinates(
        norm=normal, parallel_new_ex=parallel_new_ex, points=filter_points
    )

    return (
        (new_coords, filter_points, *filtered_labels)
        if labels_list
        else (new_coords, filter_points)
    )


def generate_line_points(r0, r1, num_points=50):
    """
    生成两点之间的等距点

    Returns
    -------
    N x 3 arrays
    """
    # 将输入点转换为 numpy 数组
    r0 = np.array(r0)
    r1 = np.array(r1)

    # 使用 linspace 在 0 到 1 之间生成 num_points 个均匀分布的参数 t
    t_values = np.linspace(0, 1, num_points)

    # 对 r0 和 r1 进行线性插值，得到沿线的点坐标
    points = (1 - t_values)[:, np.newaxis] * r0 + t_values[:, np.newaxis] * r1
    return points


def _get_plane_proj_coordinates(norm, parallel_new_ex, points):
    """
    目标平面法线为norm,新的坐标以parallel_new_ex到目标平面的投影直线为新的x轴,y轴默认为norm与x轴的叉乘.
    """
    norm = normalize_vector(norm)
    parallel_new_ex = np.array(parallel_new_ex)

    ex_new = np.cross(norm, np.cross(norm, parallel_new_ex))
    ex_new = normalize_vector(ex_new)

    if np.dot(ex_new, parallel_new_ex) < 0:
        ex_new *= -1

    ey_new = np.cross(norm, ex_new)
    ey_new = normalize_vector(ey_new)

    # 3x2
    R = np.array([ex_new, ey_new]).T

    new_coords = coordinate_transform(points, R, is_positive=True)

    return new_coords


def get_convex_vertice_points_2d(points_2d, *labels_list):
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points_2d)  # 计算凸包
    vertices_idx = hull.vertices
    ver_points = points_2d[vertices_idx]

    filter_labels = [np.array(label)[vertices_idx] for label in labels_list]

    return (ver_points, *filter_labels) if labels_list else ver_points


def get_perp_plane(point1: np.ndarray, point2: np.ndarray):
    """返回两点之间的中垂面坐标(A,B,C,D)"""
    mid_point = (point1 + point2) / 2

    norm_vector = point2 - point1

    return np.append(norm_vector, [-1 * np.dot(norm_vector, mid_point)])


def get_plane_line_cross(
    line_point1: np.ndarray, line_point2: np.ndarray, plane_parameters: np.ndarray
) -> np.ndarray:
    """得到平面与线段的交点坐标"""

    diff = line_point2 - line_point1
    norm_vector = plane_parameters[:3]

    nominator = -1 * (np.dot(norm_vector, line_point1) + plane_parameters[-1])

    if nominator == 0:
        return line_point1

    demoninator = np.dot(norm_vector, diff)
    return line_point1 + (nominator / demoninator) * diff
