from numpy import (
    arctan2,
    sin,
    tan,
    asarray,
    sqrt,
    dot,
    array,
    cos,
    ndarray,
    arccos,
    arcsin,
    pi,
    column_stack,
    zeros_like,
    float64,
    diag,
    clip,
)
from numpy.linalg import inv, norm, cond, pinv, svd


def csc(x):
    return 1 / sin(x)


def cot(x):
    return 1 / tan(x)


def normalize_vector(vector):
    """返回单位向量，处理零向量异常"""
    vector = asarray(vector)
    normal = norm(vector)
    if normal < 1e-10:
        raise ValueError("Input vector cannot be zero")
    return vector / normal


def rotation_matrix(axis, angle):
    """
    返回绕任意轴旋转指定角度的旋转矩阵。此处的矩阵是对行向量操作的。

    Parameters
    ----------
    axis: 一个三维向量，表示旋转轴的方向
    angle: 旋转的角度（弧度）
    """

    axis = normalize_vector(axis)  # 归一化轴向量

    a = cos(angle / 2.0)
    b, c, d = -axis * sin(angle / 2.0)

    # 构造旋转矩阵
    rotation_matrix = array(
        [
            [
                a * a + b * b - c * c - d * d,
                2 * (b * c - a * d),
                2 * (b * d + a * c),
            ],
            [
                2 * (b * c + a * d),
                a * a + c * c - b * b - d * d,
                2 * (c * d - a * b),
            ],
            [
                2 * (b * d - a * c),
                2 * (c * d + a * b),
                a * a + d * d - b * b - c * c,
            ],
        ]
    )
    return rotation_matrix


# def coordinate_transform(
#     old_coord: ndarray,
#     new_basis: ndarray,
#     old_basis=None,
#     is_positive=False,
#     # is_orthogonal=True,
# ) -> ndarray:
#     """
#     坐标线性变换函数

#     Parameters
#     ----------
#     old_coord: 原坐标.

#     new_basis: 新的基函数(行堆叠).若使用主动式的变换,则这里需要输入变换矩阵的逆.

#     old_basis: 旧的基函数(行堆叠).默认为单位阵.

#     is_passive: 是否为主动式变换,若选True,则令变换矩阵为new_basis.

#     # is_orthogonal:是否为正交矩阵,若是,则用转置代替求逆

#     Returns
#     --------
#     变换后的坐标 Nx3 或 Nx2.

#     """
#     if is_positive:
#         transform_matrix = new_basis
#     else:
#         # if is_orthogonal:
#         #     new_basis_inv = new_basis.T
#         # else:
#         #     new_basis_inv = inv(new_basis)

#         new_basis_inv = inv(new_basis)
#         if old_basis is not None:
#             transform_matrix = dot(old_basis, new_basis_inv)
#         else:
#             transform_matrix = new_basis_inv

#     return dot(old_coord, transform_matrix)


def coordinate_transform(
    old_coord: ndarray,
    new_basis: ndarray,
    old_basis=None,
    is_positive=False,
    # is_orthogonal=False,
    dtype=float64,
) -> ndarray:
    """
    高精度坐标变换函数
    Parameters
    ----------
    old_coord: 原坐标.

    new_basis: 新的基函数(行堆叠).若使用主动式的变换,则这里需要输入变换矩阵的逆.

    old_basis: 旧的基函数(行堆叠).默认为单位阵.

    is_passive: 是否为主动式变换,若选True,则令变换矩阵为new_basis.

    # is_orthogonal:是否为正交矩阵,若是,则用转置代替求逆

    Returns
    --------
    变换后的坐标 Nx3 或 Nx2.
    """
    # 类型提升
    old_coord = asarray(old_coord, dtype=dtype)
    new_basis = asarray(new_basis, dtype=dtype)

    if is_positive:
        transform = new_basis
    else:
        # 稳定性优化
        if cond(new_basis) > 1e12:
            transform = pinv(new_basis)  # 伪逆
        else:
            U, s, VT = svd(new_basis)
            s_inv = diag(1.0 / clip(s, 1e-12, None))
            transform = VT.T @ s_inv @ U.T

    # 旧基处理
    if old_basis is not None:
        old_basis = asarray(old_basis, dtype=dtype)
        transform = old_basis @ transform

    return old_coord @ transform


def get_points_labels(
    points: ndarray,
    new_basis: ndarray,
    old_basis=None,
    except_label="",
    decimal_places=2,
    *labels_list,
):
    """
    得到点在新基矢之下的字符串坐标标签,若提供字符串标签,则默认将在之后添加坐标标签.
    """
    label_coord = coordinate_transform(points, new_basis, old_basis)

    # 动态生成坐标字符串（支持2D/3D）
    coord_strs = [
        ",".join(f"{c:.{decimal_places}f}" for c in coord) for coord in label_coord
    ]

    # 带标签的格式处理
    if labels_list:
        return [
            [
                f"{label}_{coord_strs[i]}" if label != except_label else except_label
                for i, label in enumerate(labels)
            ]
            for labels in labels_list
        ]

    # 无标签的返回格式
    return tuple(coord_strs)


def coordinate_transform_from_xyz_to_sphere(points: ndarray) -> ndarray:
    """三维直角坐标转换为球坐标 r,theta, phi (弧度)"""
    # TODO 处理r=0
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    r = norm(points, axis=1)
    nonzero_mask = r != 0

    theta = zeros_like(r)
    theta[nonzero_mask] = arccos(z[nonzero_mask] / r[nonzero_mask])

    phi = arctan2(y, x)
    phi[(x == 0) & (y == 0)] = 0.0

    return column_stack((r, theta, phi))


import numpy as np


def wrap_to_interval(x, min_val: float, max_val: float, closed_interval: bool = False):
    """
    将数值x通过加减整数倍的(max_val - min_val)，使其落在[min_val, max_val)区间

    参数:
        x: 标量或数组
        min_val: 区间下限
        max_val: 区间上限
        closed_interval: 若为True，结果包含max_val，否则为左闭右开区间

    返回:
        调整后的数值，类型与x一致
    """
    # 校验区间合法性
    if max_val <= min_val:
        raise ValueError("max_val必须大于min_val")

    interval = max_val - min_val
    # 核心公式: 平移坐标系到以min_val为原点，取模后平移回去
    adjusted = min_val + (x - min_val) % interval

    # 处理闭区间情况：当余数为0且x非min_val时设为max_val
    if closed_interval:
        is_boundary = np.isclose((x - min_val) % interval, 0) & (x != min_val)
        adjusted = np.where(is_boundary, max_val, adjusted)

    return adjusted


def build_orthogonal_basis(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """构建正交探测器坐标系矩阵 (3x3)"""
    u_unit = normalize_vector(u)
    v_proj = v - u_unit * np.dot(v, u_unit)  # Smith正交
    v_unit = normalize_vector(v_proj)
    w_unit = np.cross(u_unit, v_unit)
    return np.vstack((u_unit, v_unit, w_unit))  # 基向量按行存储
