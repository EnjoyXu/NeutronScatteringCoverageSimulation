from numpy import sin, tan, asarray, sqrt, dot, array, cos, ndarray
from numpy.linalg import inv, norm


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


def coordinate_transform(
    old_coord: ndarray,
    new_basis: ndarray,
    old_basis=None,
    is_positive=False,
) -> ndarray:
    """
    坐标变换函数

    Parameters
    ----------
    old_coord: 原坐标.

    new_basis: 新的基函数(行堆叠).若使用主动式的变换,则这里需要输入变换矩阵的逆.

    old_basis: 旧的基函数(行堆叠).默认为单位阵.

    is_passive: 是否为主动式变换,若选True,则令变换矩阵为new_basis.

    Returns
    --------
    变换后的坐标 Nx3 或 Nx2.

    """
    if is_positive:
        transform_matrix = new_basis
    else:
        if old_basis is not None:
            transform_matrix = dot(old_basis, inv(new_basis))
        else:
            transform_matrix = inv(new_basis)

    return dot(old_coord, transform_matrix)


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
