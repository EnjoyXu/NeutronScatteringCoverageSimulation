import numpy as np

from crystal_toolkit.math_utils.math_utils import coordinate_transform


def generate_lattice_coordinates(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    Nx: int,
    Ny: int,
    Nz: int,
    atom_pos=[0, 0, 0],
) -> tuple[np.ndarray, ...]:
    """
    生成三维晶格所有原子的坐标和标签

    Parameters
    ----------
    a (np.ndarray): 晶格向量a [ax, ay, az]
    b (np.ndarray): 晶格向量b [bx, by, bz]
    c (np.ndarray): 晶格向量c [cx, cy, cz]
    atom_pos (np.ndarray): 原子在晶胞内的相对位置(N,3)
    Nx (int): x方向扩展单元数
    Ny (int): y方向扩展单元数
    Nz (int): z方向扩展单元数

    Returns
    -------
    tuple: (x坐标数组, y坐标数组, z坐标数组, 标签数组)
    """

    atom_pos = np.atleast_2d((atom_pos))

    # 生成所有晶格索引组合
    n_range = np.arange(-Nx, Nx + 1)
    m_range = np.arange(-Ny, Ny + 1)
    k_range = np.arange(-Nz, Nz + 1)
    n, m, k = np.meshgrid(n_range, m_range, k_range, indexing="ij")

    # 转换为三维位移向量 (Nx*Ny*Nz, 3)
    indices = np.column_stack([n.ravel(), m.ravel(), k.ravel()])

    # 将原子位置转换为齐次坐标 (便于向量化计算)
    atom_basis = np.array([a, b, c]).T  # 3x3 基向量矩阵

    atom_pos_h = np.hstack([atom_pos, np.ones((len(atom_pos), 1))])  # 齐次坐标

    # 计算所有原子位置 (向量化计算)
    displacements = indices @ atom_basis.T  # (N_cells, 3)
    all_positions = (atom_pos_h[:, None, :3] + displacements).reshape(-1, 3)

    # 生成标签
    atom_indices = np.repeat(np.arange(len(atom_pos)), len(indices))
    labels = np.array(
        [f"{ni}, {mi}, {ki}: {ai}" for (ni, mi, ki), ai in zip(indices, atom_indices)]
    )

    # 拆分坐标分量
    return (all_positions, labels)


def generate_conv_and_pri_lattice_3d_coordinates(
    Nx, Ny, Nz, conv_reciprocal_matrix, pri_recirpocal_matrix, tolerance=0.05
):
    """
    生成k空间单胞和初级原胞的格点.
    """
    # 得到单胞坐标
    conv_points, conv_labels = generate_lattice_coordinates(
        conv_reciprocal_matrix[0],
        conv_reciprocal_matrix[1],
        conv_reciprocal_matrix[2],
        Nx,
        Ny,
        Nz,
    )

    # 转换为b1,b2,b3为基的坐标
    pri_coords = coordinate_transform(conv_points, pri_recirpocal_matrix)

    # 筛选整数坐标
    mask = _get_integer_coordinate_mask(pri_coords, tolerance)

    return (conv_points, conv_labels, conv_points[mask], conv_labels[mask])


def _get_integer_coordinate_mask(coords: np.ndarray, tolerance: float) -> np.ndarray:
    """生成整数坐标筛选掩模"""
    residual = np.abs(coords - np.round(coords))
    return np.all(residual < tolerance, axis=1)
