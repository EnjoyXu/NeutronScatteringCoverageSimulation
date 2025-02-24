# %%
import numpy as np
import plotly.graph_objs as go
from itertools import product
from numpy import sin, cos, tan, pi
from plotly.offline import init_notebook_mode, iplot, iplot_mpl


def csc(x):
    return 1 / sin(x)


def cot(x):
    return 1 / tan(x)


def rotation_matrix(axis, angle):
    """
    返回绕任意轴旋转指定角度的旋转矩阵。此处的矩阵是对列向量操作的。

    Parameters
    ----------
    axis: 一个三维向量，表示旋转轴的方向
    angle: 旋转的角度（弧度）
    """
    # 下面这个是顺时针的旋转，所以取负号
    angle = -1 * angle
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))  # 归一化轴向量

    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)

    # 构造旋转矩阵
    rotation_matrix = np.array(
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


def where_points_along(r0, r1, width, data_x, data_y, data_z, dE):
    """
    返回归一化数组t和激发能量dE

    Parameters
    ----------
    r0,r1: 需要截取的线段的端点坐标

    width: 判断是否在直线上的距离

    data_x: 需要判断的点的x坐标

    data_y: 需要判断的点的y坐标

    data_z: 需要判断的点的z坐标

    dE: 需要判断点的激发能量
    """
    r0 = np.array(r0)
    r1 = np.array(r1)
    v = r1 - r0
    v = v / np.linalg.norm(v)

    # 判断离直线的距离
    distance = np.linalg.norm(
        np.cross(np.vstack((data_x - r0[0], data_y - r0[1], data_z - r0[2])).T, v),
        axis=1,
    )

    idx = np.argwhere(distance < width).T[0]
    data_x = data_x[idx]
    data_y = data_y[idx]
    data_z = data_z[idx]
    dE = dE[idx]

    # 判断是否落在直线内

    t = np.dot(
        np.vstack((data_x - r0[0], data_y - r0[1], data_z - r0[2])).T, v
    ) / np.linalg.norm(r1 - r0)

    idx = np.argwhere((0 <= t) & (t <= 1)).T[0]

    t = t[idx]
    dE = dE[idx]

    return t, dE


def where_points_in_plane(r0, norm, width, data_x, data_y, data_z, others_list=[]):
    r0 = np.array(r0)
    norm = np.array(norm)
    norm = norm / np.linalg.norm(norm)

    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_z = np.array(data_z)

    idx = (
        np.abs(
            np.dot(
                norm,
                np.vstack(
                    (
                        data_x - r0[0],
                        data_y - r0[1],
                        data_z - r0[2],
                    )
                ),
            )
        )
        < width
    )

    out = [data_x[idx], data_y[idx], data_z[idx]]

    if len(others_list) != 0:
        others_out_list = []

        for i in range(len(others_list)):
            others_out_list.append(np.array(others_list[i])[idx])

        return out, others_out_list

    return out


def generate_line_points(r0, r1, num_points=50):
    """
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


def get_plane_coordinates(
    norm,
    parallel_new_ex,
    data_x,
    data_y,
    data_z,
):
    """
    目标平面法线为norm，新的坐标以parallel_new_ex到目标平面的投影直线为新的x轴，y轴默认为norm与x轴的叉乘。
    """
    norm = np.array(norm)
    norm = norm / np.linalg.norm(norm)
    parallel_new_ex = np.array(parallel_new_ex)
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_z = np.array(data_z)

    ex_new = np.cross(norm, np.cross(norm, parallel_new_ex))
    ex_new = ex_new / np.linalg.norm(ex_new)

    if np.dot(ex_new, parallel_new_ex) < 0:
        ex_new *= -1

    ey_new = np.cross(norm, ex_new)
    ey_new = ey_new / np.linalg.norm(ey_new)

    # 2x3
    R = np.array([ex_new, ey_new])

    x, y = np.dot(R, np.vstack((data_x, data_y, data_z)))

    return x, y


def plane_judge_and_to_2D_cor(
    point_on_the_plane,
    norm_vector,
    parallel_new_ex,
    width,
    data_x,
    data_y,
    data_z,
    others_list=[],
):
    """
    三维点判断是否在面内，并转化为2D坐标。
    """
    if len(others_list) == 0:
        r = where_points_in_plane(
            point_on_the_plane,
            norm_vector,
            width,
            data_x,
            data_y,
            data_z,
            others_list,
        )
    else:
        r, others_list = where_points_in_plane(
            point_on_the_plane,
            norm_vector,
            width,
            data_x,
            data_y,
            data_z,
            others_list,
        )
    x = r[0]
    y = r[1]
    z = r[2]

    x, y = get_plane_coordinates(norm_vector, parallel_new_ex, x, y, z)

    if len(others_list) == 0:
        return x, y

    if len(others_list) == 1:
        others_list = others_list[0]

    return x, y, others_list


class high_symmetry_kpoints(object):
    """
    返回 Nx3坐标数组以及1xN标签数组
    """

    @classmethod
    def get_RHL1(cls, a1, a2, b1, b2, b3):
        cos_alpha = np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))
        eta = (1 + 4 * cos_alpha) / (2 + 4 * cos_alpha)
        nu = 3 / 4 - eta / 2
        RHL1 = (
            np.array(
                [
                    [0, 0, 0],  # gamma
                    eta * b1 + b2 / 2 + (1 - eta) * b3,  # B
                    b1 / 2 + (1 - eta) * b2 + (eta - 1) * b3,  # B1
                    b1 / 2 + b2 / 2,  # F
                    b1 / 2,  # L
                    -b3 / 2,  # L1
                    eta * b1 + nu * (b2 + b3),  # P,
                    (1 - nu) * (b1 + b2) + (1 - eta) * b3,  # P1
                    nu * (b1 + b2) + (eta - 1) * b3,  # P2
                    (1 - nu) * b1 + nu * b2,  # Q
                    nu * (b1 - b3),  # X
                    (b1 + b2 + b3) / 2,  # Z
                ],
            ),
            ["Γ", "B", "B1", "F", "L", "L1", "P", "P1", "P2", "Q", "X", "Z"],
        )

        return RHL1

    @classmethod
    def get_HEX(cls, b1, b2, b3):
        HEX = (
            np.array(
                [
                    [0, 0, 0],  # gamma
                    0 * b1 + 0 * b2 + 1 / 2 * b3,  # A
                    1 / 3 * b1 + 1 / 3 * b2 + 1 / 2 * b3,  # H
                    1 / 3 * b1 + 1 / 3 * b2 + 0 * b3,  # K
                    1 / 2 * b1 + 0 * b2 + 1 / 2 * b3,  # L
                    1 / 2 * b1 + 0 * b2 + 0 * b3,  # M
                ],
            ),
            ["Γ", "A", "H", "K", "L", "M"],
        )
        return HEX


class LatticeUtility(object):
    @classmethod
    def rotate_to_uv_plane(cls, u, v, *vectors):
        """
        正交坐标基矢落在uv平面内,且u为x轴。
        """
        u = np.array(u)
        v = np.array(v)
        w = np.cross(u, v)
        w /= np.linalg.norm(w)
        u /= np.linalg.norm(u)

        v = np.cross(w, u)
        v /= np.linalg.norm(v)

        vector_out_list = [np.dot(vector, np.vstack((u, v, w)).T) for vector in vectors]
        return vector_out_list

    @classmethod
    def get_lattice_vector(cls, a, b, c, alpha, beta, gamma):
        """
        Obtain lattice vectors from crystal lattice parameters.
        Vectors a_star and b_star lie in the x-y plane, with vector a_star pointing +x direction.`
        """
        if alpha > 2 * pi:
            alpha = alpha / 180 * pi

        if beta > 2 * pi:
            beta = beta / 180 * pi

        if gamma > 2 * pi:
            gamma = gamma / 180 * pi

        lattice_a = np.array([a, 0, 0])
        lattice_b = b * np.array([cos(gamma), sin(gamma), 0])
        lattice_c = c * np.array(
            [
                cos(beta),
                cos(alpha) * csc(gamma) - cos(beta) * cot(gamma),
                np.sqrt(
                    sin(beta) ** 2
                    - (cos(alpha) * csc(gamma) - cos(beta) * cot(gamma)) ** 2
                ),
            ],
        )

        (
            reciprocal_lattice_a,
            reciprocal_lattice_b,
            reciprocal_lattice_c,
        ) = LatticeUtility.get_reciprocal_lattice_from_lattice(
            lattice_a,
            lattice_b,
            lattice_c,
        )

        (
            lattice_a,
            lattice_b,
            lattice_c,
            reciprocal_lattice_a,
            reciprocal_lattice_b,
            reciprocal_lattice_c,
        ) = LatticeUtility.rotate_to_uv_plane(
            reciprocal_lattice_a,
            reciprocal_lattice_b,
            lattice_a,
            lattice_b,
            lattice_c,
            reciprocal_lattice_a,
            reciprocal_lattice_b,
            reciprocal_lattice_c,
        )

        return (
            lattice_a,
            lattice_b,
            lattice_c,
            reciprocal_lattice_a,
            reciprocal_lattice_b,
            reciprocal_lattice_c,
        )

    @classmethod
    def get_reciprocal_lattice_from_lattice(cls, a, b, c):
        A = np.vstack((a, b, c))

        reciprocal_lattice_a, reciprocal_lattice_b, reciprocal_lattice_c = (
            np.linalg.inv(A).T * 2 * pi
        )
        return reciprocal_lattice_a, reciprocal_lattice_b, reciprocal_lattice_c

    @classmethod
    def get_wigner_size_cell_3d(cls, a, b, c):
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
            # WHY 13 ????
            # The Voronoi ridges/facets are perpendicular to the lines drawn between the
            # input points. The 14th input point is [0, 0, 0].
            if pid[0] == 13 or pid[1] == 13:
                bz_ridges.append(vor.vertices[np.r_[rid, [rid[0]]]])
                bz_facets.append(vor.vertices[rid])
                bz_vertices += rid

        bz_vertices = vor.vertices[list(set(bz_vertices))]

        # convex = ConvexHull(bz_vertices)
        # for simplicy in convex.simplices:
        #     # if simplicy[0] == 13 or simplicy[1] == 13:
        #     print(simplicy)
        #     bz_facets.append(convex.points[simplicy])

        return bz_vertices, bz_ridges, bz_facets

    @classmethod
    def get_wigner_size_cell_2d(cls, a, b, c, point_on_the_plane, norm_vector, width):
        _, edges, _ = LatticeUtility.get_wigner_size_cell_3d(a, b, c)

        x = np.array([])
        y = np.array([])
        z = np.array([])

        for edge in edges:
            for i in range(len(edge) - 1):

                x1, y1, z1 = generate_line_points(edge[i], edge[i + 1]).T

                x = np.append(x, x1)
                y = np.append(y, y1)
                z = np.append(z, z1)
        else:
            x1, y1, z1 = generate_line_points(edge[-1], edge[0]).T

            x = np.append(x, x1)
            y = np.append(y, y1)
            z = np.append(z, z1)

        x, y, z = where_points_in_plane(point_on_the_plane, norm_vector, width, x, y, z)

        return (x, y, z)

    @classmethod
    def unit_cell_3d(cls, a, b, c, atom_pos, Nx, Ny, Nz):
        """Make arrays of x-, y- and z-positions of a lattice from the
        lattice vectors, the atom positions and the number of unit cells.

        Parameters:
        -----------
        a : list
            First lattice vector
        b : list
            Second lattice vector
        c : list
            Third lattice vector
        atom_pos : list
            Positions of atoms in the unit cells in terms of a, b and c
        Nx : int
            number of unit cells in the x-direction to be plotted
        Ny : int
            number of unit cells in the y-direction to be plotted
        Nz : int
            number of unit cells in the z-direction to be plotted

        Returns:
        --------
        latt_coord_x : numpy.ndarray
            Array containing the x-coordinates of all atoms to be plotted
        latt_coord_y : numpy.ndarray
            Array containing the y-coordinates of all atoms to be plotted
        latt_coord_z : numpy.ndarray
            Array containing the z-coordinates of all atoms to be plotted
        """
        latt_coord_x = []
        latt_coord_y = []
        latt_coord_z = []
        labels = []
        for index, atom in enumerate(atom_pos):
            xpos = atom[0] * a[0] + atom[1] * b[0] + atom[2] * c[0]
            ypos = atom[0] * a[1] + atom[1] * b[1] + atom[2] * c[1]
            zpos = atom[0] * a[2] + atom[1] * b[2] + atom[2] * c[2]
            label = [
                f"{n}, {m}, {k}: {index}"
                for n, m, k in product(range(-Nx, Nx), range(-Ny, Ny), range(-Nz, Nz))
            ]
            xpos_all = [
                xpos + n * a[0] + m * b[0] + k * c[0]
                for n, m, k in product(range(-Nx, Nx), range(-Ny, Ny), range(-Nz, Nz))
            ]
            ypos_all = [
                ypos + n * a[1] + m * b[1] + k * c[1]
                for n, m, k in product(range(-Nx, Nx), range(-Ny, Ny), range(-Nz, Nz))
            ]
            zpos_all = [
                zpos + n * a[2] + m * b[2] + k * c[2]
                for n, m, k in product(range(-Nx, Nx), range(-Ny, Ny), range(-Nz, Nz))
            ]
            latt_coord_x.append(xpos_all)
            latt_coord_y.append(ypos_all)
            latt_coord_z.append(zpos_all)
            labels.append(label)
        latt_coord_x = np.array(latt_coord_x).flatten()
        latt_coord_y = np.array(latt_coord_y).flatten()
        latt_coord_z = np.array(latt_coord_z).flatten()
        labels = np.array(labels).flatten()
        return latt_coord_x, latt_coord_y, latt_coord_z, labels

    @classmethod
    def get_detector_coverage_coordinates(
        cls, E_i, dE, u, v, phi_range_list, theta_range_list, psi_range
    ):
        """
        角度为单位为degree
        """
        if len(phi_range_list) != len(theta_range_list):
            raise ValueError(
                "Length of phi range list should be equal to that of theta range list!"
            )

        # 能量转化为波矢
        k_i = 0.695 * np.sqrt(E_i)
        k_f = 0.695 * np.sqrt(E_i - dE)

        # 归一化
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)
        w = np.cross(u, v)
        w = w / np.linalg.norm(w)
        v = np.cross(w, u)  # 正交v

        # 定义从x,y,z到球坐标的基矢变换矩阵R
        # (e_x,e_y,e_z) R= (e_x',e_y',e_z')
        # r = R r'
        R = np.array([u, v, w]).T

        angle_spacing = 2
        theta_spacing = 2

        X = Y = Z = np.array([])

        for i in range(len(phi_range_list)):
            phi_range = phi_range_list[i]
            theta_range = theta_range_list[i]
            phi_arr = np.arange(phi_range[0], phi_range[1], angle_spacing) / 180 * np.pi
            theta_arr = (
                np.arange(theta_range[0], theta_range[1], theta_spacing) / 180 * np.pi
            )

            phi_arr, theta_arr = np.meshgrid(phi_arr, theta_arr)

            X = np.hstack((X, (k_f * np.sin(theta_arr) * np.cos(phi_arr)).reshape(-1)))
            Y = np.hstack((Y, (k_f * np.sin(theta_arr) * np.sin(phi_arr)).reshape(-1)))
            Z = np.hstack((Z, k_f * np.cos(theta_arr).reshape(-1)))

        # (X,Y,Z)
        r_coordinates = np.dot(R, np.array([X, Y, Z])) - k_i * u.reshape((-1, 1))

        rotateded_coordinates = np.array([[], [], []])
        for angle in np.arange(psi_range[0], psi_range[1], angle_spacing) / 180 * np.pi:
            rotateded_coordinates = np.hstack(
                (
                    rotateded_coordinates,
                    np.dot(rotation_matrix(w, angle), r_coordinates),
                )
            )
        return rotateded_coordinates


class LatticePLots(object):
    al_powder = {"q_list": [2.69, 3.10, 4.39], "color": "red"}
    cu_powder = {"q_list": [3.04, 3.51, 4.97], "color": "blue"}
    epsilon = 1e-3

    def set_lattice_parameters(self, a_par, b_par, c_par, alpha, beta, gamma):
        """
        (单胞)晶体参数
        """
        self.a_par = a_par
        self.b_par = b_par
        self.c_par = c_par
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.a, self.b, self.c, self.a_star, self.b_star, self.c_star = (
            LatticeUtility.get_lattice_vector(a_par, b_par, c_par, alpha, beta, gamma)
        )
        self.a_star_par = np.linalg.norm(self.a_star)
        self.b_star_par = np.linalg.norm(self.b_star)
        self.c_star_par = np.linalg.norm(self.c_star)

    # from cif file
    def __init__(self, cif_path=""):

        self.E_spacing = 1
        if len(cif_path) != 0:
            from pymatgen.core.structure import Structure
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            from pymatgen.symmetry.kpath import KPathLatimerMunro

            structure = Structure.from_file(cif_path)
            cov_lattice = structure.lattice

            cov_reciprocal_lattice = cov_lattice.reciprocal_lattice

            # 创建空间群分析器
            analyzer = SpacegroupAnalyzer(structure, symprec=1e-5, angle_tolerance=0.5)

            # 获取初基原胞
            primitive_structure = analyzer.find_primitive(True)
            pri_lattice = primitive_structure.lattice

            pri_reciprocal_lattice = pri_lattice.reciprocal_lattice

            self.a_par, self.b_par, self.c_par = (
                cov_lattice.a,
                cov_lattice.b,
                cov_lattice.c,
            )
            self.alpha, self.beta, self.gamma = (
                cov_lattice.alpha,
                cov_lattice.beta,
                cov_lattice.gamma,
            )
            self.a, self.b, self.c = cov_lattice.matrix

            self.a_star_par, self.b_star_par, self.c_star_par = (
                cov_reciprocal_lattice.a,
                cov_reciprocal_lattice.b,
                cov_reciprocal_lattice.c,
            )

            self.a_star, self.b_star, self.c_star = cov_reciprocal_lattice.matrix

            self.a1_par, self.a2_par, self.a3_par = (
                pri_lattice.a,
                pri_lattice.b,
                pri_lattice.c,
            )

            self.a1, self.a2, self.a3 = pri_lattice.matrix

            self.b1_par, self.b2_par, self.b3_par = (
                pri_reciprocal_lattice.a,
                pri_reciprocal_lattice.b,
                pri_reciprocal_lattice.c,
            )
            self.b1, self.b2, self.b3 = pri_reciprocal_lattice.matrix

            cor, labels = KPathLatimerMunro(primitive_structure).get_kpoints()
            cor = np.array(cor)

            for i, label in enumerate(labels):
                if label != "":
                    c = cor[i]
                    l = np.vstack((self.a, self.b, self.c)).T / 2 / np.pi
                    l = np.dot(c, l)
                    new_label = label + ":" + f"({l[0]:.1f},{l[1]:.1f},{l[2]:.1f})"
                    labels[i] = new_label

            self.set_high_symmetry_points(cor[:, 0], cor[:, 1], cor[:, 2], labels)

            self.transformation_matrix_from_abc_to_a1a2a3 = np.dot(
                pri_lattice.matrix, np.linalg.inv(cov_lattice.matrix)
            ).T

    def set_conv_lattice(self, a_par, b_par, c_par, alpha, beta, gamma):
        """
        (单胞)晶体参数
        """
        self.a_par = a_par
        self.b_par = b_par
        self.c_par = c_par
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.a, self.b, self.c, self.a_star, self.b_star, self.c_star = (
            LatticeUtility.get_lattice_vector(a_par, b_par, c_par, alpha, beta, gamma)
        )

        self.a_star_par = np.linalg.norm(self.a_star)
        self.b_star_par = np.linalg.norm(self.b_star)
        self.c_star_par = np.linalg.norm(self.c_star)

        return self

    def set_pri_lattice(self, a1_cor, a2_cor, a3_cor):
        """
        在单胞坐标(a,b,c)下,初基原胞矢量的坐标
        """
        if len(a1_cor) != 3 or len(a2_cor) != 3 or len(a3_cor) != 3:
            raise ValueError("a1,a2,a3 should be a 1x3 vector!")

        a1_cor = np.array(a1_cor)
        a2_cor = np.array(a2_cor)
        a3_cor = np.array(a3_cor)

        self.transformation_matrix_from_abc_to_a1a2a3 = np.vstack(
            (a1_cor, a2_cor, a3_cor)
        ).T

        self.a1 = a1_cor[0] * self.a + a1_cor[1] * self.b + a1_cor[2] * self.c
        self.a2 = a2_cor[0] * self.a + a2_cor[1] * self.b + a2_cor[2] * self.c
        self.a3 = a3_cor[0] * self.a + a3_cor[1] * self.b + a3_cor[2] * self.c

        self.b1, self.b2, self.b3 = LatticeUtility.get_reciprocal_lattice_from_lattice(
            self.a1, self.a2, self.a3
        )

        self.a1_par = np.linalg.norm(self.a1)
        self.a2_par = np.linalg.norm(self.a2)
        self.a3_par = np.linalg.norm(self.a3)
        self.b1_par = np.linalg.norm(self.b1)
        self.b2_par = np.linalg.norm(self.b2)
        self.b3_par = np.linalg.norm(self.b3)

        return self

    def set_high_symmetry_points(self, x, y, z, label):
        self.high_symmetry_points_coordinates = np.array([x, y, z])
        self.high_symmetry_points_coordinates_label = label

        return self

    def set_detector_parameters(
        self, E_i, u_cor, v_cor, phi_range, theta_range, psi_range
    ):
        """
        E_i 单位为meV,
        u,v为a_star,b_star,c_star为基的坐标,
        角度单位均为degree
        """
        self.E_i = E_i
        self.u = (
            self.a_star * u_cor[0] + self.b_star * u_cor[1] + self.c_star * u_cor[2]
        )
        self.v = (
            self.a_star * v_cor[0] + self.b_star * v_cor[1] + self.c_star * v_cor[2]
        )
        # 为了兼容不平整的探测器范围,phi 和 theta 可以是range列表，如 phi_range =[[-20,20],[-30,30]]
        if type(phi_range[0]) == type([]):
            self.phi_range_list = [sorted(x) for x in phi_range]
            self.theta_range_list = [
                sorted([90 - x[0], 90 - x[1]]) for x in theta_range
            ]
        else:
            self.phi_range_list = [sorted(phi_range)]
            self.theta_range_list = [sorted([90 - theta_range[0], 90 - theta_range[1]])]

        self.psi_range = sorted(psi_range)

        return self

    def set_magnetic_points(
        self,
        magnetic_modulation_vector_list,
        constrain_function=None,
        is_coordinate=True,
    ):
        """
        默认是以a_star,b_star,c_star为基的坐标,若直接输入绝对位置,可将is_coordinate设置为False。若要对磁峰画图范围有限制,传入constrain_function(x,y,z),该函数应当输出True/False。
        """

        if len(np.array(magnetic_modulation_vector_list).shape) == 1:
            if len(magnetic_modulation_vector_list) != 3:
                raise ValueError("magnetic modulation should be 3 dimensional!")
            magnetic_modulation_vector_list = [magnetic_modulation_vector_list]

        if is_coordinate:
            for i in range(len(magnetic_modulation_vector_list)):
                h, k, l = magnetic_modulation_vector_list[i]
                magnetic_modulation_vector_list[i] = (
                    h * self.a_star + k * self.b_star + l * self.c_star
                )
        self.magnetic_modulation_vector_list = magnetic_modulation_vector_list

        self.magnetic_constrain_function = constrain_function

        return self

    def print_hkl_list(self, energy=5, unit="meV", N=4):
        """
        Parameters
        -----------
        energy: energy of incident neutrons. Default is 5.

        unit: unit of the energy, A (for wavelength) or meV (for energy). Default is meV.

        N: largest h/k/l
        """
        if energy == 0:
            raise ValueError("Energy of incident neutrons cannot be 0!")

        N = 4
        out_list = []

        for i in range(-N, N + 1):
            for j in range(-N, N + 1):
                for k in range(-N, N + 1):
                    if i == j == k == 0:
                        continue
                    pri_cor = np.array([i, j, k])
                    conv_cor = np.dot(
                        pri_cor,
                        np.linalg.inv(self.transformation_matrix_from_abc_to_a1a2a3),
                    )

                    # conv_cor = np.where(np.abs(conv_cor)<LatticePLots.epsilon,0,conv_cor)

                    conv_cor = np.round(conv_cor, 2)

                    q = np.linalg.norm(
                        conv_cor[0] * self.a_star
                        + conv_cor[1] * self.b_star
                        + conv_cor[2] * self.c_star
                    )
                    d = np.pi * 2 / q

                    if unit == "meV":
                        wavelength = 9.0434 / np.sqrt(energy)
                    elif unit == "A":
                        wavelength = energy

                    # 2theta
                    theta2 = 2 * np.arcsin(wavelength / 2 / d) / np.pi * 180

                    theta2_harmo = 2 * np.arcsin(wavelength / d) / np.pi * 180

                    out_list.append((conv_cor, q, d, theta2, theta2_harmo))

        # 以q的大小排序
        out_list = sorted(out_list, key=lambda x: x[1])
        msg = "h\tk\tl\t|Q|\td\t2theta(n=1)\t2theta(n=2)\n"
        for cor, q, d, theta1, theta2 in out_list:
            msg += f"{cor[0]}\t{cor[1]}\t{cor[2]}\t{q}\t{d}\t{theta1}\t{theta2}\n"
        print(msg)

    def get_hkl_vector(self, h, k, l):
        return h * self.a_star + k * self.b_star + l * self.c_star

    def xyz_to_hkl(self, x, y, z):
        """
        Return
        ------
        x,y,z均为一维array数组
        """
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        r = np.vstack((x, y, z))

        P = np.vstack((self.a_star, self.b_star, self.c_star)).T

        h, k, l = np.dot(np.linalg.inv(P), r)

        return h, k, l

    def get_plotting_Brag_lattice_coordinates(self, Nx, Ny, Nz):
        klatt_x_conv, klatt_y_conv, klatt_z_conv, label_conv = (
            LatticeUtility.unit_cell_3d(
                self.a_star, self.b_star, self.c_star, [[0, 0, 0]], Nx, Ny, Nz
            )
        )

        conv_h = np.array(
            [n for n, m, k in product(range(-Nx, Nx), range(-Ny, Ny), range(-Nz, Nz))]
        )

        conv_k = np.array(
            [m for n, m, k in product(range(-Nx, Nx), range(-Ny, Ny), range(-Nz, Nz))]
        )

        conv_l = np.array(
            [k for n, m, k in product(range(-Nx, Nx), range(-Ny, Ny), range(-Nz, Nz))]
        )

        pri_x, pri_y, pri_z = np.dot(
            (self.transformation_matrix_from_abc_to_a1a2a3).T,
            np.vstack((conv_h, conv_k, conv_l)),
        )

        is_x = np.abs(pri_x - np.round(pri_x)) < 0.05
        is_y = np.abs(pri_y - np.round(pri_y)) < 0.05
        is_z = np.abs(pri_z - np.round(pri_z)) < 0.05

        result = is_x * is_y * is_z

        idx_lst = np.argwhere(result == True).flatten()

        klatt_x_pri = klatt_x_conv[idx_lst].flatten()
        klatt_y_pri = klatt_y_conv[idx_lst].flatten()
        klatt_z_pri = klatt_z_conv[idx_lst].flatten()
        label_pri = label_conv[idx_lst].flatten()
        return (
            klatt_x_conv,
            klatt_y_conv,
            klatt_z_conv,
            label_conv,
            klatt_x_pri,
            klatt_y_pri,
            klatt_z_pri,
            label_pri,
        )

    @classmethod
    def gen_lattice_trace_3d_list(
        cls,
        a,
        b,
        c,
        x_coord,
        y_coord,
        z_coord,
        lattice_label,
        lattice_color="rgba(0,0,0,.5)",
        vector_color_list=[
            "rgb(255,0,0)",
            "rgb(0,255,0)",
            "rgb(0,0,255)",
        ],
        vector_size=0.1,
    ):
        lattice_trace = go.Scatter3d(
            x=x_coord,
            y=y_coord,
            z=z_coord,
            hovertext=lattice_label,
            mode="markers",
            marker=dict(
                size=4, color=lattice_color, line=dict(width=2, color="rgb(0, 0, 0)")
            ),
        )
        vector_a = go.Scatter3d(
            x=[0, a[0]],
            y=[0, a[1]],
            z=[0, a[2]],
            marker=dict(
                size=0,
            ),
            mode="lines",
            text="v1",
            name="v1",
            line=dict(color=vector_color_list[0], width=6),
        )

        cone_a = go.Cone(
            x=[0, a[0]],
            y=[0, a[1]],
            z=[0, a[2]],
            u=[0, a[0]],
            v=[0, a[1]],
            w=[0, a[2]],
            text="v1",
            name="v1",
            sizemode="absolute",
            sizeref=vector_size,
            showscale=False,
            colorscale=[[0, vector_color_list[0]], [1, vector_color_list[0]]],
        )

        vector_b = go.Scatter3d(
            x=[0, b[0]],
            y=[0, b[1]],
            z=[0, b[2]],
            marker=dict(
                size=0,
            ),
            mode="lines",
            text="v2",
            name="v2",
            line=dict(color=vector_color_list[1], width=6),
        )

        cone_b = go.Cone(
            x=[0, b[0]],
            y=[0, b[1]],
            z=[0, b[2]],
            u=[0, b[0]],
            v=[0, b[1]],
            w=[0, b[2]],
            name="v2",
            sizemode="absolute",
            sizeref=vector_size,
            # showlegend=False,
            showscale=False,
            colorscale=[[0, vector_color_list[1]], [1, vector_color_list[1]]],
        )

        vector_c = go.Scatter3d(
            x=[0, c[0]],
            y=[0, c[1]],
            z=[0, c[2]],
            marker=dict(
                size=0,
            ),
            text="v3",
            name="v3",
            mode="lines",
            line=dict(color=vector_color_list[2], width=6),
        )

        cone_c = go.Cone(
            x=[0, c[0]],
            y=[0, c[1]],
            z=[0, c[2]],
            u=[0, c[0]],
            v=[0, c[1]],
            w=[0, c[2]],
            name="v3",
            text="v3",
            sizemode="absolute",
            sizeref=vector_size,
            # showlegend=False,
            showscale=False,
            colorscale=[[0, vector_color_list[2]], [1, vector_color_list[2]]],
        )

        return (
            [lattice_trace],
            [
                vector_a,
                cone_a,
                vector_b,
                cone_b,
                vector_c,
                cone_c,
            ],
        )

    @classmethod
    def gen_lattice_trace_2d_list(
        cls, x_coord, y_coord, lattice_label, lattice_color="rgba(0,0,0,.7)", size=15
    ):

        return [
            go.Scatter(
                x=x_coord,
                y=y_coord,
                mode="markers",
                hovertext=lattice_label,
                marker=dict(
                    size=size,
                    color=lattice_color,
                ),
            )
        ]

    @classmethod
    def gen_wigner_size_trace_3d_list(cls, a, b, c, color="rgba(0,0,0,.3)"):
        trace_list = []
        _, edges, facets = LatticeUtility.get_wigner_size_cell_3d(a, b, c)
        for xx in edges:
            trace_list.append(
                go.Scatter3d(
                    x=xx[:, 0],
                    y=xx[:, 1],
                    z=xx[:, 2],
                    mode="lines",
                    line=dict(color=color, width=6),
                    name="wigner",
                    showlegend=False,
                )
            )
        for facet in facets:
            trace_list.append(
                go.Mesh3d(
                    x=facet[:, 0],
                    y=facet[:, 1],
                    z=facet[:, 2],
                    color=color,
                    # showlegend=True,
                    name="wigner",
                    alphahull=-1,
                )
            )
        return trace_list

    @classmethod
    def gen_wigner_size_trace_2d_list(
        cls,
        a,
        b,
        c,
        point_on_the_plane,
        norm_vector,
        parallel_new_ex,
        width,
        color="rgba(0,0,0,.3)",
    ):
        from scipy.spatial import ConvexHull

        x, y, z = LatticeUtility.get_wigner_size_cell_2d(
            a, b, c, point_on_the_plane, norm_vector, width
        )

        x, y = get_plane_coordinates(norm_vector, parallel_new_ex, x, y, z)

        points_2d = np.column_stack((x, y))
        hull = ConvexHull(points_2d)  # 计算凸包

        # 提取凸包的顶点，形成封闭的多边形路径
        x_hull = points_2d[hull.vertices, 0]
        y_hull = points_2d[hull.vertices, 1]

        trace_lst = []

        trace_lst.append(
            go.Scatter(
                x=np.append(x_hull, x_hull[0]),  # 闭合多边形
                y=np.append(y_hull, y_hull[0]),
                mode="lines",
                fill="toself",
                line=dict(
                    color=color,
                ),
            )
        )

        return trace_lst

    @classmethod
    def gen_high_symmetry_points_trace_3d_list(
        cls, x, y, z, label, color="rgb(255,0,0)"
    ):
        return [
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                hovertext=label,
                text=label,
                mode="markers+text",
                marker=dict(size=4, color=color, line=dict(width=2, color=color)),
                textfont=dict(size=20, color=color),
                name="HSP",
            )
        ]

    @classmethod
    def gen_detector_trace_3d_list(
        cls, E_i, u, v, phi_range, theta_range, psi_range, E_spacing
    ):
        trace_list = []
        for dE in np.arange(0, E_i, E_spacing):
            detector_X, detector_Y, detector_Z = (
                LatticeUtility.get_detector_coverage_coordinates(
                    E_i,
                    dE,
                    u,
                    v,
                    phi_range,
                    theta_range,
                    psi_range,
                )
            )
            trace_list += [
                go.Scatter3d(
                    z=detector_Z,
                    x=detector_X,
                    y=detector_Y,
                    opacity=0.1,
                    mode="markers",
                    marker=dict(size=4, color="#00BFFF"),
                    name="detectors",
                )
            ]
        return trace_list

    @classmethod
    def gen_detector_trace_2d_list(
        cls,
        E_i,
        u,
        v,
        phi_range,
        theta_range,
        psi_range,
        E_spacing,
        point_on_the_plane,
        norm_vector,
        parallel_new_ex,
        width,
    ):
        trace_list = []
        for dE in np.arange(0, E_i, E_spacing):
            detector_X, detector_Y, detector_Z = (
                LatticeUtility.get_detector_coverage_coordinates(
                    E_i,
                    dE,
                    u,
                    v,
                    phi_range,
                    theta_range,
                    psi_range,
                )
            )

            detector_X, detector_Y = plane_judge_and_to_2D_cor(
                point_on_the_plane=point_on_the_plane,
                norm_vector=norm_vector,
                parallel_new_ex=parallel_new_ex,
                width=width,
                data_x=detector_X,
                data_y=detector_Y,
                data_z=detector_Z,
            )

            trace_list += [
                go.Scatter(
                    x=detector_X,
                    y=detector_Y,
                    opacity=0.3,
                    mode="markers",
                    marker=dict(size=10, color="#00BFFF"),
                    name="detectors",
                )
            ]
        return trace_list

    @classmethod
    def gen_sphere_wire_trace_3d_list(
        cls,
        *radius,
        color="red",
    ):

        mode = mode.lower()

        traces = []
        angle_spacing = 20
        for r in radius:
            for phi in np.arange(0, 360, angle_spacing) / 180 * np.pi:
                if phi == 0:
                    is_showlegend = True
                else:
                    is_showlegend = False
                theta_arr = np.arange(0, 180, 1) / 180 * np.pi

                X = r * np.sin(theta_arr) * np.cos(phi)
                Y = r * np.sin(theta_arr) * np.sin(phi)
                Z = r * np.cos(theta_arr)

                traces += [
                    go.Scatter3d(
                        x=X,
                        y=Y,
                        z=Z,
                        mode="lines",
                        line=dict(color=color, dash="dash"),
                        name=f"{r}",
                        legendgroup=f"r={r}",
                        showlegend=is_showlegend,
                    )
                ]

            for theta in np.arange(angle_spacing, 180, angle_spacing) / 180 * np.pi:
                if phi == angle_spacing:
                    is_showlegend = True
                else:
                    is_showlegend = False
                phi_arr = np.arange(0, 360, 1) / 180 * np.pi

                X = r * np.sin(theta) * np.cos(phi_arr)
                Y = r * np.sin(theta) * np.sin(phi_arr)
                Z = r * np.cos(theta) * np.ones_like(X)

                traces += [
                    go.Scatter3d(
                        x=X,
                        y=Y,
                        z=Z,
                        mode="lines",
                        line=dict(color=color, dash="dash"),
                        name=f"{r}",
                        legendgroup=f"r={r}",
                        showlegend=is_showlegend,
                    )
                ]

        return traces

    def plot_3d_kspace(
        self,
        conv_N_list,
        is_plot_high_symmetry_points=False,
        is_plot_conv_vector=True,
        is_plot_pri_vector=False,
        is_plot_detectors=False,
        is_plot_magnetic_peaks=False,
        is_plot_Al_powder=False,
        is_plot_Cu_powder=False,
    ):
        if len(conv_N_list) != 3:
            raise ValueError("The length of N_lists must be 3! ")

        (
            klatt_x_conv,
            klatt_y_conv,
            klatt_z_conv,
            label_conv,
            klatt_x_pri,
            klatt_y_pri,
            klatt_z_pri,
            label_pri,
        ) = self.get_plotting_Brag_lattice_coordinates(*conv_N_list)

        # conventional lattice plot
        # klatt_x_conv, klatt_y_conv, klatt_z_conv, label_conv = (
        #     LatticeUtility.unit_cell_3d(
        #         self.a_star,
        #         self.b_star,
        #         self.c_star,
        #         [[0, 0, 0]],
        #         conv_N_list[0],
        #         conv_N_list[1],
        #         conv_N_list[2],
        #     )
        # )

        lattice_trace_conv, vector_list_conv = LatticePLots.gen_lattice_trace_3d_list(
            self.a_star,
            self.b_star,
            self.c_star,
            klatt_x_conv,
            klatt_y_conv,
            klatt_z_conv,
            label_conv,
            lattice_color="rgba(0,139,0,.1)",
            vector_size=0.01,
        )

        # primitive lattice plot

        # klatt_x_pri, klatt_y_pri, klatt_z_pri, label_pri = LatticeUtility.unit_cell_3d(
        #     self.b1,
        #     self.b2,
        #     self.b3,
        #     [[0, 0, 0]],
        #     pri_N_list[0],
        #     pri_N_list[1],
        #     pri_N_list[2],
        # )

        lattice_trace_pri, vector_list_pri = LatticePLots.gen_lattice_trace_3d_list(
            self.b1, self.b2, self.b3, klatt_x_pri, klatt_y_pri, klatt_z_pri, label_pri
        )

        # magnetic Bragg peaks plot
        if is_plot_magnetic_peaks:
            magn_x = magn_y = magn_z = np.array([])
            for magnetic_modulation in self.magnetic_modulation_vector_list:
                x, y, z = magnetic_modulation
                magn_x = np.hstack((magn_x, klatt_x_pri + x))
                magn_y = np.hstack((magn_y, klatt_y_pri + y))
                magn_z = np.hstack((magn_z, klatt_z_pri + z))

            if self.magnetic_constrain_function != None:
                constrain_idx = np.array(
                    [
                        self.magnetic_constrain_function(
                            magn_x[i], magn_y[i], magn_z[i]
                        )
                        for i in range(len(magn_x))
                    ]
                )

                magn_x = magn_x[constrain_idx]
                magn_y = magn_y[constrain_idx]
                magn_z = magn_z[constrain_idx]

            magn_h, magn_k, magn_l = self.xyz_to_hkl(magn_x, magn_y, magn_z)
            magn_label = [
                f"{magn_h[i]:.3g},{magn_k[i]:.3g},{magn_l[i]:.3g}"
                for i in range(len(magn_h))
            ]

            magnetic_trace = [
                go.Scatter3d(
                    x=magn_x,
                    y=magn_y,
                    z=magn_z,
                    hovertext=magn_label,
                    mode="markers",
                    marker=dict(
                        size=4,
                        color="rgba(255,0,127,0.1)",
                        line=dict(width=2, color="rgba(255,0,127,0.1)"),
                    ),
                    name="magnetic",
                    # textfont=dict(size=20, color="rgba(255,0,127)"),
                )
            ]
        # 1BZ plot
        bz_trace = LatticePLots.gen_wigner_size_trace_3d_list(self.b1, self.b2, self.b3)

        # high symmetry points trace
        if is_plot_high_symmetry_points:

            high_symmetry_kpoints_trace = (
                LatticePLots.gen_high_symmetry_points_trace_3d_list(
                    self.high_symmetry_points_coordinates[0],
                    self.high_symmetry_points_coordinates[1],
                    self.high_symmetry_points_coordinates[2],
                    self.high_symmetry_points_coordinates_label,
                )
            )

        # powder plots
        # Al
        if is_plot_Al_powder:
            al_powder_trace_list = LatticePLots.gen_sphere_wire_trace_3d_list(
                *LatticePLots.al_powder["q_list"], color=LatticePLots.al_powder["color"]
            )
        # Cu
        if is_plot_Cu_powder:
            cu_powder_trace_list = LatticePLots.gen_sphere_wire_trace_3d_list(
                *LatticePLots.cu_powder["q_list"], color=LatticePLots.cu_powder["color"]
            )

        # detectors trace
        if is_plot_detectors:
            detector_trace = LatticePLots.gen_detector_trace_3d_list(
                self.E_i,
                self.u,
                self.v,
                self.phi_range_list,
                self.theta_range_list,
                self.psi_range,
                self.E_spacing,
            )

        # plot traces
        plot_traces = []

        if is_plot_Al_powder:
            plot_traces += al_powder_trace_list
        if is_plot_Cu_powder:
            plot_traces += cu_powder_trace_list
        if is_plot_conv_vector:
            plot_traces += vector_list_conv
        plot_traces += lattice_trace_pri
        if is_plot_pri_vector:
            plot_traces += vector_list_pri
        plot_traces += lattice_trace_conv
        plot_traces += bz_trace
        if is_plot_high_symmetry_points:
            plot_traces += high_symmetry_kpoints_trace
        if is_plot_magnetic_peaks:
            plot_traces += magnetic_trace

        # detector traces must be the last!
        if is_plot_detectors:
            plot_traces += detector_trace

        # fig settings
        camera = dict(
            # eye position, towards origin point
            eye=dict(
                x=(self.a_star + self.b_star + self.c_star)[0] / 10,
                y=(self.a_star + self.b_star + self.c_star)[1] / 10,
                z=(self.a_star + self.b_star + self.c_star)[2] / 10,
            )
        )

        layout = go.Layout(
            scene=dict(
                camera=camera,
                aspectmode="data",  # x,y,z equal
                # zaxis_range=[1.2, 2],
                # xaxis_range=[0, 10],
                # yaxis_range=[0, 10],
                # xaxis_tickmode="array",
                # yaxis_tickmode="array",
                # xaxis_tickvals=[0, 2, 4, 6, 8, 10],
                # yaxis_tickvals=[0, 2, 4, 6, 8, 10],
                # xaxis_dtick=2,
                # yaxis_dtick=2,
            ),
        )
        fig = go.Figure()

        fig.add_traces(plot_traces)
        fig.update_layout(layout)

        if is_plot_detectors:
            detector_trace_number = len(detector_trace)
            other_trace_number = len(plot_traces) - detector_trace_number

            for i in range(-detector_trace_number + 1, 0):
                fig.data[i].visible = False

            steps = []

            for i in range(detector_trace_number):
                step = dict(
                    method="update",
                    args=[
                        {
                            "visible": [True] * other_trace_number
                            + [False] * detector_trace_number
                        },
                        {
                            "title": f"Ei={self.E_i} meV ,dE={i*self.E_spacing} meV, Ef={self.E_i-i*self.E_spacing} meV"
                        },
                    ],  # layout attribute
                    label=f"{i*self.E_spacing}",
                )
                step["args"][0]["visible"][
                    other_trace_number + i
                ] = True  # Toggle i'th trace to "visible"
                steps.append(step)

            sliders = [dict(active=0, currentvalue={"prefix": f"dE:"}, steps=steps)]

            fig.update_layout(sliders=sliders)

        iplot(fig)

    def plot_detector_coverage_along(self, k_points_list, label_list, width):

        data_x = data_y = data_z = data_dE = np.array([])
        for dE in np.arange(0, self.E_i, self.E_spacing):
            detector_X, detector_Y, detector_Z = (
                LatticeUtility.get_detector_coverage_coordinates(
                    self.E_i,
                    dE,
                    self.u,
                    self.v,
                    self.phi_range_list,
                    self.theta_range_list,
                    self.psi_range,
                )
            )

            data_x = np.hstack((data_x, detector_X))
            data_y = np.hstack((data_y, detector_Y))
            data_z = np.hstack((data_z, detector_Z))
            data_dE = np.hstack((data_dE, np.array([dE] * len(detector_X))))

        fig = go.Figure()

        for i in range(len(k_points_list) - 1):

            r0 = k_points_list[i]
            r1 = k_points_list[i + 1]
            parameter_x, dE_plot = where_points_along(
                r0=r0,
                r1=r1,
                width=width,
                data_x=data_x,
                data_y=data_y,
                data_z=data_z,
                dE=data_dE,
            )

            parameter_x = parameter_x + i
            fig.add_trace(go.Scatter(x=parameter_x, y=dE_plot, mode="markers"))

        layout = go.Layout(
            scene=dict(
                aspectmode="data",  # x,y,z equal
            ),
            xaxis=dict(
                range=[-0.2, len(k_points_list) - 0.8],  # 设置 x 轴的数据范围
                autorange=False,  # 关闭自动调整数据范围
                showgrid=True,  # 显示网格
            ),
            yaxis=dict(title=r"$\Delta\text{E(meV)}$"),
        )
        fig.update_layout(layout)
        fig.update_xaxes(
            tickvals=list(range(len(k_points_list))),
            ticktext=label_list,  # 设置刻度值  # 设置刻度文本
        )

        iplot(fig)

    def plot_plane_slice(
        self,
        point_on_the_plane,
        norm_vector,
        parallel_new_ex,
        width,
        conv_N_list=[3, 3, 3],
        is_plot_high_symmetry_points=False,
        is_plot_detectors=False,
        is_plot_magnetic_peaks=False,
        is_plot_Al_powder=False,
        is_plot_Cu_powder=False,
    ):
        """
        所有矢量均不是hkl坐标
        """

        if len(conv_N_list) != 3:
            raise ValueError("The length of N_lists must be 3! ")

        (
            klatt_x_conv,
            klatt_y_conv,
            klatt_z_conv,
            label_conv,
            klatt_x_pri,
            klatt_y_pri,
            klatt_z_pri,
            label_pri,
        ) = self.get_plotting_Brag_lattice_coordinates(*conv_N_list)

        klatt_x_conv_2d, klatt_y_conv_2d, label_conv_2d = plane_judge_and_to_2D_cor(
            point_on_the_plane,
            norm_vector,
            parallel_new_ex,
            width,
            klatt_x_conv,
            klatt_y_conv,
            klatt_z_conv,
            [label_conv],
        )

        klatt_x_pri_2d, klatt_y_pri_2d, label_pri_2d = plane_judge_and_to_2D_cor(
            point_on_the_plane,
            norm_vector,
            parallel_new_ex,
            width,
            klatt_x_pri,
            klatt_y_pri,
            klatt_z_pri,
            [label_pri],
        )

        # conventional lattice plot
        lattice_trace_conv = LatticePLots.gen_lattice_trace_2d_list(
            klatt_x_conv_2d,
            klatt_y_conv_2d,
            label_conv_2d,
            lattice_color="rgba(0,255,0,.3)",
        )

        # primitive lattice plot
        lattice_trace_pri = LatticePLots.gen_lattice_trace_2d_list(
            klatt_x_pri_2d, klatt_y_pri_2d, label_pri_2d
        )

        # magnetic Bragg peaks plot
        if is_plot_magnetic_peaks:
            magn_x = magn_y = magn_z = np.array([])
            for magnetic_modulation in self.magnetic_modulation_vector_list:
                x, y, z = magnetic_modulation
                magn_x = np.hstack((magn_x, klatt_x_pri + x))
                magn_y = np.hstack((magn_y, klatt_y_pri + y))
                magn_z = np.hstack((magn_z, klatt_z_pri + z))

            if self.magnetic_constrain_function != None:
                constrain_idx = np.array(
                    [
                        self.magnetic_constrain_function(
                            magn_x[i], magn_y[i], magn_z[i]
                        )
                        for i in range(len(magn_x))
                    ]
                )

                magn_x = magn_x[constrain_idx]
                magn_y = magn_y[constrain_idx]
                magn_z = magn_z[constrain_idx]

            magn_x, magn_y, magn_z = where_points_in_plane(
                point_on_the_plane, norm_vector, width, magn_x, magn_y, magn_z
            )

            magn_h, magn_k, magn_l = self.xyz_to_hkl(magn_x, magn_y, magn_z)
            magn_label = [
                f"{magn_h[i]:.3g},{magn_k[i]:.3g},{magn_l[i]:.3g}"
                for i in range(len(magn_h))
            ]

            magn_x, magn_y = get_plane_coordinates(
                norm_vector, parallel_new_ex, magn_x, magn_y, magn_z
            )

            magnetic_trace = [
                go.Scatter(
                    x=magn_x,
                    y=magn_y,
                    # label=magn_label,
                    hovertext=magn_label,
                    mode="markers",
                    marker=dict(
                        size=10,
                        color="rgba(255,0,127,0.5)",
                        line=dict(width=2, color="rgba(255,0,127,0.5)"),
                    ),
                    name="magnetic",
                    # textfont=dict(size=20, color="rgba(255,0,127)"),
                )
            ]
        # 1BZ plot
        bz_trace = LatticePLots.gen_wigner_size_trace_2d_list(
            self.b1,
            self.b2,
            self.b3,
            point_on_the_plane=point_on_the_plane,
            norm_vector=norm_vector,
            parallel_new_ex=parallel_new_ex,
            width=width,
        )

        # # high symmetry points trace
        # if is_plot_high_symmetry_points:

        #     high_symmetry_kpoints_trace = (
        #         LatticePLots.gen_high_symmetry_points_trace_3d_list(
        #             self.high_symmetry_points_coordinates[0],
        #             self.high_symmetry_points_coordinates[1],
        #             self.high_symmetry_points_coordinates[2],
        #             self.high_symmetry_points_coordinates_label,
        #         )
        #     )

        # powder plots
        # Al
        # if is_plot_Al_powder:
        #     al_powder_trace_list = LatticePLots.gen_sphere_wire_trace_list(
        #         *LatticePLots.al_powder["q_list"],
        #         color=LatticePLots.al_powder["color"],
        #         mode="2d",
        #         point_on_the_plane=point_on_the_plane,
        #         norm_vector=norm_vector,
        #         parallel_new_ex=parallel_new_ex,
        #         width=width,
        #     )
        # # Cu
        # if is_plot_Cu_powder:
        #     cu_powder_trace_list = LatticePLots.gen_sphere_wire_trace_list(
        #         *LatticePLots.cu_powder["q_list"],
        #         color=LatticePLots.cu_powder["color"],
        #         mode="2d",
        #         point_on_the_plane=point_on_the_plane,
        #         norm_vector=norm_vector,
        #         parallel_new_ex=parallel_new_ex,
        #         width=width,
        #     )

        # detectors trace
        if is_plot_detectors:
            detector_trace = LatticePLots.gen_detector_trace_2d_list(
                self.E_i,
                self.u,
                self.v,
                self.phi_range_list,
                self.theta_range_list,
                self.psi_range,
                self.E_spacing,
                point_on_the_plane=point_on_the_plane,
                norm_vector=norm_vector,
                parallel_new_ex=parallel_new_ex,
                width=width,
            )

        # plot traces
        plot_traces = []

        # if is_plot_Al_powder:
        #     plot_traces += al_powder_trace_list
        # if is_plot_Cu_powder:
        #     plot_traces += cu_powder_trace_list

        plot_traces += lattice_trace_pri
        plot_traces += lattice_trace_conv
        plot_traces += bz_trace
        # if is_plot_high_symmetry_points:
        #     plot_traces += high_symmetry_kpoints_trace
        if is_plot_magnetic_peaks:
            plot_traces += magnetic_trace

        # detector traces must be the last!
        if is_plot_detectors:
            plot_traces += detector_trace

        # fig settings
        # camera = dict(
        #     # eye position, towards origin point
        #     eye=dict(
        #         x=(self.a_star + self.b_star + self.c_star)[0] / 10,
        #         y=(self.a_star + self.b_star + self.c_star)[1] / 10,
        #         z=(self.a_star + self.b_star + self.c_star)[2] / 10,
        #     )
        # )

        layout = go.Layout(
            scene=dict(
                # camera=camera,
                aspectmode="data",  # x,y,z equal
                # zaxis_range=[1.2, 2],
                # xaxis_range=[0, 10],
                # yaxis_range=[0, 10],
                # xaxis_tickmode="array",
                # yaxis_tickmode="array",
                # xaxis_tickvals=[0, 2, 4, 6, 8, 10],
                # yaxis_tickvals=[0, 2, 4, 6, 8, 10],
                # xaxis_dtick=2,
                # yaxis_dtick=2,
            ),
            xaxis=dict(
                scaleanchor="y",  # 锁定 x 轴比例以匹配 y 轴
                scaleratio=1,  # x 轴和 y 轴比例 1:1
            ),
            yaxis=dict(
                scaleanchor="x",  # 锁定 y 轴比例以匹配 x 轴
                scaleratio=1,  # y 轴和 x 轴比例 1:1
            ),
        )
        fig = go.Figure()

        fig.add_traces(plot_traces)
        fig.update_layout(layout)

        if is_plot_detectors:
            detector_trace_number = len(detector_trace)
            other_trace_number = len(plot_traces) - detector_trace_number

            for i in range(-detector_trace_number + 1, 0):
                fig.data[i].visible = False

            steps = []

            for i in range(detector_trace_number):
                step = dict(
                    method="update",
                    args=[
                        {
                            "visible": [True] * other_trace_number
                            + [False] * detector_trace_number
                        },
                        {
                            "title": f"Ei={self.E_i} meV ,dE={i*self.E_spacing} meV, Ef={self.E_i-i*self.E_spacing} meV"
                        },
                    ],  # layout attribute
                    label=f"{i*self.E_spacing}",
                )
                step["args"][0]["visible"][
                    other_trace_number + i
                ] = True  # Toggle i'th trace to "visible"
                steps.append(step)

            sliders = [dict(active=0, currentvalue={"prefix": f"dE:"}, steps=steps)]

            fig.update_layout(sliders=sliders)

        iplot(fig)

    def plot_detector_coverage_along(self, k_points_list, label_list, width):

        data_x = data_y = data_z = data_dE = np.array([])
        for dE in np.arange(0, self.E_i, self.E_spacing):
            detector_X, detector_Y, detector_Z = (
                LatticeUtility.get_detector_coverage_coordinates(
                    self.E_i,
                    dE,
                    self.u,
                    self.v,
                    self.phi_range_list,
                    self.theta_range_list,
                    self.psi_range,
                )
            )

            data_x = np.hstack((data_x, detector_X))
            data_y = np.hstack((data_y, detector_Y))
            data_z = np.hstack((data_z, detector_Z))
            data_dE = np.hstack((data_dE, np.array([dE] * len(detector_X))))

        fig = go.Figure()

        for i in range(len(k_points_list) - 1):

            r0 = k_points_list[i]
            r1 = k_points_list[i + 1]
            parameter_x, dE_plot = where_points_along(
                r0=r0,
                r1=r1,
                width=width,
                data_x=data_x,
                data_y=data_y,
                data_z=data_z,
                dE=data_dE,
            )

            parameter_x = parameter_x + i
            fig.add_trace(go.Scatter(x=parameter_x, y=dE_plot, mode="markers"))

        layout = go.Layout(
            scene=dict(
                aspectmode="data",  # x,y,z equal
            ),
            xaxis=dict(
                range=[-0.2, len(k_points_list) - 0.8],  # 设置 x 轴的数据范围
                autorange=False,  # 关闭自动调整数据范围
                showgrid=True,  # 显示网格
            ),
            yaxis=dict(title=r"$\Delta\text{E(meV)}$"),
        )
        fig.update_layout(layout)
        fig.update_xaxes(
            tickvals=list(range(len(k_points_list))),
            ticktext=label_list,  # 设置刻度值  # 设置刻度文本
        )

        iplot(fig)

        pass


# %%
# lattice parameters

lp = LatticePLots()
lp.set_conv_lattice(
    a_par=3.95499,
    b_par=3.95499,
    c_par=21.02424,
    alpha=90.0000,
    beta=90.0000,
    gamma=120.0000,
)
# primitive lattice vectors coordinates based on vectors a,b,c
lp.set_pri_lattice(
    a1_cor=[2 / 3, 1 / 3, 1 / 3],
    a2_cor=[-1 / 3, 1 / 3, 1 / 3],
    a3_cor=[-1 / 3, -2 / 3, 1 / 3],
)

# # detector parameters
lp.set_detector_parameters(
    E_i=15,
    u_cor=[1, 0, 0],
    v_cor=[0, 1, 0],
    phi_range=[
        [-20, 120],
    ],
    theta_range=[[-8, 8]],
    psi_range=[0, 120],
)


# hs_points, hsp_label = high_symmetry_kpoints.get_RHL1(lp.a1, lp.a2, lp.b1, lp.b2, lp.b3)

# lp.set_high_symmetry_points(
#     hs_points[:, 0], hs_points[:, 1], hs_points[:, 2], label=hsp_label
# )


q1 = lp.a_star * 0.138 + lp.c_star * 1.457
q2 = np.dot(q1, rotation_matrix(lp.c_star, pi * 2 / 3).T)
q3 = np.dot(q1, rotation_matrix(lp.c_star, pi * 4 / 3).T)
magnetic_Bragg_list = [q1, -q1, q2, -q2, q3, -q3]


def constrain(x, y, z):
    if (
        np.abs(z) < 1.2
        and np.power(x, 2) + np.power(y, 2) + np.power(z, 2)
        <= 3 * np.linalg.norm(lp.a_star) ** 2
    ):
        return True
    else:
        return False


# lp.set_magnetic_points(
#     magnetic_Bragg_list, is_coordinate=False, constrain_function=constrain
# )


# 1. plot kspace
# lp.plot_kspace(
#     conv_N_list=[3, 3, 4],
#     pri_N_list=[3, 3, 3],
#     is_plot_high_symmetry_points=True,
#     is_plot_detectors=True,
#     is_plot_magnetic_peaks=True,
#     is_plot_Al_powder=True,
#     is_plot_Cu_powder=False,
# )


# 2. plot q-E along some q points
# lp.plot_detector_coverage_along(
#     [
#         [0, 0, 0],
#         q1,
#         q2,
#         q3,
#         [-0.92, -1.59, 0.29],
#         [-0.79, -1.37, -0.1365],
#         [-0.66, -1.58, -0.16],
#         [-0.79, -1.8, -0.13],
#     ],
#     ["$\Gamma$", "q1", "q2", "q3", "G", "a1", "a2", "a3"],
#     width=np.linalg.norm(lp.a_star) / 10,
# )

# 3. print hkl list
# lp.print_hkl_list()


# lp.plot_plane_slice(
#     lp.get_hkl_vector(0, 0, 0),
#     lp.get_hkl_vector(0, 0, 1),
#     lp.get_hkl_vector(1, 0, 0),
#     lp.a_star_par / 20,
#     is_plot_detectors=True,
#     is_plot_magnetic_peaks=True,
# )
