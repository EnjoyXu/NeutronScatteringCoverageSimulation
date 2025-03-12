from dataclasses import dataclass, field

from numpy import ndarray
from numpy.linalg import norm


@dataclass
class LatticeData:
    # 基本晶格参数
    alpha: float
    beta: float
    gamma: float

    # 单胞基矢
    conv_lattice_matrix: ndarray

    # 变换后基矢
    pri_lattice_matrix: ndarray

    # 倒易空间基矢
    conv_reciprocal_matrix: ndarray

    # 布里渊区边界
    pri_reciprocal_matrix: ndarray

    # 高对称点
    high_symmetry_kpoints: tuple

    # 以下为自动计算的派生属性
    a: ndarray = field(init=False)
    b: ndarray = field(init=False)
    c: ndarray = field(init=False)

    a_par: float = field(init=False)
    b_par: float = field(init=False)
    c_par: float = field(init=False)

    a1: ndarray = field(init=False)
    a2: ndarray = field(init=False)
    a3: ndarray = field(init=False)

    a1_par: float = field(init=False)
    a2_par: float = field(init=False)
    a3_par: float = field(init=False)

    a_star: ndarray = field(init=False)
    b_star: ndarray = field(init=False)
    c_star: ndarray = field(init=False)

    a_star_par: float = field(init=False)
    b_star_par: float = field(init=False)
    c_star_par: float = field(init=False)

    b1: ndarray = field(init=False)
    b2: ndarray = field(init=False)
    b3: ndarray = field(init=False)

    b1_par: float = field(init=False)
    b2_par: float = field(init=False)
    b3_par: float = field(init=False)

    def __post_init__(self):
        """自动计算所有模长参数"""
        self.a = self.conv_lattice_matrix[0]
        self.b = self.conv_lattice_matrix[1]
        self.c = self.conv_lattice_matrix[2]

        self.a1 = self.pri_lattice_matrix[0]
        self.a2 = self.pri_lattice_matrix[1]
        self.a3 = self.pri_lattice_matrix[2]

        self.a_star = self.conv_reciprocal_matrix[0]
        self.b_star = self.conv_reciprocal_matrix[1]
        self.c_star = self.conv_reciprocal_matrix[2]

        self.b1 = self.pri_reciprocal_matrix[0]
        self.b2 = self.pri_reciprocal_matrix[1]
        self.b3 = self.pri_reciprocal_matrix[2]

        # 原始基矢模长
        self.a_par = norm(self.a)
        self.b_par = norm(self.b)
        self.c_par = norm(self.c)

        # 变换基矢模长
        self.a1_par = norm(self.a1)
        self.a2_par = norm(self.a2)
        self.a3_par = norm(self.a3)

        # 倒易空间模长
        self.a_star_par = norm(self.a_star)
        self.b_star_par = norm(self.b_star)
        self.c_star_par = norm(self.c_star)

        # 边界模长
        self.b1_par = norm(self.b1)
        self.b2_par = norm(self.b2)
        self.b3_par = norm(self.b3)
