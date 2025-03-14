from numpy import array, vstack, unique


from crystal_toolkit.lattice.lattice_data import LatticeData
from crystal_toolkit.lattice.lattice_utils import (
    generate_conv_and_pri_lattice_3d_coordinates,
    generate_lattice_coordinates,
)
from crystal_toolkit.math_utils.math_utils import get_points_labels


class Lattice(object):
    def __init__(
        self,
        lattice_data: LatticeData,
        reciprocal_lattice_N_list=[3, 3, 3],
        lattice_N_list=[3, 3, 3],
    ):
        self.lattice_data = lattice_data
        self._define_3d_k_points(*reciprocal_lattice_N_list)

    def _define_3d_k_points(self, Nx, Ny, Nz):
        self.conv_k_points, self.conv_k_labels, self.pri_k_points, self.pri_k_labels = (
            generate_conv_and_pri_lattice_3d_coordinates(
                Nx,
                Ny,
                Nz,
                self.lattice_data.conv_reciprocal_matrix,
                self.lattice_data.pri_reciprocal_matrix,
            )
        )

    @classmethod
    def from_cif(
        cls,
        cif_path,
        reciprocal_lattice_N_list=[3, 3, 3],
        lattice_N_list=[3, 3, 3],
    ):
        """
        直接从cif文件读取晶格参数.
        """
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

        high_symmetry_kpoints_cor, high_symmetry_kpoints_labels = KPathLatimerMunro(
            primitive_structure
        ).get_kpoints()
        high_symmetry_kpoints_cor = array(high_symmetry_kpoints_cor)

        high_symmetry_kpoints_labels = get_points_labels(
            high_symmetry_kpoints_cor,
            cov_reciprocal_lattice.matrix,
            None,
            "",
            2,
            high_symmetry_kpoints_labels,
        )[0]

        lattice_data = LatticeData(
            alpha=cov_lattice.alpha,
            beta=cov_lattice.beta,
            gamma=cov_lattice.gamma,
            conv_lattice_matrix=cov_lattice.matrix,
            pri_lattice_matrix=pri_lattice.matrix,
            conv_reciprocal_matrix=cov_reciprocal_lattice.matrix,
            pri_reciprocal_matrix=pri_reciprocal_lattice.matrix,
            high_symmetry_kpoints=high_symmetry_kpoints_cor,
            high_symmetry_kpoints_label=high_symmetry_kpoints_labels,
        )

        return Lattice(
            lattice_data,
            reciprocal_lattice_N_list=reciprocal_lattice_N_list,
            lattice_N_list=lattice_N_list,
        )

    @classmethod
    def from_parameter(cls):
        # TODO
        pass

    def set_magnetic_points(
        self,
        magnetic_modulation_vector_list,
        constrain_function=None,
        is_hkl=True,
    ):
        """
        默认是以a_star,b_star,c_star为基的坐标,若直接输入绝对位置,可将is_hkl设置为False。若要对磁峰画图范围有限制,传入constrain_function(x,y,z),该函数应当输出True/False。
        """

        self._magnetic_constrain_function = constrain_function

        vector_array = (
            array(magnetic_modulation_vector_list)
            if is_hkl == False
            else array(
                [
                    vector[0] * self.lattice_data.a_star
                    + vector[1] * self.lattice_data.b_star
                    + vector[2] * self.lattice_data.c_star
                    for vector in magnetic_modulation_vector_list
                ]
            )
        )

        # 增加对应的-q点
        vector_array = vstack(
            (
                vector_array,
                vector_array * -1,
            )
        )

        # 防止已经添加过-q了，所以去重
        self._magnetic_modulation_vector_array = unique(vector_array, axis=0)

        self._define_magnetic_points()

    def _define_magnetic_points(self):
        # 使用列表推导式批量生成坐标点
        self.magn_k_points = vstack(
            [
                magn_modulation + self.pri_k_points
                for magn_modulation in self._magnetic_modulation_vector_array
            ]
        )

        # 应用约束条件
        if self._magnetic_constrain_function is not None:
            mask = [
                self._magnetic_constrain_function(point[0], point[1], point[2])
                for point in self.magn_k_points
            ]
            self.magn_k_points = self.magn_k_points[mask]

        # 坐标转换和标签生成
        self.magn_label_list = get_points_labels(
            self.magn_k_points, new_basis=self.lattice_data.conv_reciprocal_matrix
        )

    def get_hkl_vector(self, h, k, l):
        return (
            h * self.lattice_data.a_star
            + k * self.lattice_data.b_star
            + l * self.lattice_data.c_star
        )

    def get_conv_lattice_positions(self, Nx, Ny, Nz):
        return generate_lattice_coordinates(
            self.lattice_data.a, self.lattice_data.b, self.lattice_data.c, Nx, Ny, Nz
        )

    def get_pri_lattice_positions(self, Nx, Ny, Nz):
        return generate_lattice_coordinates(
            self.lattice_data.a1, self.lattice_data.a2, self.lattice_data.a3, Nx, Ny, Nz
        )

    def get_conv_reciprocal_positions(self, Nx, Ny, Nz):
        return generate_lattice_coordinates(
            self.lattice_data.a_star,
            self.lattice_data.b_star,
            self.lattice_data.c_star,
            Nx,
            Ny,
            Nz,
        )

    def get_pri_reciprocal_positions(self, Nx, Ny, Nz):
        return generate_lattice_coordinates(
            self.lattice_data.b1, self.lattice_data.b2, self.lattice_data.b3, Nx, Ny, Nz
        )
