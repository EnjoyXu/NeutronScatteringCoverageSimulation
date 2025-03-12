from numpy import array


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

        high_symmetry_kpoints_cor, k_labels = KPathLatimerMunro(
            primitive_structure
        ).get_kpoints()
        high_symmetry_kpoints_cor = array(high_symmetry_kpoints_cor)

        k_labels = get_points_labels(
            high_symmetry_kpoints_cor,
            cov_reciprocal_lattice.matrix,
            None,
            "",
            2,
            k_labels,
        )

        lattice_data = LatticeData(
            alpha=cov_lattice.alpha,
            beta=cov_lattice.beta,
            gamma=cov_lattice.gamma,
            conv_lattice_matrix=cov_lattice.matrix,
            pri_lattice_matrix=pri_lattice.matrix,
            conv_reciprocal_matrix=cov_reciprocal_lattice.matrix,
            pri_reciprocal_matrix=pri_reciprocal_lattice.matrix,
            high_symmetry_kpoints=(high_symmetry_kpoints_cor, k_labels),
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

        self.magnetic_constrain_function = constrain_function

        self.magnetic_modulation_vector_list = (
            magnetic_modulation_vector_list
            if is_hkl == False
            else [
                vector[0] * self.lattice_data.a_star
                + vector[1] * self.lattice_data.b_star
                + vector[2] * self.lattice_data.c_star
                for vector in magnetic_modulation_vector_list
            ]
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
