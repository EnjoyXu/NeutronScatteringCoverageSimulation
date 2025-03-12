from crystal_toolkit.lattice.lattice import Lattice
from crystal_toolkit.visualization.plot_2d.plotter_2d_base import BasePlotter2D
import plotly.graph_objs as go
from numpy import ndarray


class Lattice2DPlotter(BasePlotter2D):
    def __init__(
        self,
        lattice: Lattice,
        norm_vector: ndarray,
        plane_point: ndarray,
        thickness: float,
        parallel_new_ex: ndarray,
    ):
        super().__init__(norm_vector, plane_point, thickness, parallel_new_ex)
        self.lattice = lattice

        self.conv_k_points, _, self.conv_k_labels = self._get_plane_slice(
            self.lattice.conv_k_points, self.lattice.conv_k_labels
        )
        self.pri_k_points, _, self.pri_k_labels = self._get_plane_slice(
            self.lattice.pri_k_points, self.lattice.pri_k_labels
        )

    def plot(
        self,
    ) -> go.Figure:
        """绘制单胞晶格基矢和原胞,以及初级原胞矢量"""

        # 绘制单胞基矢
        self._plot_crystal_vectors(
            self.lattice.lattice_data.conv_lattice_matrix,
            ["a", "b", "c"],
            self.config.colors["conv_vector_color_list"],
        )
        pos, label = self.lattice.get_conv_lattice_positions(1, 1, 1)
        # 绘制单胞原子
        self._plot_atoms(pos, label, self.config.colors["conv_lat_color"])
        # 绘制初级原胞基矢
        self._plot_crystal_vectors(
            self.lattice.lattice_data.pri_lattice_matrix,
            ["a1", "a2", "a3"],
            self.config.colors["pri_vector_color_list"],
        )

        self._apply_layout("Lattice")
        return self.fig

    # def _plot_crystal_vectors(self, vectors_list, name_list, colors_list) -> None:
    #     """绘制晶格基矢（含箭头）"""
    #     for vec, color, name in zip(vectors_list, colors_list, name_list):
    #         # 矢量线
    #         self.add_trace(
    #             go.Scatter(
    #                 x=[0, vec[0]],
    #                 y=[0, vec[1]],
    #                 mode="lines",
    #                 marker=dict(
    #                     size=0,
    #                 ),
    #                 line=dict(color=color, width=self.config.widths["vector"]),
    #                 name=name,
    #                 text=name,
    #                 legendgroup=name,
    #             )
    #         )
    #         # 矢量箭头
    #         self.add_trace(
    #             go.Cone(
    #                 x=[0, vec[0]],
    #                 y=[0, vec[1]],
    #                 z=[0, vec[2]],
    #                 u=[0, vec[0]],
    #                 v=[0, vec[1]],
    #                 w=[0, vec[2]],
    #                 showscale=False,
    #                 colorscale=[[0, color], [1, color]],
    #                 sizemode="absolute",
    #                 sizeref=self.config.sizes["cone"],
    #                 name=name,
    #                 text=name,
    #                 legendgroup=name,
    #             )
    #         )

    def _plot_atoms(self, atom_positions, atom_labels, color) -> None:
        """绘制晶格原子点阵"""

        self.add_trace(
            go.Scatter(
                x=atom_positions[:, 0],
                y=atom_positions[:, 1],
                mode="markers",
                marker=dict(
                    size=self.config.sizes["atom_2d"],
                    color=color,
                    # line=dict(width=self.config.widths["atom_marker"], color=color),
                ),
                hovertext=atom_labels,
            )
        )

    def plot_2d_reciprocal_lattice(
        self,
        is_plot_conv_vector=True,
        is_plot_pri_vector=False,
    ):

        # # 绘制单胞基矢
        # if is_plot_conv_vector:
        #     self._plot_crystal_vectors(
        #         self.lattice.lattice_data.conv_reciprocal_matrix,
        #         ["a_star", "b_star", "c_star"],
        #         self.config.colors["conv_vector_color_list"],
        #     )
        # # 绘制初级原胞基矢
        # if is_plot_pri_vector:
        #     self._plot_crystal_vectors(
        #         self.lattice.lattice_data.pri_reciprocal_matrix,
        #         ["b1", "b2", "b3"],
        #         self.config.colors["pri_vector_color_list"],
        #     )
        # 绘制单胞原子
        self._plot_atoms(
            self.conv_k_points,
            self.conv_k_labels,
            self.config.colors["conv_lat_2d_color"],
        )

        # 绘制初级原胞原子
        self._plot_atoms(
            self.pri_k_points,
            self.pri_k_labels,
            self.config.colors["pri_lat_2d_color"],
        )

        self._apply_layout("2d recirpocal lattice")
        return self.fig
