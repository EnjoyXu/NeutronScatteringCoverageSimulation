from pyclbr import Class
from typing import Optional
from numpy import ndarray
from crystal_toolkit.detector.detector import Detector
from crystal_toolkit.lattice.lattice import Lattice
from crystal_toolkit.visualization.composite.composite_base import CompositePlotter
from crystal_toolkit.visualization.plot_2d.bz2d import BrillouinZone2DPlotter
from crystal_toolkit.visualization.plot_2d.detector2d import Detector2DPlotter
from crystal_toolkit.visualization.plot_2d.lattice_2d import Lattice2DPlotter
from crystal_toolkit.visualization.plot_2d.magnetic_points2d import (
    MagneticPoints2DPlotter,
)
from crystal_toolkit.visualization.plot_2d.plotter_2d_base import BasePlotter2D
from crystal_toolkit.visualization.plot_3d.bz3d import BrillouinZone3DPlotter
from crystal_toolkit.visualization.plot_3d.detector3d import Detector3DPlotter
from crystal_toolkit.visualization.plot_3d.lattice3d import Lattice3DPlotter
from crystal_toolkit.visualization.plot_3d.magnetic_points3d import (
    MagneticPoints3DPlotter,
)


class KSpace3D(CompositePlotter):
    title = "3D K space"

    def __init__(self, lattice: Lattice, detector: Optional[Detector] = None):
        super().__init__()
        self._lattice = lattice
        self._detector = detector

    def plot(
        self,
        is_plot_high_symmetry_points=False,
        is_plot_conv_vector=True,
        is_plot_pri_vector=False,
        is_plot_detectors=False,
        is_plot_magnetic_peaks=False,
        is_plot_Al_powder=False,
        is_plot_Cu_powder=False,
    ):

        # 注册倒空间格点
        self.add_figure(
            Lattice3DPlotter(self._lattice).plot_3d_reciprocal_lattice(
                is_plot_conv_vector=is_plot_conv_vector,
                is_plot_pri_vector=is_plot_pri_vector,
            )
        )

        # 注册磁峰
        if is_plot_magnetic_peaks:
            self.add_figure(
                MagneticPoints3DPlotter(
                    self._lattice.magn_k_points, self._lattice.magn_label_list
                ).plot()
            )

        # 注册BZ
        self.add_figure(
            BrillouinZone3DPlotter(
                self._lattice.lattice_data.pri_reciprocal_matrix,
                self._lattice.lattice_data.conv_reciprocal_matrix,
            ).plot()
        )

        # 注册探测器
        if is_plot_detectors:
            self.add_detector_figure(
                Detector3DPlotter(self._detector).plot(),
                self._detector.title_list,
                self._detector.label_list,
            )

        # 组装
        self.build_plot(is_plot_detectors)

        # 统一layout
        self._apply_unified_layout(KSpace3D.title)
        return self.fig


class KSpace2D(CompositePlotter, BasePlotter2D):
    title = "2D K space"

    def __init__(
        self,
        lattice: Lattice,
        norm_vector: ndarray,
        plane_point: ndarray,
        thickness: float,
        parallel_new_ex: ndarray,
        detector: Optional[Detector] = None,
    ):
        CompositePlotter.__init__(
            self,
        )

        BasePlotter2D.__init__(
            self, norm_vector, plane_point, thickness, parallel_new_ex
        )

        self._lattice = lattice
        self._detector = detector

    def plot(
        self,
        is_plot_high_symmetry_points=False,
        is_plot_conv_vector=True,
        is_plot_pri_vector=False,
        is_plot_detectors=False,
        is_plot_magnetic_peaks=False,
        is_plot_Al_powder=False,
        is_plot_Cu_powder=False,
    ):

        # 注册倒空间格点
        self.add_figure(
            Lattice2DPlotter(
                self._lattice,
                self.norm_vector,
                self.plane_point,
                self.thickness,
                self.parallel_new_ex,
            ).plot_2d_reciprocal_lattice(
                is_plot_conv_vector=is_plot_conv_vector,
                is_plot_pri_vector=is_plot_pri_vector,
            )
        )

        # 注册磁峰
        if is_plot_magnetic_peaks:
            self.add_figure(
                MagneticPoints2DPlotter(
                    self._lattice.magn_k_points,
                    self._lattice.magn_label_list,
                    self.norm_vector,
                    self.plane_point,
                    self.thickness,
                    self.parallel_new_ex,
                ).plot()
            )

        # 注册BZ
        self.add_figure(
            BrillouinZone2DPlotter(
                self._lattice.lattice_data.pri_reciprocal_matrix,
                self._lattice.lattice_data.conv_reciprocal_matrix,
                self.norm_vector,
                self.plane_point,
                self.thickness,
                self.parallel_new_ex,
            ).plot()
        )

        # 注册探测器
        if is_plot_detectors:
            self.add_detector_figure(
                Detector2DPlotter(
                    self._detector,
                    self.norm_vector,
                    self.plane_point,
                    self.thickness,
                    self.parallel_new_ex,
                ).plot(),
                self._detector.title_list,
                self._detector.label_list,
            )

        # 组装
        self.build_plot(is_plot_detectors)

        self._apply_unified_layout(KSpace2D.title)

        return self.fig
