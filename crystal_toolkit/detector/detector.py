from narwhals import Boolean
from traitlets import Bool
from crystal_toolkit.detector.detector_config import DetectorConfig
from crystal_toolkit.math_utils.geometry import get_perp_plane, get_plane_line_cross
from crystal_toolkit.math_utils.math_utils import (
    coordinate_transform,
    coordinate_transform_from_xyz_to_sphere,
    normalize_vector,
    rotation_matrix,
    wrap_to_interval,
)
from typing import List, Tuple
import numpy as np


class Detector(object):
    def __init__(
        self, detector_config: DetectorConfig, slice_number=10, angle_step=2.0
    ):
        self.config = detector_config

        self.dE = np.linspace(
            0, self.config.incident_energy, slice_number, endpoint=False
        )

        # python能不能像kotlin一样用到的时候再去加载？
        self.detector_points_list = [
            get_detector_coordinates(
                incident_energy=self.config.incident_energy,
                energy_loss=energy_loss,
                detector_u=self.config.detector_u,
                detector_v=self.config.detector_v,
                phi_ranges=self.config.phi_ranges,
                theta_ranges=self.config.theta_ranges,
                psi_range=sorted(
                    [-psi for psi in self.config.psi_range]
                ),  # 由于定义的是晶体旋转的角度，但是之后的计算中，是转动入射方向的，所以相对的，这里需要将角度全部反向
                angle_step=angle_step,
            )
            for energy_loss in self.dE
        ]

        self.title_list = [
            f"Ei={self.config.incident_energy:.1f} meV, dE={energy_loss:.1f} meV, Ef = {self.config.incident_energy-energy_loss:.1f}"
            for energy_loss in self.dE
        ]

        self.label_list = [f"{energy_loss:.1f}:" for energy_loss in self.dE]

    # # TODO add from .par
    # @staticmethod
    # def get_from_par(par_file_path: str):
    #     pass

    def get_available_points_coordinates_white_beam(
        self,
        points: np.ndarray,
        k_min: float,
        k_max: float,
    ):
        """判断点是否能被n=1级Bragg散射探测到,并返回能探测到的点的探测中子vec(kf)的球坐标以及探测点的原始坐标"""

        # 首先查看点是否在k_min,k_max包围之内
        points_filter = self._filter_distance(points, k_min, k_max)

        # 点与(0,0,0)的中垂面
        normal_plane_par_list = [
            get_perp_plane([0, 0, 0], point) for point in points_filter
        ]

        # 求中垂面与u_vec的交点
        cross_point = np.array(
            [
                get_plane_line_cross([0, 0, 0], self.config.detector_u, plane_par)
                for plane_par in (normal_plane_par_list)
            ]
        )

        sphere_cor = coordinate_transform_from_xyz_to_sphere(
            coordinate_transform(
                points_filter - cross_point,
                _build_detector_basis(self.config.detector_u, self.config.detector_v),
            )
        )

        mask = self._is_covered(sphere_cor[:, 1], sphere_cor[:, 2])

        return sphere_cor[mask], points_filter[mask]

    def _is_covered(self, theta_arr: np.ndarray, phi_arr: np.ndarray) -> List[Boolean]:
        """通过theta,phi判断点是否在探测器角度cover范围内,角度为弧度制"""
        theta_arr = wrap_to_interval(np.rad2deg(theta_arr), 0, 180)
        phi_arr = wrap_to_interval(np.rad2deg(phi_arr), -180, 180)
        mask = []
        for theta, phi in zip(theta_arr, phi_arr):
            for theta_range, phi_range in zip(
                self.config.theta_ranges,
                self.config.phi_ranges,
            ):

                if (theta_range[0] <= theta <= theta_range[1]) and (
                    phi_range[0] <= phi <= phi_range[1]
                ):
                    mask.append(True)
                    break
            else:
                mask.append(False)
        return mask

    def _filter_distance(self, k_points, k_min, k_max):
        """用于白光的定向,首先通过波长先确定白光能探测到的k点"""
        n = np.linalg.norm(
            k_points + (self.config.detector_u * k_min).reshape(1, 3), axis=1
        )

        mask = n >= k_min

        points_filter = k_points[mask]

        n = np.linalg.norm(
            points_filter + (self.config.detector_u * k_max).reshape(1, 3), axis=1
        )

        mask = n <= k_max

        points_filter = points_filter[mask]

        return points_filter


def get_detector_coordinates(
    incident_energy: float,
    energy_loss: float,
    detector_u: np.ndarray,
    detector_v: np.ndarray,
    phi_ranges: List[Tuple[float, float]],
    theta_ranges: List[Tuple[float, float]],
    psi_range: Tuple[float, float],
    angle_step: float = 2.0,
) -> np.ndarray:
    """
    行向量版探测器坐标计算

    参数：
        incident_energy: 入射能量 (meV)
        energy_loss: 能量损失 (meV)
        detector_u: 探测器u轴方向行向量 (1,3)
        detector_v: 探测器v轴方向行向量 (1,3)
        ...其他参数同前...

    返回：
        Nx3 的探测器坐标数组（每行为一个点）
    """
    # 输入验证
    _validate_angle_ranges(phi_ranges, theta_ranges)

    # 转换能量到波矢
    k_i, k_f = _calculate_wave_vectors(incident_energy, energy_loss)

    # 构建探测器坐标系变换矩阵
    basis_matrix = _build_detector_basis(detector_u, detector_v)  # 3x3

    # 生成球坐标系采样点（每行为一个点）
    sphere_points = _generate_sphere_points(
        phi_ranges, theta_ranges, k_f, angle_step
    )  # Nx3

    # 坐标系变换
    detector_points = (
        coordinate_transform(
            sphere_points,
            new_basis=np.identity(3),
            old_basis=basis_matrix,
        )
        - k_i * basis_matrix[0]
    )  # 行向量运算

    # 绕法线轴旋转
    return _apply_psi_rotation(detector_points, basis_matrix[2], psi_range, angle_step)


def _validate_angle_ranges(
    phi_ranges: List[Tuple[float, float]], theta_ranges: List[Tuple[float, float]]
) -> None:
    """验证角度范围输入有效性"""
    if len(phi_ranges) != len(theta_ranges):
        raise ValueError("Phi和Theta范围列表长度必须相同")

    for (phi_min, phi_max), (theta_min, theta_max) in zip(phi_ranges, theta_ranges):
        if phi_min >= phi_max or theta_min >= theta_max:
            raise ValueError("角度范围必须满足 min < max")


def _calculate_wave_vectors(E_i: float, dE: float) -> Tuple[float, float]:
    """计算入射和散射波矢 (Å⁻¹)"""
    return (0.695 * np.sqrt(E_i), 0.695 * np.sqrt(max(E_i - dE, 1e-6)))  # 防止负能量


def _build_detector_basis(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """构建正交探测器坐标系矩阵 (3x3)"""
    u_unit = normalize_vector(u)
    v_proj = v - u_unit * np.dot(v, u_unit)  # Smith正交
    v_unit = normalize_vector(v_proj)
    w_unit = np.cross(u_unit, v_unit)
    return np.vstack((u_unit, v_unit, w_unit))  # 基向量按行存储


def _generate_sphere_points(
    phi_ranges: List[Tuple[float, float]],
    theta_ranges: List[Tuple[float, float]],
    radius: float,
    step_deg: float,
) -> np.ndarray:
    """生成球坐标点 (Nx3 行向量)"""
    points_list = []

    for (phi_min, phi_max), (theta_min, theta_max) in zip(phi_ranges, theta_ranges):
        # 生成角度网格
        phi = np.deg2rad(np.arange(phi_min, phi_max, step_deg))
        theta = np.deg2rad(np.arange(theta_min, theta_max, step_deg))

        # 创建网格并向量化计算
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        x = radius * np.sin(theta_grid) * np.cos(phi_grid)
        y = radius * np.sin(theta_grid) * np.sin(phi_grid)
        z = radius * np.cos(theta_grid)

        # 按行存储点
        points_list.append(np.column_stack((x.ravel(), y.ravel(), z.ravel())))

    return np.vstack(points_list)


def _apply_psi_rotation(
    points: np.ndarray,
    rotation_axis: np.ndarray,
    psi_range: Tuple[float, float],
    step_deg: float,
) -> np.ndarray:
    """应用绕轴旋转（行向量版）"""
    psi_angles = np.deg2rad(np.arange(psi_range[0], psi_range[1], step_deg))
    rotation_mats = [rotation_matrix(rotation_axis, angle) for angle in psi_angles]

    # 批量旋转并拼接结果
    return np.vstack(
        [coordinate_transform(points, R, is_positive=True) for R in rotation_mats]
    )
