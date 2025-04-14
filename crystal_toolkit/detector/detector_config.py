from types import NoneType
from typing import List, Optional, Tuple
import numpy as np


from dataclasses import dataclass, field

from crystal_toolkit.math_utils.math_utils import normalize_vector


@dataclass
class DetectorConfig:
    incident_energy: float
    detector_u: np.ndarray
    detector_v: np.ndarray
    # 晶体旋转的角度
    psi_range: Tuple[float, float]
    phi_ranges: List[Tuple[float, float]]
    theta_ranges_direct: List[Tuple[float, float]]

    detector_w: Optional[np.ndarray] = None
    is_parallel = True  # 兼容u与ki反平行的情况

    def __post_init__(self):
        # 将theta值从以u,v平面为0转化为一般的球坐标系中的theta值
        self.theta_ranges = [
            sorted([90 - x[0], 90 - x[1]]) for x in self.theta_ranges_direct
        ]

        if self.is_parallel == False:
            # u 反平行
            self.detector_u *= -1
            # 由于v决定手性，为保持手性不变，v也必须反号
            self.detector_v *= -1

        # 检查v vector的手性，是否与定义的w冲突
        if type(self.detector_w) != NoneType:
            self.detector_v = (
                self.detector_v
                if np.dot(np.cross(self.detector_u, self.detector_v), self.detector_w)
                >= 0
                else -self.detector_v
            )
        # 归一化uv
        self.detector_u = normalize_vector(self.detector_u)
        self.detector_v = normalize_vector(self.detector_v)
        self.detector_w = normalize_vector(np.cross(self.detector_u, self.detector_v))


@dataclass
class MAPSConfig(DetectorConfig):
    phi_ranges: List[Tuple[float, float]] = field(
        default_factory=lambda: [[3, 20], [-20, 20], [-20, 20], [-60, -20], [-20, -3]]
    )
    theta_ranges_direct: List[Tuple[float, float]] = field(
        default_factory=lambda: [[-3, 3], [3, 20], [-20, -3], [-7, 7], [-3, 3]]
    )


@dataclass
class LETConfig(DetectorConfig):
    phi_ranges: List[Tuple[float, float]] = field(default_factory=lambda: [[-40, 140]])
    theta_ranges_direct: List[Tuple[float, float]] = field(
        default_factory=lambda: [[-30, 30]]
    )


@dataclass
class FourSEASONSConfig(DetectorConfig):

    phi_ranges: List[Tuple[float, float]] = field(default_factory=lambda: [[-35, 130]])
    theta_ranges_direct: List[Tuple[float, float]] = field(
        default_factory=lambda: [[-25, 27]]
    )
