from types import NoneType
from typing import List, Optional, Tuple
import numpy as np


from dataclasses import dataclass, field


@dataclass
class DetectorConfig:
    incident_energy: float
    detector_u: np.ndarray
    detector_v: np.ndarray
    # 晶体旋转的角度
    psi_range: Tuple[float, float]
    phi_ranges: List[Tuple[float, float]]
    theta_ranges_: List[Tuple[float, float]]

    detector_w: Optional[np.ndarray] = None

    def __post_init__(self):
        # 将theta值从以u,v平面为0转化为一般的球坐标系中的theta值
        self.theta_ranges = [sorted([90 - x[0], 90 - x[1]]) for x in self.theta_ranges_]

        # 由于定义的是晶体旋转的角度，但是之后的计算中，是转动入射方向的，所以相对的，这里需要将角度全部反向
        self.psi_range = sorted([-psi for psi in self.psi_range])

        # 检查v vector的手性，是否与定义的w冲突
        if type(self.detector_w) != NoneType:
            self.detector_v = (
                self.detector_v
                if np.dot(np.cross(self.detector_u, self.detector_v), self.detector_w)
                >= 0
                else -self.detector_v
            )


@dataclass
class MAPSConfig(DetectorConfig):
    phi_ranges: List[Tuple[float, float]] = field(
        default_factory=lambda: [[3, 20], [-20, 20], [-20, 20], [-60, -3]]
    )
    theta_ranges: List[Tuple[float, float]] = field(
        default_factory=lambda: [[-7, 7], [3, 20], [-20, -3], [-7, 7]]
    )


@dataclass
class FourSEASONSConfig(DetectorConfig):

    phi_ranges: List[Tuple[float, float]] = field(default_factory=lambda: [[-35, 130]])
    theta_ranges: List[Tuple[float, float]] = field(default_factory=lambda: [[-25, 27]])
