from typing import List, Tuple
import numpy as np


from dataclasses import dataclass, field


@dataclass
class DetectorConfig:
    incident_energy: float
    detector_u: np.ndarray
    detector_v: np.ndarray
    psi_range: Tuple[float, float]
    phi_ranges: List[Tuple[float, float]]
    theta_ranges: List[Tuple[float, float]]

    def __post_init__(self):
        # 将theta值从以u,v平面为0转化为一般的球坐标系中的theta值
        self.theta_ranges = [sorted([90 - x[0], 90 - x[1]]) for x in self.theta_ranges]


@dataclass
class MAPSConfig(DetectorConfig):
    phi_ranges: List[Tuple[float, float]] = field(
        default_factory=lambda: [(-20.0, 20.0), (20.0, 60.0)]
    )
    theta_ranges: List[Tuple[float, float]] = field(
        default_factory=lambda: [(-20.0, 20.0), (-7.0, 7.0)]
    )


@dataclass
class FourSEASONSConfig(DetectorConfig):
    phi_ranges = [
        [-35, 130],
    ]
    theta_ranges = [[-25, 27]]
