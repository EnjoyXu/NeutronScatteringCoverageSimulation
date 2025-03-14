from crystal_toolkit.lattice.high_symm_points import get_high_symmetry_path_list
from easyAnalysis import crystal_toolkit

from crystal_toolkit.detector.detector import Detector
from crystal_toolkit.detector.detector_config import MAPSConfig
from crystal_toolkit.lattice.lattice import Lattice
from crystal_toolkit.visualization.composite.hybrid_plotters import KSpace2D, KSpace3D

from plotly.offline import iplot

lattice = Lattice.from_cif(
    "/Users/joy/Library/CloudStorage/OneDrive-南京大学/Edu/DataAnalysis/NiI2/NiI2.cif",
    [3, 3, 5],
)

lattice.set_magnetic_points([[0.138, 0, 1.457]])

maps = MAPSConfig(
    20.0,
    lattice.get_hkl_vector(1, 0, 0),
    lattice.get_hkl_vector(0, 1, 0),
    psi_range=[0.0, 180.0],
)

norm = lattice.get_hkl_vector(0, 0, 1)
plane_point = lattice.get_hkl_vector(0, 0, 0.45)

thick = lattice.lattice_data.a_star_par / 20
parallel = lattice.get_hkl_vector(1, 0, 0)

detector = Detector(maps, slice_number=10, angle_step=2)

import plotly.graph_objs as go

# fig = KSpace3D(lattice, detector).plot()

iplot(
    KSpace3D(lattice, detector).plot(
        is_plot_detectors=True,
        is_plot_magnetic_peaks=True,
        is_plot_high_symmetry_points=True,
    )
)

# iplot(
#     KSpace2D(lattice, norm, plane_point, thick, parallel, detector).plot(
#         is_plot_detectors=True, is_plot_magnetic_peaks=True
#     )
# )
