from plotly.offline import iplot

from crystal_toolkit.detector.detector import Detector
from crystal_toolkit.detector.detector_config import MAPSConfig
from crystal_toolkit.lattice.lattice import Lattice
from crystal_toolkit.visualization.plot_2d.bz2d import BrillouinZone2DPlotter
from crystal_toolkit.visualization.plot_2d.detector2d import Detector2DPlotter
from crystal_toolkit.visualization.plot_2d.lattice_2d import Lattice2DPlotter

lattice = Lattice.from_cif(
    "/Users/joy/Library/CloudStorage/OneDrive-南京大学/Edu/DataAnalysis/NiI2/NiI2.cif",
    [3, 3, 5],
)

norm = lattice.get_hkl_vector(0, 0, 1)
plane_point = lattice.get_hkl_vector(0, 0, 0)

thick = lattice.lattice_data.a_star_par / 20
parallel = lattice.get_hkl_vector(1, 0, 0)

maps = MAPSConfig(
    20.0,
    lattice.get_hkl_vector(1, 0, 0),
    lattice.get_hkl_vector(0, 1, 0),
    psi_range=[0.0, 10.0],
)


detector = Detector(maps, slice_number=2)


# iplot(
#     Lattice2DPlotter(
#         lattice, norm, plane_point, thick, parallel
#     ).plot_2d_reciprocal_lattice()
# )

# iplot(
#     BrillouinZone2DPlotter(
#         lattice.lattice_data.pri_reciprocal_matrix,
#         lattice.lattice_data.conv_reciprocal_matrix,
#         norm,
#         plane_point,
#         thick,
#         parallel,
#     ).plot()
# )

# iplot(
#     Detector2DPlotter(
#         detector,
#         norm,
#         plane_point,
#         thick,
#         parallel,
#     ).plot()
# )
