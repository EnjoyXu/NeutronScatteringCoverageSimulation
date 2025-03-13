from crystal_toolkit.lattice.lattice import Lattice
from crystal_toolkit.visualization.plot_3d.bz3d import BrillouinZone3DPlotter
from crystal_toolkit.visualization.plot_3d.lattice3d import Lattice3DPlotter
from plotly.offline import iplot
import numpy as np

from crystal_toolkit.visualization.plot_3d.magnetic_points3d import (
    MagneticPoints3DPlotter,
)

lattice = Lattice.from_cif(
    "/Users/joy/Library/CloudStorage/OneDrive-南京大学/Edu/DataAnalysis/NiI2/NiI2.cif",
    [3, 3, 3],
)

lattice.set_magnetic_points([[0.138, 0, 1.457]])
# lp = Lattice3DPlotter(
#     lattice,
# )


# lp.lattice.lattice_data.pri_lattice_matrix = np.dot(
#     np.array([[1, 0, -1], [0, 1, -1], [0, 0, -1]]),
#     lp.lattice.lattice_data.pri_lattice_matrix,
# )


# iplot(lp.plot())

# bz = BrillouinZone3DPlotter(lp.lattice.lattice_data.pri_reciprocal_matrix)

# iplot(bz.plot())

# iplot(lp.plot_3d_reciprocal_lattice())

iplot(MagneticPoints3DPlotter(lattice.magn_k_points, lattice.magn_label_list).plot())
