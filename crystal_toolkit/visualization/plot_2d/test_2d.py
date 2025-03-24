import numpy as np
from plotly.offline import iplot
import plotly.graph_objs as go

from crystal_toolkit.detector.detector import Detector
from crystal_toolkit.detector.detector_config import MAPSConfig
from crystal_toolkit.lattice.lattice import Lattice
from crystal_toolkit.visualization.composite.hybrid_plotters import KSpace3D
from crystal_toolkit.visualization.plot_2d.alignment2d import Alignment2DPlotter
from crystal_toolkit.visualization.plot_2d.bz2d import BrillouinZone2DPlotter
from crystal_toolkit.visualization.plot_2d.detector2d import Detector2DPlotter
from crystal_toolkit.visualization.plot_2d.lattice_2d import Lattice2DPlotter
from crystal_toolkit.visualization.plot_3d.lattice3d import Lattice3DPlotter

from crystal_toolkit.math_utils.math_utils import coordinate_transform, rotation_matrix

lattice = Lattice.from_cif(
    "/Users/joy/Library/CloudStorage/OneDrive-南京大学/Edu/DataAnalysis/NiI2/NiI2.cif",
    [3, 1, 3],
)
# lattice = Lattice.from_cif(
#     "/Users/joy/Library/CloudStorage/OneDrive-南京大学/Edu/DataAnalysis/MnTe/MnTe.cif",
#     [1, 1, 1],
# )


norm = lattice.get_hkl_vector(-1, 0, 0)
plane_point = lattice.get_hkl_vector(0, 0, 0)

thick = lattice.lattice_data.a_star_par / 20
parallel = lattice.get_hkl_vector(1, 0, 0)


new_u = coordinate_transform(
    lattice.get_hkl_vector(-1, 0, 0),
    rotation_matrix(lattice.get_hkl_vector(0, 0, 1), 40 / 180 * np.pi),
    is_positive=True,
)

# new_u = lattice.get_hkl_vector(-1, 0, 0)

maps = MAPSConfig(
    20.0,
    new_u,
    lattice.get_hkl_vector(0, -1, 0),
    psi_range=[0.0, 10.0],
    detector_w=lattice.get_hkl_vector(0, 0, 1),
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

iplot(Alignment2DPlotter(detector, lattice).plot())


def gen_sphere(new_u, color="rgba(255,0,0,0.5)"):
    # --- 计算球面参数 ---
    r = np.linalg.norm(new_u)  # 矢量的模长（半径）
    center = -new_u  # 球心坐标 [-2, -3, -4]

    # 生成球面网格点
    theta = np.linspace(0, np.pi, 50)  # 极角范围 [0, π]
    phi = np.linspace(0, 2 * np.pi, 50)  # 方位角范围 [0, 2π]
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    # 球面参数方程转换为笛卡尔坐标
    x = r * np.sin(theta_grid) * np.cos(phi_grid) + center[0]
    y = r * np.sin(theta_grid) * np.sin(phi_grid) + center[1]
    z = r * np.cos(theta_grid) + center[2]

    # --- 绘制半透明球面 ---
    fig = go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=0.5,  # 设置透明度为50%
        colorscale=[
            [0, color],
            [1, color],
        ],  # 纯红色半透明
        showscale=False,  # 隐藏颜色条
    )

    return fig


fig = KSpace3D(lattice, detector).plot()
fig.add_trace(gen_sphere(maps.detector_u * 21.9778))
fig.add_trace(
    gen_sphere(maps.detector_u * 0.98, "rgba(0,255,0,0.5)"),
)

fig.update_layout(
    scene=dict(
        aspectmode="data",
    ),
    overwrite=True,
)
iplot(fig)
# np.linalg.norm(lattice.get_hkl_vector(-1,1,0)+detector.config.detector_u*0.98)

# np.linalg.norm(lattice.get_hkl_vector(-1,1,0)+detector.config.detector_u*21.9778)
