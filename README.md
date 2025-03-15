# NeutronScatteringCoverageSimulation

中子散射的模拟程序。目前还在开发中。



## 快速开始 Quick start 

### 安装 Installation 

下载解压crystal_toolkit即可。目前还没计划上传到PyPI。



### 使用 Usage

在解压的同一目录下创建python脚本，导入模块`from crystal_toolkit import *`



#### 例子 Example

##### 三维K空间 3D K space

```python
from crystal_toolkit import *

#从cif文件创建Lattice
lattice = Lattice.from_cif(
    "cif path",
    reciprocal_lattice_N_list = [3, 3, 5], # -3<=H<=3 , -3<=K<=3,-5<=L<=5
)

lattice.set_magnetic_points([[0.5,0,0]]) #设置磁峰，出现位置为G+q, G-q

u = lattice.get_hkl_vector(0, 0, 1) # u的方向就是入射中子的方向
v = lattice.get_hkl_vector(1, 0, 0) # v与u一起决定散射平面
w = lattice.get_hkl_vector(-1, 2, 0) # w为旋转轴，默认为cross(u,v)

incident_energy = 20 #单位为meV
psi = [0,180] #psi最大值与最小值
#phi与theta的范围，两个列表一一对应，将探测器范围cover全
phi = [[-20,30],[30,50]] 
theta = [[-10,10],[-7,7]]

slice_number = 10 #dE从0到Ei共画多少张图
angle_step = 2 #张成探测器coverage时的角度步长

detector_config = DetectorConfig(
    incident_energy=incident_energy,
    detector_u=u,
    detector_v=v,
  	detector_w=w,
    theta_ranges=theta
    phi_ranges=phi
    psi_range= psi,
)


detector = Detector(detector_config, slice_number=slice_number, angle_step=angle_step)

iplot(
    KSpace3D(lattice, detector).plot(
        is_plot_detectors=True, is_plot_magnetic_peaks=True
    )
)

```



##### 二维K空间 2D K space



```python
from crystal_toolkit import *

#从cif文件创建Lattice
lattice = Lattice.from_cif(
    "cif path",
    reciprocal_lattice_N_list = [3, 3, 5], # -3<=H<=3 , -3<=K<=3,-5<=L<=5
)

lattice.set_magnetic_points([[0.5,0,0]]) #设置磁峰，出现位置为G+q, G-q

u = lattice.get_hkl_vector(0, 0, 1) # u的方向就是入射中子的方向
v = lattice.get_hkl_vector(1, 0, 0) # v与u一起决定散射平面
w = lattice.get_hkl_vector(-1, 2, 0) # w为旋转轴，默认为cross(u,v)

incident_energy = 20 #单位为meV
psi = [0,180] #psi最大值与最小值
#phi与theta的范围，两个列表一一对应，将探测器范围cover全
phi = [[-20,30],[30,50]] 
theta = [[-10,10],[-7,7]]

slice_number = 10 #dE从0到Ei共画多少张图
angle_step = 2 #张成探测器coverage时的角度步长

norm = lattice.get_hkl_vector(1, -2, 0) # 平面的法向量
plane_point = lattice.get_hkl_vector(0, 0, 0) # 平面上的一个点
new_ex = lattice.get_hkl_vector(1, 0, 0) # 决定2维图的x方向，该方向为new_ex在平面上的投影
thick = lattice.lattice_data.a_star_par / 20 # 平面切片的厚度，这里设置为a*的1/20

detector_config = DetectorConfig(
    incident_energy=incident_energy,
    detector_u=u,
    detector_v=v,
  	detector_w=w,
    theta_ranges=theta
    phi_ranges=phi
    psi_range= psi,
)


detector = Detector(detector_config, slice_number=slice_number, angle_step=angle_step)

iplot(
    KSpace2D(lattice, norm, plane_point, thick, new_ex, detector).plot(
        is_plot_detectors=True, is_plot_magnetic_peaks=True
    )
)


```



##### 一维Q-E覆盖图 1D Q-E coverage



```python
from crystal_toolkit import *

#从cif文件创建Lattice
lattice = Lattice.from_cif(
    "cif path",
    reciprocal_lattice_N_list = [3, 3, 5], # -3<=H<=3 , -3<=K<=3,-5<=L<=5
)

lattice.set_magnetic_points([[0.5,0,0]]) #设置磁峰，出现位置为G+q, G-q

u = lattice.get_hkl_vector(0, 0, 1) # u的方向就是入射中子的方向
v = lattice.get_hkl_vector(1, 0, 0) # v与u一起决定散射平面
w = lattice.get_hkl_vector(-1, 2, 0) # w为旋转轴，默认为cross(u,v)

incident_energy = 20 #单位为meV
psi = [0,180] #psi最大值与最小值
#phi与theta的范围，两个列表一一对应，将探测器范围cover全
phi = [[-20,30],[30,50]] 
theta = [[-10,10],[-7,7]]

slice_number = 10 #dE从0到Ei共画多少张图
angle_step = 2 #张成探测器coverage时的角度步长

# 定义需要cover的q点
q_points = [
    lattice.get_hkl_vector(0, 0, 0),
    lattice.get_hkl_vector(-1, 0, 2),
    lattice.get_hkl_vector(-1, 0, -1),
]

width = lattice.lattice_data.a_star_par / 20 #设置投影的线的宽度


detector_config = DetectorConfig(
    incident_energy=incident_energy,
    detector_u=u,
    detector_v=v,
  	detector_w=w,
    theta_ranges=theta
    phi_ranges=phi
    psi_range= psi,
)


detector = Detector(detector_config, slice_number=slice_number, angle_step=angle_step)

iplot(
    Detector1DPlotter(
        detector,
        q_points,
        width,
        lattice.lattice_data.conv_reciprocal_matrix,
    ).plot()
)

```

