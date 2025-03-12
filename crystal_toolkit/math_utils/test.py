from crystal_toolkit.math_utils.geometry import *
from crystal_toolkit.math_utils.math_utils import get_points_labels

# %% test for points along line
p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

start = np.array([0, 0, -1])

end = np.array([0, 0, 5])

width = 0.1

label = ["0,0,0", "1,0,0", "0,1,0", "0,0,1"]
label2 = ["000", "100", "010", "001"]


# print(points_along_line(p, start, end, width))

# print(points_along_line(p, start, end, width, label, label2))


# %%

p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

plane_point = np.array([0, 0, 0])

norm = np.array([0, 0, 1])

width = 0.1

label = ["0,0,0", "1,0,0", "0,1,0", "0,0,1"]
label2 = ["000", "100", "010", "001"]


# print(points_in_plane(p, plane_point, norm, width))

# print(points_in_plane(p, plane_point, norm, width, label, label2))


# %%
p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

newbasis = np.array(
    (
        [
            2,
            0,
            0,
        ],
        [0, 1, 0],
        [0, 0, 0.5],
    )
)

print(get_points_labels(p, newbasis))

# %%
