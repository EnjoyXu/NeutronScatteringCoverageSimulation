from dataclasses import dataclass


@dataclass
class VisualConfig:

    colors = {
        "conv_vector_color_list": ["#FF0000", "#00FF00", "#0000FF"],
        "pri_vector_color_list": ["#FF0000", "#00FF00", "#0000FF"],
        "conv_lat_color": "rgba(0,139,0,.1)",
        "pri_lat_color": "rgba(0,0,0,.5)",
        "conv_lat_2d_color": "rgba(0,255,0,.3)",
        "pri_lat_2d_color": "rgba(0,0,0,.7)",
        "BZ": "rgba(0,0,0,.3)",
        "detector": "#00BFFF",
        "magnetic_points": "rgba(255,0,127,0.1)",
        "magnetic_points_2d": "rgba(255,0,127,0.5)",
        "high_symmetry_points": "rgb(255,0,0)",
    }

    widths = {
        "vector": 6.0,
        "atom_marker": 4,
        "BZ_edge": 6,
        "magnetic_points": 2,
        "high_symmetry_points": 4,
    }

    sizes = {
        "cone": 0.1,
        "atom": 4,
        "detector": 4,
        "atom_2d": 15,
        "magnetic_points": 4,
        "magnetic_points_2d": 10,
        "high_symmetry_points": 4,
    }

    opacity = {"detector_2d": 0.3}
