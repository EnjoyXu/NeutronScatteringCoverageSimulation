from numpy import ndarray, median, array, argwhere, split
from numpy.linalg import norm

# import numpy as np


# def get_high_symmetry_path_list(hsp_points: np.ndarray, hsp_label_list: np.ndarray):
#     """将连续的高对称路径分割为多个线段

#     Args:
#         hsp_points: 高对称点坐标数组，形状为(N,3)
#         hsp_label_list: 对应点的标签数组，形状为(N,)

#     Returns:
#         tuple: (分割后的坐标数组列表，分割后的标签数组列表)
#     """
#     # 计算相邻点间距（向量化替代循环）
#     deltas = np.diff(hsp_points, axis=0)
#     distances = norm(deltas, axis=1)

#     # 动态计算分割阈值（添加最小距离保护）
#     # min_dist = np.finfo(hsp_points.dtype).eps * 10
#     median_dist = np.median(distances)

#     threshold = 3 * median_dist

#     # 获取分割点位置（修正索引偏移）
#     split_indices = np.where(distances > threshold)[0] + 1

#     # 去重并排序分割点（增强鲁棒性）
#     split_indices = np.unique(np.sort(split_indices))

#     # 执行分割操作
#     points_segments = (
#         np.split(hsp_points, split_indices, axis=0)
#         if split_indices.size > 0
#         else [hsp_points]
#     )
#     labels_segments = (
#         np.split(hsp_label_list, split_indices, axis=0)
#         if split_indices.size > 0
#         else [hsp_label_list]
#     )

#     # 过滤空线段（添加最小点数约束）
#     min_points = 2
#     valid_mask = [len(seg) >= min_points for seg in points_segments]
#     return (
#         [seg for seg, valid in zip(points_segments, valid_mask) if valid],
#         [seg for seg, valid in zip(labels_segments, valid_mask) if valid],
#     )


def get_high_symmetry_path_list(hsp_points: ndarray, hsp_label_list: ndarray):
    """由于从pymatgen得到的高对称点包含路径上的点，这样不方便直接画图，故将其分成不同的Line来画"""

    dif = array(
        [norm(hsp_points[i - 1] - hsp_points[i]) for i in range(1, len(hsp_points))]
    )

    dif_median = median(dif)

    edge_idx = argwhere(dif > dif_median * 3)

    # 如果中间没有断点
    if len(edge_idx) == 0:
        return [hsp_points], [hsp_label_list]

    edge_idx = edge_idx[0]

    points_split = split(hsp_points, edge_idx, axis=0)
    labels_split = split(hsp_label_list, edge_idx, axis=0)

    mask_list = [label != "" for label in labels_split]

    return [points_split[i][mask_list[i]] for i in range(len(points_split))], [
        labels_split[i][mask_list[i]] for i in range(len(labels_split))
    ]
