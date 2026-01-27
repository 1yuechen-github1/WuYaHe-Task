import json
import os
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
#
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.colors import LinearSegmentedColormap
#
# plt.rcParams['font.sans-serif'] = ['SimHei']   # 黑体
# plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题
#
# def read_data(path, filename):
#     data = {}
#     with open(path, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue  # 跳过空行
#             key, value = line.split(',')
#             if key[0:3] == filename:
#                 data[key] = int(value)
#     max_n = np.max(list(data.values()))
#     return data,max_n
#
# def get_data(path):
#     data = []
#     for file in os.listdir(path):
#         if file.endswith(".jpg"):
#            filename = file[0:3]
#            data.append(filename)
#     data = list(dict.fromkeys(data))
#     # print(data)
#     return data
#
# def draw_col(b_path):
#     path = os.path.join(b_path,'img')
#     path_col = os.path.join(b_path,'col')
#     path_vis= os.path.join(b_path,'png')
#     os.makedirs(path_col,exist_ok=True)
#     os.makedirs(path_vis,exist_ok=True)
#     # os.makedirs(path, exist_ok=True)
#     data1 = get_data(path) # 文件名字的list
#     for filename in data1:
#         # 定义颜色锚点
#         colors = ['green', 'red']
#         data, max_n = read_data(path +'\\'+ 'len.txt', filename)  # 骨高度的map
#         n_bins = len(data) # 颜色数量
#         cmap_name = 'red_green'
#         # 创建自定义色彩映射
#         cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
#         values = np.linspace(0, 1, n_bins)
#         rgba_colors = cm(values)
#         rgb_colors = rgba_colors[:, :3]
#         print(rgb_colors)
#         print(len(rgb_colors))
#         with open(os.path.join(path_col, filename+'.txt'), 'w') as f:
#             f.write(str(rgb_colors))
#     #         # 创建数据
#         x = np.linspace(0, max_n, max_n * 10)
#         y = x * 0.3
#         # 绘图
#         plt.figure(figsize=(10, 6))
#         scatter = plt.scatter(x, y, c=y, cmap=cm)
#         plt.colorbar(scatter)
#         plt.title(f'{filename} 骨高度映射图', )
#         save_path = os.path.join(path_vis, f'{filename}.png')
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         plt.close()
#
#
# draw_col(r'C:\yuechen\code\wuyahe\1.code\2.data-缩放\screenshot\pca-sum')




def color_by_z_height(pcd, colormap='viridis', normalize=True):
    """
    根据Z轴高度为点云着色（类似CloudCompare）

    参数：
    pcd: open3d.geometry.PointCloud
    colormap: 颜色映射，可选 'viridis', 'plasma', 'rainbow', 'jet', 'hot', 'cool' 等
    normalize: 是否归一化Z值到[0,1]
    """
    points = np.asarray(pcd.points)
    z_values = points[:, 2]  # Z坐标

    # 归一化到[0,1]
    if normalize:
        z_min, z_max = z_values.min(), z_values.max()
        if z_max > z_min:
            normalized_z = (z_values - z_min) / (z_max - z_min)
        else:
            normalized_z = np.zeros_like(z_values)
    else:
        normalized_z = z_values

    # 获取颜色映射
    cmap = plt.get_cmap(colormap)
    colors = cmap(normalized_z)[:, :3]  # 取RGB，忽略Alpha


    # 应用颜色到点云
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print('colors:', colors)
    # 打印高度信息
    print(f"Z轴范围: [{z_min:.3f}, {z_max:.3f}]")
    print(f"Z轴中值: {np.median(z_values):.3f}")

    return pcd, z_min, z_max


xhg = np.loadtxt(r'C:\yuechen\code\wuyahe\1.code\1022\1020\output\001\001_step1.txt')
# 使用示例
# pcd = o3d.io.read_point_cloud("your_point_cloud.ply")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xhg)

# 方法1：使用默认颜色映射
colored_pcd, z_min, z_max = color_by_z_height(pcd, colormap='rainbow')

# 可视化
o3d.visualization.draw_geometries([colored_pcd],
                                  window_name=f"Z轴高度着色 (范围: {z_min:.2f} - {z_max:.2f})")