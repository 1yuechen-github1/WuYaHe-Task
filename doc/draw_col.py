import matplotlib
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

path = r'C:\yuechen\code\wuyahe\1.code\1112\1110\data\002_left.txt'
# pcd = o3d.io.read_point_cloud(path)
data = np.loadtxt(path)  # (N, 3)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data)
pts = np.asarray(pcd.points)          # (N, 3)
s = pts[:, 2]                         # 渚嬶細鐢?z 浣滀负鏍囬噺

s_min, s_max = s.min(), s.max()
s_norm = (s - s_min) / (s_max - s_min + 1e-12)

# rgb = cmap(s_norm)[:, :3]             # RGBA -> RGB, 鑼冨洿[0,1]
cmap = matplotlib.colormaps.get_cmap("viridis")
rgb = cmap(s_norm)[:, :3]

pcd.colors = o3d.utility.Vector3dVector(rgb)
o3d.visualization.draw_geometries([pcd])