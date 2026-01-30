# import matplotlib.pyplot as plt
# import numpy as np
#
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator
#
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#
# # Make data.
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
#
# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
#
# # Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
#
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# # 定义颜色锚点
# colors = ['red', 'green']
# n_bins = 100  # 颜色数量
# cmap_name = 'red_green'
#
# # 创建自定义色彩映射
# cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
#
# # 创建数据
# x = np.linspace(0, 10, 100)
# y = np.sin(x)
#
# # 绘图
# plt.figure(figsize=(10, 6))
# scatter = plt.scatter(x, y, c=y, cmap=cm)
# plt.colorbar(scatter)
# plt.title('How2matplotlib.com - Custom Red-Green Colormap')
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

# n_lines = 21
# cmap = mpl.colormaps['plasma']
#
# # Take colors at regular intervals spanning the colormap.
# colors = cmap(np.linspace(0, 1, n_lines))
#
# fig, ax = plt.subplots(layout='constrained')
#
# for i, color in enumerate(colors):
#     ax.plot([0, i], color=color)
# plt.show()


import numpy as np

n_lines = 21  # 你要多少条线
colors = []

for t in np.linspace(0, 1, n_lines):
    if t < 0.5:
        # 红 -> 黄
        r = 1.0
        g = t * 2
        b = 0.0
    else:
        # 黄 -> 绿
        r = 2 - t * 2
        g = 1.0
        b = 0.0

    colors.append((r, g, b))


# colors = [
#     (255/255, 0/255, 0/255),     # 红
#     (255/255, 255/255, 0/255),   # 黄
#     (0/255, 255/255, 0/255),     # 绿
# ]

fig, ax = plt.subplots()

for i, color in enumerate(colors):
    ax.plot([0, i+1], color=color, linewidth=3)

plt.show()
