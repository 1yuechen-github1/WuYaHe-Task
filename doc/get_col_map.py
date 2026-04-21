import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap



rgb_anchors = np.array([
        [253, 11, 0],
        [253, 39, 0],
        [253, 63, 1],
        [254, 88, 2],
        [255, 115, 0],
        [253, 143, 1],
        [255, 163, 0],
        [254, 187, 3],
        [251, 222, 1],
        [254, 237, 1],
        [244, 248, 0],
        [211, 230, 26],
        [187, 219, 0],
        [169, 209, 0],
        [140, 196, 0],
        [114, 186, 0],
        [85, 168, 0],
        [64, 159, 0],
        [36, 147, 0],
        [10, 134, 0],
        [10, 134, 0],
], dtype=float)

colors = rgb_anchors / 255.0
cmap = LinearSegmentedColormap.from_list("my_bar", colors, N=256)

fig, ax = plt.subplots(figsize=(8, 1.8))

# 0~1 gradient
gradient = np.linspace(0, 1, 512).reshape(1, -1)

# 显示 colorbar
im = ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[3,16, 0, 5])

ax.set_yticks([])

# 🔥 关键：刻度 6~8，步长 0.1
ticks = np.arange(3, 16, 0.5)
ax.set_xticks(ticks)

ax.set_xlabel("Value")

plt.tight_layout()
plt.show()