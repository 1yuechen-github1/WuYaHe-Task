import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# 1️⃣ 自定义颜色映射：红 -> 黄 -> 绿
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'red_yellow_green',
    ['red', 'yellow', 'green'],
)

# 2️⃣ 数值范围
vmin, vmax = 6, 8
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

# 3️⃣ 每 0.2 mm 生成刻度
ticks = np.arange(vmin, vmax + 0.001, 0.1)

# 4️⃣ 打印每个刻度对应的颜色（RGBA）
rgba_list = []
for t in ticks:
    rgba = cmap(norm(t))  # t → 归一化 → RGBA
    rgba = np.round(np.array(rgba) * 255).astype(int)
    print(f'{t:.1f} 对应颜色 RGBA: {rgba}')
    rgba_list.append(rgba)
# 5️⃣ 创建画布和 colorbar
fig, ax = plt.subplots(figsize=(6, 1), layout='constrained')

cb = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=ax,
    orientation='horizontal',
    label='Some Units'
)

cb.set_ticks(ticks)
cb.set_ticklabels([f'{t:.1f}' for t in ticks])

plt.show()
