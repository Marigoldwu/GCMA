import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

model_names = ['GAE', 'SDCN', 'DFCN', 'DCRN', 'AGC-DRR', 'CCGC', 'HSAN', 'Ours']
face_colors = ["steelblue", "coral", "forestgreen", "grey", "darkorchid", "sienna", "violet", "crimson"]
markers = ['.', 'o', 's', '<', '>', '^', 'p', '*']
indicators = ['ACC', 'NMI', 'ARI', 'F1', '1-Time', '1-Params', '1-AIT', 'ACC']

data = np.array([[86.02, 55.86, 62.53, 86.05, 3.60, 0.48, 0.07, 86.02],
                 [89.72, 66.90, 72.10, 89.76, 7.89, 6.62, 0.16, 89.72],
                 [83.11, 54.13, 56.90, 83.30, 120.26, 0.48, 2.41, 83.11],
                 [88.99, 65.86, 70.36, 89.04, 130.90, 0.60, 2.62, 88.99],
                 [91.70, 70.36, 76.82, 91.73, 150.56, 1.60, 3.01, 91.70],
                 [89.06, 64.48, 70.46, 89.02, 4.52, 1.87, 0.13, 89.06],
                 [82.51, 51.81, 55.34, 82.46, 22.79, 10.58, 0.46, 82.51],
                 [92.76, 73.71, 79.66, 92.77, 7.96, 1.09, 0.16, 92.76]])

min_value = np.min(data, axis=0)
max_value = np.max(data, axis=0)
norm_data = (data - min_value) / (max_value - min_value)
for i in range(0, 8):
    for j in range(4, 7):
        norm_data[i][j] = 1 - norm_data[i][j]

angles = np.linspace(0, 2 * np.pi, len(indicators) - 1, endpoint=False)
angles = (angles + np.pi / 14).tolist()
angles.append(angles[0])
values = norm_data.tolist()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)
font_legend = {'family': 'Times New Roman',
               'weight': 'normal',
               'size': 14,
               }
for i in range(len(model_names)):
    face_color = face_colors[i]
    marker = markers[i]
    line, = ax.plot(angles, values[i], marker=marker, linewidth=1, label=model_names[i], color=face_color)
    ax.fill(angles, values[i], facecolor=face_color, alpha=0.1)
    ax.set_xticks(angles)
    font = FontProperties(family='Times New Roman', size=15)
    ax.set_xticklabels(indicators, fontproperties=font)
    ax.tick_params(axis='x', pad=10)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0], fontname='Times New Roman', fontsize=14)

font_legend = {'family': 'Times New Roman',
               'weight': 'normal',
               'size': 14,
               }
plt.legend(bbox_to_anchor=(0., 1.04, 1., .104), loc='lower left', prop=font_legend,
           ncol=4, mode="expand", borderaxespad=0.)

# plt.show()
plt.savefig("performance.pdf", dpi=300)
