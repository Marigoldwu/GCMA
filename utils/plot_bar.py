import numpy as np
import matplotlib.pyplot as plt

categories = ['ACC', 'NMI', 'ARI', 'F1']
# values = [[86.46, 58.98, 64.04, 86.55],
#           [90.35, 67.40, 73.54, 90.34],
#           [92.00, 71.63, 77.70, 92.00],
#           [92.74, 73.60, 79.60, 92.75]]
# values = [[56.69, 29.45, 28.78, 54.19],
#           [69.50, 43.53, 44.65, 63.74],
#           [71.56, 45.61, 47.69, 65.61],
#           [71.78, 46.39, 48.13, 65.73]]
# values = [[51.17, 21.99, 21.09, 50.19],
#           [80.55, 49.93, 55.90, 79.94],
#           [82.87, 53.93, 60.48, 82.26],
#           [83.49, 55.13, 61.70, 82.94]]
# values = [[43.03, 12.78, 11.66, 40.85],
#           [46.22, 14.69, 13.63, 41.88],
#           [56.92, 27.29, 25.62, 56.77],
#           [57.34, 28.59, 26.62, 57.33]]
values = [[75.29, 61.54, 55.83, 68.52],
          [75.21, 62.53, 53.78, 71.92],
          [80.77, 68.65, 61.41, 79.61],
          [81.45, 69.77, 62.69, 79.97]]
bar_width = 0.15

bar_positions = np.arange(len(categories))

colors = plt.cm.get_cmap('GnBu', len(values[0])+1)
label = ["G", "T", "T+P", "Ours"]

for i in range(len(values[0])):
    offset = (i-0.5) * bar_width
    plt.bar(bar_positions + offset, [row[i] for row in np.array(values).T/100], width=bar_width, label=f'{label[i]}', color=colors(i+1))

# plt.xlabel('Metric', fontproperties='Times New Roman', fontsize=16, fontweight='bold')
# plt.ylabel('Value', fontproperties='Times New Roman', fontsize=16, fontweight='bold')
plt.xticks(bar_positions + bar_width, categories, fontproperties='Times New Roman', fontsize=14, fontweight='normal')
plt.yticks(fontproperties='Times New Roman', fontsize=14, fontweight='normal')
font_legend = {'family': 'Times New Roman',
               'weight': 'normal',
               'size': 14,
               }
plt.legend(loc="upper center", ncol=2, prop=font_legend)

# plt.ylim(0.55, 0.95)
# plt.ylim(0.25, 0.75)
# plt.ylim(0.20, 0.85)
# plt.ylim(0.10, 0.60)
plt.ylim(0.50, 0.85)

# plt.show()
plt.savefig("ablation_amap.pdf", dpi=300)
