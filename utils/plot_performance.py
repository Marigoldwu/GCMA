# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

model_name_list = ["GAE", "SDCN", "DFCN", "DCRN", "AGCDRR", "CCGC", "HSAN", "DGAE"]
markers = ['.', 'o', 's', '<', '>', '^', 'p', '*']
colors = ["steelblue", "coral", "forestgreen", "grey", "darkorchid", "sienna", "violet", "crimson"]
plt_acc = plt.figure(figsize=(1, 1), dpi=500)
for i in range(len(model_name_list)):
    model_name = model_name_list[i]
    acc_path = f"./performance/acc/{model_name}.npy"
    # loss_path = f"./performance/loss/{model_name}.npy"
    acc = np.load(acc_path)
    if model_name == "CCGC":
        x = [i for i in range(1, len(acc) + 1)]
        f = interp1d(x, acc, kind='linear')
        x_new = np.linspace(min(x), max(x), num=102)
        y_smooth = f(x_new)
    else:
        x = [i for i in range(1, 51)]
        f = interp1d(x, acc, kind='linear')
        x_new = np.linspace(min(x), max(x), num=150)
        y_smooth = f(x_new)
    line, = plt.plot(x_new, y_smooth, label=model_name, color=colors[i], marker=markers[i], markevery=10, markersize=4)
    if model_name == "DGAE":
        line.set_label("Ours")

font_legend = {'family': 'Times New Roman',
               'weight': 'normal',
               'size': 8,
               }

plt.xlabel("Iterations", fontproperties='Times New Roman', fontsize=15)
plt.ylabel("Accuracy", fontproperties='Times New Roman', fontsize=15)

plt.yticks(fontproperties='Times New Roman', size=12)
plt.xticks(fontproperties='Times New Roman', size=12)
plt.legend(loc="lower right", prop=font_legend)
# plt.show()
plt.savefig("./performance/acc/acc_1000.pdf")
