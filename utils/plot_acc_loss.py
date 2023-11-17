# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.font_manager import FontProperties

font = FontProperties(family='Times New Roman', size=12)
datasets = ["acm", "cite", "dblp", "amap"]
# datasets = ["acm"]

for dataset_name in datasets:
    acc_path = f"./performance/acc/DGAE/{dataset_name}.npy"
    loss_path = f"./performance/loss/DGAE/{dataset_name}.npy"
    acc = np.load(acc_path)
    loss = np.load(loss_path)
    x = [i for i in range(1, len(acc)+1)]

    f1 = interp1d(x, acc, kind='linear')
    f2 = interp1d(x, loss, kind='linear')
    x_new = np.linspace(min(x), max(x), num=150)
    if dataset_name == "amap":
        x_new = np.linspace(min(x), max(x), num=300)
    acc_smooth = f1(x_new)
    loss_smooth = f2(x_new)

    fig, ax1 = plt.subplots()

    # 绘制第一个 y 轴的曲线
    ax1.plot(x_new, acc_smooth, color='crimson', label="Accuracy", marker='^', markevery=30, markersize=5)
    ax1.set_xlabel('Iterations', fontproperties='Times New Roman', fontsize=15)
    ax1.set_ylabel('Accuracy', fontproperties='Times New Roman', fontsize=15)
    ax1.tick_params('y', colors='crimson')
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)

    # 创建第二个 y 轴并绘制曲线
    ax2 = ax1.twinx()
    ax2.plot(x_new, loss_smooth, 'dodgerblue', label="Loss", marker='s', markevery=30, markersize=5)
    ax2.set_ylabel('Loss', fontproperties='Times New Roman', fontsize=15)
    ax2.tick_params('y', colors='dodgerblue')
    plt.yticks(fontproperties='Times New Roman', size=14)
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines = lines1 + lines2
    labels = labels1 + labels2
    font_legend = {'family': 'Times New Roman',
                   'weight': 'normal',
                   'size': 12,
                   }
    ax1.legend(lines, labels, loc='center right', prop=font_legend)

    # plt.show()
    plt.savefig(f"./acc_loss_{dataset_name}.pdf", dpi=300)
