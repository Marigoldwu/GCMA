# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


dataset_name = "dblp"
label = np.load(f"dataset/{dataset_name}/{dataset_name}_label.npy")
num_nodes = label.shape[0]
print(num_nodes)

similarity_matrix = np.zeros((num_nodes, num_nodes))
sort_label = np.sort(label)

for i in range(num_nodes):
    for j in range(num_nodes):
        if sort_label[i] == sort_label[j]:
            similarity_matrix[i, j] = 1

plt.imshow(similarity_matrix, cmap='GnBu')
plt.axis("off")
plt.colorbar()

# plt.show()
plt.savefig(f"{dataset_name}_ground_truth.pdf", dpi=300)

