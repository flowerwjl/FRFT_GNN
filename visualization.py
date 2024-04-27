from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx
# 使用PyG内置数据集
from torch_geometric.datasets import KarateClub


def visualize_graph(G, color):
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False, node_color=color, cmap='Set2')
    plt.show()


if __name__ == '__main__':
    dataset = KarateClub()
    data = dataset[0]
    G = to_networkx(data, to_undirected=True)
    visualize_graph(G, color=data.y)
