import torch
from torch_geometric.utils import degree, to_dense_adj, get_laplacian, to_scipy_sparse_matrix, add_self_loops, add_remaining_self_loops
import numpy as np
import scipy
from dataset_loader import DataLoader
import os


def get_L(edge_index: torch.Tensor, num_nodes=None, is_remove_self_loops=False):

    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    if is_remove_self_loops:
        """
        以下返回「去除掉自环的图」的拉普拉斯矩阵
        """

        l_edge, l_w = get_laplacian(edge_index, normalization='sym')  # 返回的是「去除掉自环的图」的拉普拉斯矩阵
        laplacian = to_scipy_sparse_matrix(l_edge, l_w)  # 将edge_index转换为COO格式的稀疏矩阵
        laplacian = laplacian.toarray()         # 将稀疏矩阵转换为numpy数组
        laplacian = torch.tensor(laplacian)     # 将numpy数组转换为torch张量

        # eigenvalue, _ = scipy.sparse.linalg.eigsh(laplacian, k=1, which="LM")   # 求解稀疏矩阵的特征值

    else:
        """
        以下返回原图的拉普拉斯矩阵
        """

        # 计算邻接矩阵
        A = to_dense_adj(edge_index)[0]

        # 计算节点度数
        row, col = edge_index
        deg = degree(row)

        # 计算度矩阵
        # D = torch.diag(deg)
        # D_half = torch.diag(deg ** 0.5)

        # 计算正则化度矩阵
        D_half_neg = torch.diag(deg ** -0.5)

        # 计算正则化邻接矩阵
        P = D_half_neg @ A @ D_half_neg

        # 计算正则化图拉普拉斯矩阵
        I = torch.eye(num_nodes)
        laplacian = I - P

    return laplacian


def get_P_tilde(edge_index: torch.Tensor, num_nodes=None):

    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    I = torch.eye(num_nodes)
    L_tilde = get_L_tilde(edge_index, num_nodes=num_nodes)

    P_tilde = I - L_tilde

    return P_tilde


def get_L_tilde(edge_index: torch.Tensor, num_nodes=None):

    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    edge_index, _ = add_remaining_self_loops(edge_index=edge_index, num_nodes=num_nodes)
    L_tilde = get_L(edge_index, num_nodes=num_nodes, is_remove_self_loops=False)
    return L_tilde


def get_L_P_tilde_alpha(edge_index: torch.Tensor, num_nodes=None, power=0.5, return_np=False):

    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    L_tilde = get_L_tilde(edge_index=edge_index, num_nodes=num_nodes)
    e_vals, e_vecs = eigen_decomposition(L_tilde)

    # e_vals = e_vals ** power
    e_vecs = scipy.linalg.fractional_matrix_power(e_vecs, power)
    e_vecs_H = np.conj(e_vecs.T)

    L_tilde_alpha = e_vecs @ np.diag(e_vals) @ e_vecs_H

    I = np.eye(num_nodes)
    P_tilde_alpha = I - L_tilde_alpha

    if return_np:
        return L_tilde_alpha, P_tilde_alpha

    return torch.tensor(L_tilde_alpha, dtype=torch.complex128), torch.tensor(P_tilde_alpha, dtype=torch.complex128)


def eigen_decomposition(matrix):

    # 断言保证matrix是实对称矩阵
    assert np.array_equal(matrix, matrix.T)

    # np.linalg.eigh()用于计算对称矩阵的特征值和特征向量, 并返回按照升序排列的特征值。
    eigen_values, eigen_vectors = np.linalg.eigh(matrix)

    # 将小于1e-7的数值视为0
    eigen_values[abs(eigen_values) < 1e-7] = 0
    eigen_vectors[abs(eigen_vectors) < 1e-7] = 0

    # U是正交矩阵
    assert np.allclose(eigen_vectors.T, np.linalg.inv(eigen_vectors), atol=1e-5)

    return eigen_values, eigen_vectors


def save_fractional_matrix():

    for dataset_name in ['Actor']:

        dataset = DataLoader(dataset_name)
        data = dataset[0]

        # create directory if not exist
        os.makedirs(f'fractional_matrix/{dataset_name}', exist_ok=True)

        for i in [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
            L_tilde_alpha, P_tilde_alpha = get_L_P_tilde_alpha(edge_index=data.edge_index, power=i, return_np=True)
            np.save(f'fractional_matrix/{dataset_name}/L_tilde_alpha_{i}.npy', L_tilde_alpha)
            np.save(f'fractional_matrix/{dataset_name}/P_tilde_alpha_{i}.npy', P_tilde_alpha)
            print(f'{dataset_name} fractional order {i} matrix saved.')


def load_fractional_matrix(dataset_name: str, power: float):

    L_tilde_alpha = np.load(f'fractional_matrix/{dataset_name}/L_tilde_alpha_{power}.npy')
    P_tilde_alpha = np.load(f'fractional_matrix/{dataset_name}/P_tilde_alpha_{power}.npy')

    return torch.tensor(L_tilde_alpha, dtype=torch.complex128), torch.tensor(P_tilde_alpha, dtype=torch.complex128)


def print_dataset_info(dataset):
    print()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')


def print_data_info(data):
    print()
    print(data)
    print('======================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def random_splits(data, num_classes, percls_trn, val_lb, seed=42):
    index = [i for i in range(0, data.y.shape[0])]
    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx) < percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn, replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
    test_idx = [i for i in rest_index if i not in val_idx]

    data.train_mask = index_to_mask(train_idx, size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx, size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx, size=data.num_nodes)
    return data


if __name__ == '__main__':
    save_fractional_matrix()

    # np.load('fractional_matrix/Texas/L_tilde_alpha_0.2.npy')
