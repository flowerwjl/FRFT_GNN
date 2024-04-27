import torch
from torch_geometric.utils import degree, to_dense_adj, get_laplacian, to_scipy_sparse_matrix, add_self_loops
import numpy as np
from scipy.linalg import fractional_matrix_power
import sympy
from sympy import Matrix
import scipy


def get_L(edge_index: torch.Tensor, num_nodes: int):

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
    L = I - P

    # l_edge, l_w = get_laplacian(edge_index, normalization='sym')    # 返回的是「去除掉自环的图」的拉普拉斯矩阵
    # laplacian = to_scipy_sparse_matrix(l_edge, l_w)  # 将edge_index转换为COO格式的稀疏矩阵
    # print(laplacian.toarray())
    # eigenvalue, _ = scipy.sparse.linalg.eigsh(laplacian, k=1, which="LM")   # 求解稀疏矩阵的特征值

    return L


def get_P_tilde(edge_index: torch.Tensor, num_nodes: int):
    I = torch.eye(num_nodes)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    # 计算邻接矩阵
    A_tilde = to_dense_adj(edge_index)[0]

    # 计算节点度数
    row, col = edge_index
    deg = degree(row)

    # 计算正则化度矩阵
    D_half_neg_tilde = torch.diag(deg ** -0.5)
    P_tilde = D_half_neg_tilde @ A_tilde @ D_half_neg_tilde

    return P_tilde

def eigen_decomposition(matrix):

    # 断言保证matrix是实对称矩阵
    assert np.array_equal(matrix, matrix.T)

    # np.linalg.eigh()用于计算对称矩阵的特征值和特征向量, 并返回按照升序排列的特征值。
    eigen_values, eigen_vectors = np.linalg.eigh(matrix)

    # 将小于1e-7的数值视为0
    eigen_values[abs(eigen_values) < 1e-7] = 0
    eigen_vectors[abs(eigen_vectors) < 1e-7] = 0

    # U是正交矩阵
    assert np.allclose(eigen_vectors.T, np.linalg.inv(eigen_vectors), atol=1e-7)

    return eigen_values, eigen_vectors


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
    # test data
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 4, 4, 5, 5], [1, 2, 3, 0, 2, 1, 0, 4, 5, 0, 2, 5, 2, 4]])
    L = get_L(edge_index, 6)
    print(L)
    #
    # eigenvalues, eigenvectors = eigen_decomposition(L)
    # print(eigenvalues)
    # print(eigenvectors)
    #
    # print('----------------------------------------------------------')
    # U = eigenvectors
    # print(np.linalg.eigvals(U))
    # U_12 = fractional_matrix_power(U, 0.5)
    # print(U_12)
    # print('----------------------------------------------------------')
    # # M = Matrix(np.array([[2, 0, -1, 0], [-1, 1, 0, -1], [0, 0, 2, 0], [1, 1, 1, 3]]))
    # # # M = Matrix(U)
    # # P, Ja = M.jordan_form()
    # # print(Ja)
    # print('----------------------------------------------------------')
    # # M = I - U
    # M = np.eye(U.shape[0]) - U
    # logU = 0
    # for k in range(1, 6):
    #     logU += M / k
    #     M = M @ M
    # print(logU)
