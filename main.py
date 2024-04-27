from dataset_loader import DataLoader
from utils import *


if __name__ == '__main__':
    dataset = DataLoader('texas')
    data = dataset[0]
    print(data)
    row, col = data.edge_index
    deg = degree(col, data.x.size(0), dtype=data.x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    print(deg_inv_sqrt)
    print(deg_inv_sqrt[row])
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    print(norm)


    input()

    L = get_L(data.edge_index, len(data.y))
    print(L)

    eigenvalues, eigenvectors = eigen_decomposition(L)
    print(eigenvalues)
    print(eigenvectors)
    print(eigenvectors.shape)

    # print("---------------------------------------------------------------------------------")
    print(np.linalg.eigvals(eigenvectors))
    # print(fractional_matrix_power(eigenvectors, 0.5))
