from utils import get_L_tilde, eigen_decomposition
import scipy
from dataset_loader import DataLoader
import os
import numpy as np


dataset_name_list = ['Chameleon', 'Squirrel', 'Actor']

for dataset_name in dataset_name_list:

    dataset = DataLoader(dataset_name)
    data = dataset[0]

    # create directory if not exist
    os.makedirs(f'fractional_matrix/{dataset_name}', exist_ok=True)

    L_tilde = get_L_tilde(edge_index=data.edge_index, num_nodes=data.num_nodes)
    e_val, e_vec = eigen_decomposition(L_tilde)

    e_vecs = [1 for _ in range(20)]

    e_vecs[1] = scipy.linalg.fractional_matrix_power(e_vec, 0.1)

    e_vecs[2] = np.linalg.matrix_power(e_vecs[1], 2)
    e_vecs[3] = np.linalg.matrix_power(e_vecs[1], 3)
    e_vecs[4] = np.linalg.matrix_power(e_vecs[2], 2)
    e_vecs[5] = np.linalg.matrix_power(e_vecs[1], 5)
    e_vecs[6] = np.linalg.matrix_power(e_vecs[3], 2)
    e_vecs[7] = np.linalg.matrix_power(e_vecs[1], 7)
    e_vecs[8] = np.linalg.matrix_power(e_vecs[4], 2)
    e_vecs[9] = np.linalg.matrix_power(e_vecs[3], 3)
    e_vecs[10] = np.linalg.matrix_power(e_vecs[5], 2)

    e_vecs_H = [1 for _ in range(20)]

    for i in range(1, 11):
        e_vecs_H[i] = np.conj(e_vecs[i].T)
        L_tilde_alpha = e_vecs[i] @ np.diag(e_val) @ e_vecs_H[i]

        I = np.eye(data.num_nodes)

        P_tilde_alpha = I - L_tilde_alpha
        np.save(f'fractional_matrix/{dataset_name}/L_tilde_alpha_{i/10}.npy', L_tilde_alpha)
        np.save(f'fractional_matrix/{dataset_name}/P_tilde_alpha_{i/10}.npy', P_tilde_alpha)
        print(f'{dataset_name} fractional order {i/10} matrix saved.')
