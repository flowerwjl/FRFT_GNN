import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from dataset_loader import DataLoader
import numpy as np
from torch.nn import Parameter
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class GPR_prop(MessagePassing):
    """
    propagation class for GPR_GNN
    """

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0 * np.ones(K + 1)
            TEMP[-1] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha) ** np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, edge_index, P_tilde, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x * (self.temp[0])
        for k in range(self.K):
            # x = self.propagate(edge_index, x=x, norm=norm)
            x = torch.matmul(P_tilde, x)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class Test_GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')    # 'add' Aggregation
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)
        # x has shape [N, out_channels]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        # out has shape [N, out_channels]

        return out

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class FracGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, P_tilde):
        x = self.lin(x)

        I = torch.eye(x.size(0))

        out = torch.matmul(P_tilde, x)

        return out


# class Test_ChebConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.lin = torch.nn.Linear(in_channels, out_channels)
#         self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         glorot(self.weight)
#
#     def forward(self, x, edge_index):


if __name__ == '__main__':
    pass
