import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from dataset_loader import DataLoader
from utils import get_L
from torch_geometric.nn import GCNConv


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


class Test_GCNConv2(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, L):
        x = self.lin(x)

        I = torch.eye(x.size(0))

        out = torch.matmul(2*I - L, x)
        # out = torch.matmul(L, x)

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
    # dataset = DataLoader('texas')
    # data = dataset[0]
    #
    # L = get_L(edge_index=data.edge_index, num_nodes=data.num_nodes)
    #
    # model = Test_GCNConv(1703, 100)
    # result = model(data.x, data.edge_index)
    #
    # model2 = Test_GCNConv2(1703, 100)
    # result2 = model2(data.x, data.edge_index, L)

    x = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                     dtype=torch.float)
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 0, 0]])
    L = get_L(edge_index=edge_index, num_nodes=3)

    layer1 = Test_GCNConv(3, 3)
    result1 = layer1(x, edge_index)
    print(result1)

    layer2 = Test_GCNConv2(3, 3)
    result2 = layer2(x, edge_index, L)
    print(result2)

    layer3 = GCNConv(3, 3)
    result3 = layer3(x, edge_index)
    print(result3)
