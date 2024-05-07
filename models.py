import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, ChebConv
from layers import Test_GCNConv, Test_GCNConv2
import torch.nn.functional as F
from utils import get_L, get_P_tilde
from scipy.linalg import fractional_matrix_power
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
import numpy as np


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

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x * (self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GPRGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)

        self.prop1 = GPR_prop(args.K, args.alpha, args.Init)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)


class MLP(torch.nn.Module):
    """
    搭建一个具有两个全连接层、一个ReLU和一个DropOut层的MLP模型，
    将1433维的节点特征映射到一个低维的特征（hidden_channels）上，
    然后输出为类别数
    """
    def __init__(self, dataset, args):
        super().__init__()

        # 两个全连接层
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x = data.x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class GCN(torch.nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class Test_GCN(torch.nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.conv1 = Test_GCNConv(dataset.num_features, args.hidden)
        self.conv2 = Test_GCNConv(args.hidden, dataset.num_classes)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class Test_GCN2(torch.nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.conv1 = Test_GCNConv2(dataset.num_features, args.hidden)
        self.conv2 = Test_GCNConv2(args.hidden, dataset.num_classes)
        self.dropout = args.dropout
        self.L = get_L(edge_index=dataset[0].edge_index, num_nodes=dataset[0].num_nodes)
        self.P_tilde = get_P_tilde(edge_index=dataset[0].edge_index, num_nodes=dataset[0].num_nodes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x=x, L=self.L)
        # x = self.conv1(x=x, L=self.P_tilde)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x=x, L=self.L)
        # x = self.conv2(x=x, L=self.P_tilde)
        return F.log_softmax(x, dim=1)


class ChebNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.conv1 = ChebConv(dataset.num_features, args.hidden, K=args.K+1)
        self.conv2 = ChebConv(args.hidden, dataset.num_classes, K=args.K+1)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
