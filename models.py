import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, ChebConv
from layers import FracGCNConv, GPR_prop
import torch.nn.functional as F
from utils import get_L_tilde, get_P_tilde
from scipy.linalg import fractional_matrix_power


class GPRGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)

        self.prop1 = GPR_prop(args.K, args.alpha, args.Init)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

        self.P_tilde = get_P_tilde(edge_index=dataset[0].edge_index)\
            .to(torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu'))

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
            x = self.prop1(x, edge_index, self.P_tilde)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index, self.P_tilde)
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


class FracGCN(torch.nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.conv1 = FracGCNConv(dataset.num_features, args.hidden)
        self.conv2 = FracGCNConv(args.hidden, dataset.num_classes)
        self.dropout = args.dropout

        self.L_tilde = get_L_tilde(edge_index=dataset[0].edge_index)\
            .to(torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu'))
        self.P_tilde = get_P_tilde(edge_index=dataset[0].edge_index)\
            .to(torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu'))

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x=x, P_tilde=self.P_tilde)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x=x, P_tilde=self.P_tilde)
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
