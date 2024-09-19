import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, ChebConv
from layers import FracGCNConv, GPR_prop, Bern_prop
import torch.nn.functional as F
from utils import load_fractional_matrix, get_L_P_tilde_alpha


class BernNet(torch.nn.Module):
    def __init__(self, dataset, args, data):
        super(BernNet, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.m = torch.nn.BatchNorm1d(dataset.num_classes)
        self.prop1 = Bern_prop(args.K)

        self.dprate = args.dprate
        self.dropout = args.dropout

        self.L_tilde_alpha, self.P_tilde_alpha = load_fractional_matrix(
            dataset_name=args.dataset, power=args.frac_power)

        self.L_tilde_alpha = self.L_tilde_alpha.to(
            torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'))
        self.P_tilde_alpha = self.P_tilde_alpha.to(
            torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'))

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        #x= self.m(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index, L_tilde_alpha=self.L_tilde_alpha, P_tilde_alpha=self.P_tilde_alpha)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index, L_tilde_alpha=self.L_tilde_alpha, P_tilde_alpha=self.P_tilde_alpha)
            return F.log_softmax(x, dim=1)


class GPRGNN(torch.nn.Module):
    def __init__(self, dataset, args, data):
        super().__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)

        self.prop1 = GPR_prop(args.K, args.alpha, args.Init)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

        # self.L_tilde_alpha, self.P_tilde_alpha = load_fractional_matrix(
        #     dataset_name=args.dataset, power=args.frac_power)
        self.L_tilde_alpha, self.P_tilde_alpha = get_L_P_tilde_alpha(
            edge_index=data.edge_index.to('cpu'), power=args.frac_power, num_nodes=dataset[0].num_nodes)
        self.L_tilde_alpha2, self.P_tilde_alpha2 = get_L_P_tilde_alpha(
            edge_index=data.edge_index.to('cpu'), power=args.frac_power - 0.1, num_nodes=dataset[0].num_nodes)
        self.L_tilde_alpha3, self.P_tilde_alpha3 = get_L_P_tilde_alpha(
            edge_index=data.edge_index.to('cpu'), power=args.frac_power + 0.1, num_nodes=dataset[0].num_nodes)

        self.P_tilde_alpha = self.P_tilde_alpha.to(
            torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'))
        self.P_tilde_alpha2 = self.P_tilde_alpha.to(
            torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'))
        self.P_tilde_alpha3 = self.P_tilde_alpha.to(
            torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'))

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        x2, x3 = x, x

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index, P_tilde_alpha=self.P_tilde_alpha)
            x2 = self.prop1(x2, edge_index, P_tilde_alpha=self.P_tilde_alpha)
            x3 = self.prop1(x3, edge_index, P_tilde_alpha=self.P_tilde_alpha)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x2 = F.dropout(x2, p=self.dprate, training=self.training)
            x3 = F.dropout(x3, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index, P_tilde_alpha=self.P_tilde_alpha)
            x2 = self.prop1(x2, edge_index, P_tilde_alpha=self.P_tilde_alpha)
            x3 = self.prop1(x3, edge_index, P_tilde_alpha=self.P_tilde_alpha)

        x = torch.mean(torch.stack([x, x2, x3], dim=0), dim=0)

        return F.log_softmax(x, dim=1)


class MLP(torch.nn.Module):
    """
    搭建一个具有两个全连接层、一个ReLU和一个DropOut层的MLP模型，
    将1433维的节点特征映射到一个低维的特征（hidden_channels）上，
    然后输出为类别数
    """
    def __init__(self, dataset, args, data):
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
    def __init__(self, dataset, args, data):
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
    def __init__(self, dataset, args, data):
        super().__init__()
        self.conv1 = FracGCNConv(dataset.num_features, args.hidden)
        self.conv2 = FracGCNConv(args.hidden, dataset.num_classes)
        self.dropout = args.dropout

        # self.P_tilde = get_P_tilde(edge_index=dataset[0].edge_index).to(
        #     torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'))

        # self.L_tilde_alpha, self.P_tilde_alpha = load_fractional_matrix(
        #     dataset_name=args.dataset, power=args.frac_power)
        self.L_tilde_alpha, self.P_tilde_alpha = get_L_P_tilde_alpha(
            edge_index=data.edge_index.to('cpu'), power=args.frac_power, num_nodes=dataset[0].num_nodes)

        # self.L_tilde_alpha2, self.P_tilde_alpha2 = get_L_P_tilde_alpha(
        #     edge_index=data.edge_index.to('cpu'), power=args.frac_power - 0.1, num_nodes=dataset[0].num_nodes)
        #
        # self.L_tilde_alpha3, self.P_tilde_alpha3 = get_L_P_tilde_alpha(
        #     edge_index=data.edge_index.to('cpu'), power=args.frac_power + 0.1, num_nodes=dataset[0].num_nodes)

        # self.L_tilde_alpha = self.L_tilde_alpha.to(
        #     torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'))
        self.P_tilde_alpha = self.P_tilde_alpha.to(
            torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'))
        # self.P_tilde_alpha2 = self.P_tilde_alpha2.to(
        #     torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'))
        # self.P_tilde_alpha3 = self.P_tilde_alpha3.to(
        #     torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'))


    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x=x, P_tilde_alpha=self.P_tilde_alpha)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x=x, P_tilde_alpha=self.P_tilde_alpha)

        # x2 = data.x
        # x2 = self.conv1(x=x2, P_tilde_alpha=self.P_tilde_alpha2)
        # x2 = F.relu(x2)
        # x2 = F.dropout(x2, p=self.dropout, training=self.training)
        # x2 = self.conv2(x=x2, P_tilde_alpha=self.P_tilde_alpha2)
        #
        # x3 = data.x
        # x3 = self.conv1(x=x3, P_tilde_alpha=self.P_tilde_alpha3)
        # x3 = F.relu(x3)
        # x3 = F.dropout(x3, p=self.dropout, training=self.training)
        # x3 = self.conv2(x=x3, P_tilde_alpha=self.P_tilde_alpha3)
        #
        # x = torch.mean(torch.stack([x, x2, x3], dim=0), dim=0)
        # x = torch.max(x3, x)

        return F.log_softmax(x, dim=1)


class ChebNet(torch.nn.Module):
    def __init__(self, dataset, args, data):
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
