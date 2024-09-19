import argparse
from models import MLP, GCN, FracGCN, ChebNet, GPRGNN, BernNet
from dataset_loader import DataLoader
from utils import *
import torch.nn.functional as F
from tqdm import tqdm
import time
import seaborn as sns
import seaborn.algorithms
import pandas as pd
from torch_geometric.utils import dropout_edge, add_random_edge, mask_feature
import torch


def train(model, optimizer, data, dprate):
    model.train()
    optimizer.zero_grad()
    out = model(data)[data.train_mask]
    loss = F.nll_loss(out, data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    del out


def test(model, data):
    model.eval()
    logits, accs, losses, preds = model(data), [], [], []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        loss = F.nll_loss(model(data)[mask], data.y[mask])
        preds.append(pred.detach().cpu())
        accs.append(acc)
        losses.append(loss.detach().cpu())
    return accs, preds, losses


def RunExp(args, dataset, data, Net, percls_trn, val_lb):
    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    tmp_net = Net(dataset, args, data=data)

    # Using the dataset splits described in the paper.
    data = random_splits(data, dataset.num_classes, percls_trn, val_lb, args.seed)

    model, data = tmp_net.to(device), data.to(device)

    if args.net == 'GPRGNN':
        optimizer = torch.optim.Adam(
            [{'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
             {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
             {'params': model.prop1.parameters(), 'weight_decay': 0.00, 'lr': args.lr}])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    time_run = []
    for epoch in range(args.epochs):
        t_st = time.time()
        train(model, optimizer, data, args.dprate)
        time_epoch = time.time() - t_st  # each epoch train times
        time_run.append(time_epoch)
        [train_acc, val_acc, tmp_test_acc], preds, [train_loss, val_loss, tmp_test_loss] = test(model, data)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc

            theta = args.alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if 0 < args.early_stopping < epoch:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break
    return test_acc, best_val_acc, theta, time_run


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay.')
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')
    parser.add_argument('--train_rate', type=float, default=0.6, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')
    parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')
    parser.add_argument('--runs', type=int, default=10, help='number of runs.')
    parser.add_argument('--device', type=int, default=2, help='GPU device.')

    parser.add_argument('--K', type=int, default=10, help='propagation steps.')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha for APPN.')
    parser.add_argument('--Init', type=str, choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                        default='NPPR', help='initialization for GPRGNN.')

    parser.add_argument('--dataset', type=str,
                        choices=['Cora', 'Citeseer', 'Pubmed', 'Chameleon', 'Squirrel', 'Actor', 'Texas', 'Cornell'],
                        default='Cornell')
    parser.add_argument('--net', type=str, choices=['GCN', 'FracGCN', 'GPRGNN', 'SGC', 'APPNP', 'BernNet'],
                        default='FracGCN')
    # parser.add_argument('--prop_lr', type=float, default=0.01, help='learning rate for propagation layer.')
    # parser.add_argument('--prop_wd', type=float, default=0.0005, help='learning rate for propagation layer.')

    # parser.add_argument('--q', type=int, default=0, help='The constant for ChebBase.')
    parser.add_argument('--full', type=bool, default=True, help='full-supervise with random splits')
    # parser.add_argument('--semi_rnd', type=bool, default=False, help='semi-supervised with random splits')
    # parser.add_argument('--semi_fix', type=bool, default=False, help='semi-supervised with fixed splits')

    parser.add_argument('--frac_power', type=float, choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        default=1.0, help='fractional power order for Fractional GNNs')
    parser.add_argument('--dropout_edge', type=float, default=0.0,
                        help='dropout rate of dropping edge')
    parser.add_argument('--add_edge', type=float, default=0.0,
                        help='add rate of edge')
    parser.add_argument('--mask_p', type=float, default=0.0)
    parser.add_argument('--noise', type=bool, default=False)
    parser.add_argument('--drop_node', type=float, default=0.0)
    parser.add_argument('--change_labels', type=float, default=0.0)

    args = parser.parse_args()

    print(args)
    print("---------------------------------------------")

    dataset = DataLoader(args.dataset)
    data = dataset[0]


    gnn_name = args.net
    if gnn_name == 'GCN':
        Net = GCN
    elif gnn_name == 'MLP':
        Net = MLP
    elif gnn_name == 'FracGCN':
        Net = FracGCN
    elif gnn_name == 'ChebNet':
        Net = ChebNet
    elif gnn_name == 'GPRGNN':
        Net = GPRGNN
        args.Init = 'NPPR'
    elif gnn_name == 'SGC':
        args.Init = 'SGC'
        Net = GPRGNN
    elif gnn_name == 'APPNP':
        args.Init = 'PPR'
        Net = GPRGNN
    elif gnn_name == 'BernNet':
        Net = BernNet

    if args.full:
        args.train_rate = 0.6
        args.val_rate = 0.2
    else:
        args.train_rate = 0.025
        args.val_rate = 0.025

    percls_trn = int(round(args.train_rate * len(data.y) / dataset.num_classes))
    val_lb = int(round(args.val_rate * len(data.y)))

    results = []
    time_results = []
    for RP in tqdm(range(args.runs)):
        data = dataset[0]
        data.edge_index, _ = add_random_edge(edge_index=data.edge_index, force_undirected=True, p=args.add_edge)
        data.edge_index, _ = dropout_edge(data.edge_index, force_undirected=True, p=args.dropout_edge)
        data.x, _ = mask_feature(data.x, mode='all', p=args.mask_p)
        if args.noise:
            noise = torch.randn_like(data.x) * 0.005
            data.x = data.x + noise
        if args.change_labels != 0:
            # 计算要替换的元素数量
            num_elements_to_replace = int(len(data.y) * args.change_labels)
            # 随机选择要替换的索引
            indices_to_replace = torch.randperm(len(data.y))[:num_elements_to_replace]
            # 生成0到k范围内的随机整数
            random_integers = torch.randint(0, dataset.num_classes, (num_elements_to_replace,))
            # 替换指定索引处的元素
            data.y[indices_to_replace] = random_integers

        test_acc, best_val_acc, theta_0, time_run = RunExp(args, dataset, data, Net, percls_trn, val_lb)
        time_results.append(time_run)
        results.append([test_acc, best_val_acc, theta_0])
        print(f'run_{str(RP + 1)} \t test_acc: {test_acc:.4f}')

    run_sum = 0
    epochsss = 0
    for i in time_results:
        run_sum += sum(i)
        epochsss += len(i)
    print("each run avg_time:", run_sum / args.runs, "s")
    print("each epoch avg_time:", 1000 * run_sum / epochsss, "ms")

    results.remove(max(results))
    results.remove(min(results))
    test_acc_mean, val_acc_mean, _ = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100
    values = np.asarray(results, dtype=object)[:, 0]
    uncertainty = np.max(
        np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))
    print(f'{gnn_name} on dataset {args.dataset}, in {args.runs} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty * 100:.4f}  \t val acc mean = {val_acc_mean:.4f}')


    with open('run_log/run_log.txt', 'a') as f:
        f.write(f'\n{gnn_name} on dataset {args.dataset}, frac_order={args.frac_power}, drop:{args.dropout_edge}, add:{args.add_edge}')
        f.write(f'\neach run avg_time: {run_sum / args.runs}s')
        f.write(f'\neach epoch avg_time: {1000 * run_sum / epochsss}ms')
        f.write(f'\ntest acc mean = {test_acc_mean:.2f}±{uncertainty * 100:.2f}  \t val acc mean = {val_acc_mean:.2f}')
        f.write(f'\n')
    #
    #
    # # 读取CSV文件
    # df = pd.read_csv(f'run_log/{gnn_name}_acc.csv')
    #
    # # 定位到列名为'AAA'的列
    # column_name = args.dataset
    # column_index = df.columns.get_loc(column_name)
    #
    # # 在第3行写入数据90
    # df.at[int(args.frac_power * 10), column_name] = f'{test_acc_mean:.2f}±{uncertainty * 100:.2f}'
    #
    # # 保存修改后的CSV文件
    # df.to_csv(f'run_log/{gnn_name}_acc.csv', index=False)
    #
    #
    # # 读取CSV文件
    # df = pd.read_csv(f'run_log/{gnn_name}_time.csv')
    #
    # # 在第3行写入数据90
    # df.at[int(args.frac_power * 10), column_name] = f'{run_sum / args.runs: .2f}'
    #
    # # 保存修改后的CSV文件
    # df.to_csv(f'run_log/{gnn_name}_time.csv', index=False)
