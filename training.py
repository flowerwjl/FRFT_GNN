import argparse
from models import MLP, GCN, FracGCN, ChebNet, GPRGNN
from dataset_loader import DataLoader
from utils import *
import torch.nn.functional as F
from tqdm import tqdm
import time
import seaborn as sns
import seaborn.algorithms


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
    tmp_net = Net(dataset, args)

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
    parser.add_argument('--K', type=int, default=10, help='propagation steps.')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha for APPN.')
    parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')
    parser.add_argument('--Init', type=str, choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                        default='SGC', help='initialization for GPRGNN.')

    parser.add_argument('--dataset', type=str,
                        choices=['Cora', 'Citeseer', 'Pubmed', 'Chameleon', 'Squirrel', 'Actor', 'Texas', 'Cornell'],
                        default='Cornell')
    parser.add_argument('--device', type=int, default=0, help='GPU device.')
    parser.add_argument('--runs', type=int, default=10, help='number of runs.')
    parser.add_argument('--net', type=str, choices=['MLP', 'GCN', 'FracGCN', 'ChebNet', 'GPRGNN'],
                        default='GPRGNN')
    # parser.add_argument('--prop_lr', type=float, default=0.01, help='learning rate for propagation layer.')
    # parser.add_argument('--prop_wd', type=float, default=0.0005, help='learning rate for propagation layer.')

    # parser.add_argument('--q', type=int, default=0, help='The constant for ChebBase.')
    parser.add_argument('--full', type=bool, default=True, help='full-supervise with random splits')
    # parser.add_argument('--semi_rnd', type=bool, default=False, help='semi-supervised with random splits')
    # parser.add_argument('--semi_fix', type=bool, default=False, help='semi-supervised with fixed splits')

    args = parser.parse_args()

    print(args)
    print("---------------------------------------------")

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

    dataset = DataLoader(args.dataset)
    data = dataset[0]

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
    test_acc_mean, val_acc_mean, _ = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100
    values = np.asarray(results, dtype=object)[:, 0]
    uncertainty = np.max(
        np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))
    print(f'{gnn_name} on dataset {args.dataset}, in {args.runs} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty * 100:.4f}  \t val acc mean = {val_acc_mean:.4f}')
