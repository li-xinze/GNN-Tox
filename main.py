import os
import torch
import shutil
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score
from torch_geometric.data import DataLoader
from gnn_tox.data import MoleculeDataset
from gnn_tox.model import GNN_graphpred
from gnn_tox.splitters import scaffold_split, random_split


def seed_everything(seed):
    """sets seed for pseudo-random number generators in: pytorch, numpy
    """
    print(f"Global seed set to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


criterion = nn.BCEWithLogitsLoss(reduction='none')
def train(args, model, device, loader, optimizer):
    model.train()
    for step, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)
        is_valid = y**2 > 0
        loss_mat = criterion(pred.double(), (y+1)/2)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()
        optimizer.step()

def eval(args, model, device, loader, save=False):
    model.eval()
    y_true = []
    y_scores = []
    for step, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)
    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    # save predictions of test dataset for error analysis
    if save:
        np.save(f'analysis/tox21_{args.split}_pred.npy', np.array(y_scores))
    
    roc_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) 


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of gnn-tox')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 5)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of maximum epochs to train (default: 500)')
    parser.add_argument('--patience', type=int, default=50,
                        help='patience of early stopping (default: 100)')         
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean_sum",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gcn")
    parser.add_argument('--dataset', type=str, default = 'tox21', help='root directory of dataset. tox21/toxcast/clintox')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="random", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)
    print(dataset)
    smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
    if args.split == "scaffold":
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, 0.8, 0.1, 0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset, _ = random_split(dataset, 0.8, 0.1, 0.1, seed=args.seed, smiles_list=smiles_list)
        print("random")
    else:
        raise ValueError("Invalid split option.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    train_acc_list, val_acc_list, test_acc_list = [], [], []

    if not args.filename == "":
        fname = f'runs/{args.dataset}/{args.filename}/cls_runseed{args.runseed}'
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)

    patience = 0
    best_acc = 0
    for epoch in range(1, args.epochs+1):
        if patience == args.patience: break
        print("====epoch " + str(epoch))
        train(args, model, device, train_loader, optimizer)
        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader)
        else:
            train_acc = 0
        val_acc = eval(args, model, device, val_loader)
        test_acc = eval(args, model, device, test_loader)
        if val_acc > best_acc:
            patience = 0
            best_acc = val_acc
            eval(args, model, device, test_loader, save=True)
        else:
            patience += 1

        print(f'train: {train_acc} val: {val_acc} test: {test_acc}')
        print('p:', patience)
        print()

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)
        print('best auc: ', test_acc_list[val_acc_list.index(max(val_acc_list))])

        if not args.filename == '':
            writer.add_scalar('data/train auc', train_acc, epoch)
            writer.add_scalar('data/val auc', val_acc, epoch)
            writer.add_scalar('data/test auc', test_acc, epoch)

        print("")
    print('best auc: ', test_acc_list[val_acc_list.index(max(val_acc_list))])
    print('best auc: ', max(test_acc_list))

    if not args.filename == "":
        writer.close()

if __name__ == "__main__":
    main()
