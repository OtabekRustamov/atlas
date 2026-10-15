import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from models.gintransformer import GINTransformer
from utils import *

MODELS = [GINConvNet, GATNet, GAT_GCN, GCNNet, GINTransformer]
DATASETS = ['davis', 'kiba']


def train(model, device, train_loader, optimizer, loss_fn, epoch, log_interval):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        mask1 = create_padding_mask(data.target, 0).to(device)
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data, mask1)
        loss = loss_fn(output, data.y.view(-1, 1).float())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * train_loader.batch_size,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


def predicting(model, device, loader):
    model.eval()
    total_preds, total_labels = [], []
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            mask1 = create_padding_mask(data.target, 0).to(device)
            data = data.to(device)
            output = model(data, mask1)
            total_preds.append(output.cpu())
            total_labels.append(data.y.view(-1, 1).cpu())
    return torch.cat(total_labels).numpy().flatten(), torch.cat(total_preds).numpy().flatten()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train on the train split, select the checkpoint on a held-out validation split, '
                    'and evaluate the selected checkpoint once on the test split.')
    parser.add_argument('dataset', type=int, choices=range(len(DATASETS)), help='0: davis, 1: kiba')
    parser.add_argument('model', type=int, choices=range(len(MODELS)),
                        help='0: GINConvNet, 1: GATNet, 2: GAT_GCN, 3: GCNNet, 4: GINTransformer')
    parser.add_argument('cuda', type=int, nargs='?', default=0, help='CUDA device index')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=0,
                        help='stop after this many epochs without validation improvement (0 = never stop early)')
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--model_dir', type=str, default='trained_model/',
                        help='where the best checkpoint is saved (inference.py reads from here by default)')
    opt = parser.parse_args()
    if opt.epochs < 1:
        parser.error('--epochs must be >= 1')
    if opt.patience < 0:
        parser.error('--patience must be >= 0')
    if not 0 < opt.val_frac < 1:
        parser.error('--val_frac must be between 0 and 1')
    return opt


def main():
    opt = parse_args()
    dataset = DATASETS[opt.dataset]
    modeling = MODELS[opt.model]
    model_st = modeling.__name__

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    device = torch.device('cuda:{}'.format(opt.cuda) if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    print('Learning rate: ', opt.lr)
    print('Epochs: ', opt.epochs)
    print('\nrunning on ', model_st + '_' + dataset)

    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if not os.path.isfile(processed_data_file_train) or not os.path.isfile(processed_data_file_test):
        print('please run create_data.py to prepare data in pytorch format!')
        return

    full_train_data = TestbedDataset(root='data', dataset=dataset + '_train')
    test_data = TestbedDataset(root='data', dataset=dataset + '_test')

    valid_size = int(opt.val_frac * len(full_train_data))
    train_size = len(full_train_data) - valid_size
    train_data, valid_data = torch.utils.data.random_split(
        full_train_data, [train_size, valid_size], generator=torch.Generator().manual_seed(opt.seed))

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=opt.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)

    model = modeling().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    os.makedirs(opt.model_dir, exist_ok=True)
    model_file_name = os.path.join(opt.model_dir, 'model_' + model_st + '_' + dataset + '.model')
    result_file_name = 'result_' + model_st + '_' + dataset + '.csv'

    best_val_mse = float('inf')
    best_epoch = -1
    best_state = None
    for epoch in range(1, opt.epochs + 1):
        train(model, device, train_loader, optimizer, loss_fn, epoch, opt.log_interval)
        G, P = predicting(model, device, valid_loader)
        val_mse = mse(G, P)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, model_file_name)
            print('val mse improved at epoch', best_epoch, '; best_val_mse:', best_val_mse, model_st, dataset)
        else:
            print(val_mse, 'No improvement since epoch', best_epoch, '; best_val_mse:', best_val_mse, model_st, dataset)
        if opt.patience and epoch - best_epoch >= opt.patience:
            print('Early stopping at epoch', epoch)
            break

    if best_state is None:
        raise SystemExit('Validation MSE never improved (training diverged to NaN?); no test results written.')

    # The test split is used exactly once, on this run's in-memory best-validation weights
    # (not re-read from disk, where another run could have overwritten the file).
    model.load_state_dict(best_state)
    G, P = predicting(model, device, test_loader)
    ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
    with open(result_file_name, 'w') as f:
        f.write('rmse,mse,pearson,spearman,ci,best_epoch,best_val_mse\n')
        f.write(','.join(map(str, ret + [best_epoch, best_val_mse])) + '\n')
    print('test results ({} {}, best epoch {}): rmse={:.4f} mse={:.4f} pearson={:.4f} spearman={:.4f} ci={:.4f}'
          .format(model_st, dataset, best_epoch, *ret))


if __name__ == '__main__':
    main()
