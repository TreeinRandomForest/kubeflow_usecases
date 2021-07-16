import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pylab as plt
import numpy as np
from torchvision.transforms import Compose, PILToTensor
from torchvision.datasets import MNIST

#DDP
import os
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

plt.ion()

#device = 'cuda:0' if torch.cuda.is_available() else "cpu"
device = 'cpu'

key_list = ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE']

def create_dsdl(download_loc='./', batch_size=32):
    ds_train = MNIST(root=download_loc, 
                     train=True, 
                     download=True,
                     transform=PILToTensor())
    ds_test = MNIST(root=download_loc, 
                    train=False, 
                    download=True,
                    transform=PILToTensor())

    dl_train = DataLoader(ds_train, batch_size=batch_size)
    dl_test = DataLoader(ds_test, batch_size=batch_size)

    return dl_train, dl_test

def create_net(lr=1e-2):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.layer_list = nn.ModuleList()
            self.layer_list.append(nn.Conv2d(1, 16, 3, 1))
            self.layer_list.append(nn.Conv2d(16, 32, 3, 1))
            self.layer_list.append(nn.Conv2d(32, 64, 3, 1))
            self.activation = nn.ReLU()

            linear_input = ((28 - 2*len(self.layer_list))**2)*self.layer_list[-1].out_channels
            self.output = nn.Linear(linear_input, 10)
            #self.output_activation = nn.Softmax(dim=1)

        def forward(self, x):
            for layer in self.layer_list:
                x = self.activation(layer(x))
            
            x = self.output(x.reshape(-1, self.output.in_features)) #logits
            
            return x

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    return net, criterion, optimizer 

def train_loop(n_epochs=50, batch_size=32, lr=1e-2, print_freq=2, device=device):
    dl_train, dl_test = create_dsdl(batch_size=batch_size)

    net, criterion, optimizer = create_net(lr=lr)

    net = net.train() #in place?

    net = net.to(device)

    with torch.profiler.profile(schedule=torch.profiler.schedule(wait=2, warmup=2, active=4), 
                                with_stack=True,
                                on_trace_ready=torch.profiler.tensorboard_trace_handler,
        ) as prof:

        for n in range(n_epochs):

            if n % print_freq == 0:
                total_loss = 0
                total_n = 0
                total_correct = 0

            for idx, (X, y) in enumerate(dl_train):
                X = (X - 128.) / 255.

                X = X.to(device)
                y = y.to(device)

                pred = net(X)
                loss = criterion(pred, y)

                if n % print_freq == 0:
                    total_loss += loss
                    total_n += X.shape[0]
                    total_correct += (pred.argmax(dim=1)==y).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                prof.step()

            if n % print_freq == 0:
                val_acc = evaluate_model(net, dl_test)
                print(f'Epoch = {n} Loss = {loss:.3f} Train Accuracy={total_correct/total_n:.2f} Val Accuracy={val_acc:.2f}')

    return net

def ddp_train_loop(rank, world_size, n_epochs=50, batch_size=32, lr=1e-2, print_freq=2):
    print(f'rank = {rank} world_size = {world_size}')
    env_dict = {key: os.environ[key] for key in key_list}

    print(env_dict)

    dist.init_process_group(backend='nccl')

    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        f"world_size = {dist.get_world_size()}"
    )

    dl_train, dl_test = create_dsdl(batch_size=batch_size)

    net, criterion, optimizer = create_net(lr=lr)

    net = net.train() #in place?
    net = net.to(rank)
    net = DDP(net, device_ids=[rank])

    for n in range(n_epochs):

        if n % print_freq == 0:
            total_loss = 0
            total_n = 0
            total_correct = 0

        for idx, (X, y) in enumerate(dl_train):
            X = (X - 128.) / 255.

            X = X.to(rank)
            y = y.to(rank)

            pred = net(X)
            loss = criterion(pred, y)

            if n % print_freq == 0:
                total_loss += loss
                total_n += X.shape[0]
                total_correct += (pred.argmax(dim=1)==y).sum()

            print(f'rank={rank} batch loss={loss} epoch={n} batch_id={idx}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if n % print_freq == 0:
            val_acc = evaluate_model(net, dl_test)
            print(f'Epoch = {n} Loss = {loss:.3f} Train Accuracy={total_correct/total_n:.2f} Val Accuracy={val_acc:.2f}')

    dist.destroy_process_group()

    return net


def evaluate_model(net, dl_test, device='cpu'):
    net = net.eval()

    labels = torch.tensor([]).to(device)
    preds = torch.tensor([]).to(device)

    for idx, (X, y) in enumerate(dl_test):
        X = (X - 128.) / 255.
        X = X.to(device)
        y = y.to(device)

        pred = net(X)

        preds = torch.hstack((preds, pred.argmax(dim=1)))
        labels = torch.hstack((labels, y))

    accuracy = (preds==labels).sum() / len(labels)

    return accuracy

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--local_world_size", type=int)
    args = parser.parse_args()

    ddp_train_loop(args.local_rank, args.local_world_size)


