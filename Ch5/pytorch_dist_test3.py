import argparse
import os
import time
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader

import matplotlib.pylab as plt
import numpy as np
from torchvision.transforms import Compose, PILToTensor
from torchvision.datasets import MNIST

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


#to run DDP on one local node:
#python run.py --nnode 1 --nproc_per_node cpu pytorch_dist_test2.py --local_world_size 12


key_list = ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE']

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

        def forward(self, x):
            for layer in self.layer_list:
                x = self.activation(layer(x))
            
            x = self.output(x.reshape(-1, self.output.in_features)) #logits
            
            return x

    net = Net()
    criterion = nn.CrossEntropyLoss()

    return net, criterion

def create_dsdl(download_loc='./', dataset='MNIST'):
    if dataset == 'MNIST':

        ds_train = MNIST(root=download_loc, 
                         train=True, 
                         download=True,
                         transform=PILToTensor())

        ds_test = MNIST(root=download_loc, 
                        train=False, 
                        download=True,
                        transform=PILToTensor())
    
    elif dataset == 'CIFAR10':
    
        ds_train = CIFAR10(root=download_loc, 
                           train=True, 
                           download=True,
                           transform=PILToTensor())

        ds_test = CIFAR10(root=download_loc, 
                          train=False, 
                          download=True,
                          transform=PILToTensor())

    return ds_train, ds_test

def ddp_train_loop(rank, world_size, dataset='MNIST', n_epochs=50, batch_size=32, lr=1e-2, print_freq=2):
    print(f'rank = {rank} world_size = {world_size}')
    env_dict = {key: os.environ[key] for key in key_list}

    print(env_dict)

    dist.init_process_group(backend='gloo')

    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        f"world_size = {dist.get_world_size()}"
    )

    #temp logging
    f = open(f'log_world{world_size}_{rank}', 'w')

    #distributed sampling from dataset
    ds_train, ds_test = create_dsdl(dataset=dataset)
    sampler_train = DistributedSampler(ds_train)
    dl_train = DataLoader(dataset=ds_train, sampler=sampler_train, batch_size=batch_size)

    sampler_test = DistributedSampler(ds_test)
    #dl_test = DataLoader(dataset=ds_test, sampler=sampler_test, batch_size=batch_size)
    dl_test = DataLoader(dataset=ds_test, batch_size=batch_size)

    #model init
    net, criterion = create_net(lr=lr)    
    net = DDP(net, device_ids=None, output_device=None) #None for CPU-based

    optimizer = optim.Adam(net.parameters(), lr=lr) #net refers to the ddp model here

    #net = DDP(net, device_ids=rank, output_device=rank)
    #torch.device(rank) #torch.device("cuda") #torch.device(0)
    net = net.train() #in place?
    #net = net.to(rank)

    time_dict = {}
    start = time.time()
    for n in range(n_epochs):

        if n % print_freq == 0:
            total_loss = 0
            total_n = 0
            total_correct = 0

        for idx, (X, y) in enumerate(dl_train):
            X = (X - 128.) / 255.

            #X = X.to(rank)
            #y = y.to(rank)

            pred = net(X)
            loss = criterion(pred, y)

            if n % print_freq == 0:
                total_loss += loss
                total_n += X.shape[0]
                total_correct += (pred.argmax(dim=1)==y).sum()

            #print(f'rank={rank} batch loss={loss} epoch={n} batch_id={idx}')
            print(f'rank={rank} batch loss={loss} epoch={n} batch_id={idx} data_sum={X.sum()} data_shape={X.shape}', file=f)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(list(net.parameters()), file=f)

        if n % print_freq == 0:
            
            for idx, p in enumerate(list(net.parameters())):
                print(f'CHECK ME: rank={rank} epoch={n} param_n={idx} {p.sum()}', file=f)
            
            dist.barrier()
            val_acc = evaluate_model(net, dl_test)
            dist.barrier()
            
            print(f'rank={rank} epoch = {n} loss = {total_loss:.3f} train accuracy = {total_correct/total_n} val accuracy={val_acc:.2f}', file=f)

            time_dict[n] = (time.time()-start, val_acc)            
            print(time_dict, file=f)

    pickle.dump(time_dict, open(f'perf_world{world_size}_rank{rank}', 'wb'))

    f.close()
    dist.destroy_process_group()

    return net


def evaluate_model(net, dl_test, device='cpu'):
    net = net.eval()

    labels = torch.tensor([]).to(device)
    preds = torch.tensor([]).to(device)

    for idx, (X, y) in enumerate(dl_test):
        X = (X - 128.) / 255.
        #X = X.to(device)
        #y = y.to(device)

        pred = net(X)

        preds = torch.hstack((preds, pred.argmax(dim=1)))
        labels = torch.hstack((labels, y))

    accuracy = (preds==labels).sum() / len(labels)

    return accuracy

def example(rank, world_size):
    print(f'rank = {rank} world_size = {world_size}')
    env_dict = {key: os.environ[key] for key in key_list}

    print(env_dict)

    dist.init_process_group(backend='gloo')

    if rank == 0:
        data = torch.zeros(world_size)
        print(f'Data: ', data)

        for i in range(1, world_size):
            dist.recv(tensor=data[i], src=i)

        print(f'Data: ', data)

    else:
        dist.send(tensor=torch.tensor(float(world_size-rank)), dst=0)

        print(f'Rank {rank} sending {torch.tensor(rank)} to 0')

    dist.destroy_process_group()

#def main(world_size):
#    mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)

if __name__=='__main__':
    #not needed if using run.py which is recommended
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--local_rank", type=int)
    #parser.add_argument("--local_world_size", type=int)
    #args = parser.parse_args()

    #launch.py
    #example(args.local_rank, args.local_world_size)

    #run.py
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    #example(rank, world_size)
    ddp_train_loop(rank, world_size, dataset='MNIST', n_epochs=10, batch_size=120, lr=1e-2, print_freq=1)