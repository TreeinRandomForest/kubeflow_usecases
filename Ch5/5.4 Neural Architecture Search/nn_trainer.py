import argparse
import os
import time
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pylab as plt
import numpy as np
from torchvision.transforms import Compose, PILToTensor
from torchvision.datasets import MNIST, CIFAR10

import katib_nn_parser as parser

device = "cuda:0"

key_list = ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE']

def create_net(arch, nn_config, lr=1e-2):
    net = parser.Net(arch, nn_config)
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

def train_loop(arch, nn_config, dataset='MNIST', n_epochs=50, batch_size=32, lr=1e-2, print_freq=2):
    #distributed sampling from dataset
    ds_train, ds_test = create_dsdl(dataset=dataset)
    
    dl_train = DataLoader(dataset=ds_train, batch_size=batch_size)
    dl_test = DataLoader(dataset=ds_test, batch_size=batch_size)

    #model init
    net, criterion = create_net(arch, nn_config, lr=lr)

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr) #net refers to the ddp model here

    net = net.train() #in place?

    time_dict = {}
    start = time.time()
    for n in range(n_epochs):

        if n % print_freq == 0:
            total_loss = 0
            total_n = 0
            total_correct = 0

        for idx, (X, y) in enumerate(dl_train):
            X = (X - 128.) / 255.

            X = X.flatten(start_dim=1, end_dim=-1)
            X = X.to(device)
            y = y.to(device)

            pred = net(X)
            loss = criterion(pred, y)

            if n % print_freq == 0:
                total_loss += loss
                total_n += X.shape[0]
                total_correct += (pred.argmax(dim=1)==y).sum()

            #print(f'batch loss={loss} epoch={n} batch_id={idx} data_sum={X.sum()} data_shape={X.shape}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if n % print_freq == 0:
            val_acc = evaluate_model(net, dl_test, device=device)
            print(f'epoch = {n} total_examples={total_n} loss = {total_loss:.3f} train accuracy = {total_correct/total_n} val accuracy={val_acc:.2f}')

    return net


def evaluate_model(net, dl_test, device='cpu'):
    net = net.eval()

    labels = torch.tensor([]).to(device)
    preds = torch.tensor([]).to(device)

    for idx, (X, y) in enumerate(dl_test):
        X = (X - 128.) / 255.
        
        X = X.flatten(start_dim=1, end_dim=-1)
        X = X.to(device)
        y = y.to(device)

        pred = net(X)

        preds = torch.hstack((preds, pred.argmax(dim=1)))
        labels = torch.hstack((labels, y))

    accuracy = (preds==labels).sum() / len(labels)

    return accuracy

