import torch.nn as nn
import torch
import torch.optim as optim
import argparse
import json
import time
import numpy as np

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import PILToTensor
from torch.utils.data import DataLoader

def parse_config(arch, nn_config):
    '''
    Note: this was tested on conv nets only.

    arch: [[i1], [i2, X], [i3,X,X]]
    where: X is either 0 or 1 (boolean)

    iK is a key in nn_config["embedding"] signifying which layer to pick

    X booleans signify whether a skip connection from preivous layers
    is present or not

    Note: i1 is the first layer *after* the input layer

    In particular,
    i1 only has inputs from input layer
    i2 has inputs from i1 and possibly from input layer (one X)
    i3 has inputs from i2 and possibly from i1/input
    iK has inputs from i(K-1) and possibly from i{1,...,K-2}/input

    Skip connections (from previous layers) are: 

    '''
    embedding = nn_config['embedding']
    input_size = nn_config['input_sizes'][0]
    output_size = nn_config['output_sizes'][0]

    unit_list = []
    act_list = [input_size]
    in_val = input_size
    out_val = None
    carry = 0
    skip_cons = {0: []}

    for idx, l in enumerate(arch):
        layer_type = embedding[str(l[0])]
        if layer_type['opt_type']!='dense': raise ValueError("found non-dense layer")

        out_val = int(layer_type['opt_params']['units'])
        act_list.append(out_val)

        unit_list.append((in_val + carry, out_val))
        in_val = out_val
        carry = 0

        skip_cons[idx+1] = []
        for skip_id in range(1, len(l)):
            skip_val = l[skip_id]
            
            if skip_val==1:
                #update in_val
                #carry += unit_list[skip_id-1][0] #FIX
                carry += act_list[skip_id-1]
                skip_cons[idx+1].append(skip_id-1)

    unit_list.append((in_val + carry, output_size))

    return unit_list, skip_cons


class Net(nn.Module):
    def __init__(self, arch, nn_config):
        super(Net, self).__init__()

        self.unit_list, self.skip_cons = parse_config(arch, nn_config)
        
        self.layer_list = nn.ModuleList()
        for l in self.unit_list:
            self.layer_list.append(nn.Linear(l[0], l[1]))
        
        self.activation = nn.ReLU()
        #self.output_activation = nn.Softmax(dim=1)

    def forward(self, x):
        out = x
        self.buffer = [x]
        for idx, l in enumerate(self.layer_list):
            
            out = torch.cat([self.buffer[t] for t in self.skip_cons[idx]] + [out], dim=1)

            if idx==len(self.layer_list)-1:
                out = l(out)
            else:
                out = self.activation(l(out))            

            self.buffer.append(out)

        return out


def train_model(numepochs, arch, nn_config):
    #TODO add gpu support
    device='cpu'
    #Get the training set
    cfarDatasets = datasets.CIFAR10
    
    ds_train = cfarDatasets(root="./trainingdata", train=True, download=True, transform=PILToTensor())
    ds_test = cfarDatasets(root="./trainingdata", train=False, download=True, transform=PILToTensor())
    
    learningrate=1e-2

    arch_dict = json.loads(arch)
    nn_config_dict = json.loads(nn_config)
    
    dl_train = DataLoader(dataset=ds_train, batch_size=32)
    dl_test = DataLoader(dataset=ds_test, batch_size=32)

    model = Net(arch_dict, nn_config_dict)
    model = model.to(device)    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate) #Adam optimizer
    model=model.train()
    
    time_dict = {}
    start = time.time()
    print_freq=2
    for n in range(numepochs): 
        if n % print_freq == 0:
            total_loss = 0
            total_n = 0
            total_correct = 0

        for idx, (X, y) in enumerate(dl_train):
            X = (X - 128.) / 255.

            X = X.flatten(start_dim=1, end_dim=-1)
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            loss = criterion(pred, y)

            if n % print_freq == 0:
                total_loss += loss
                total_n += X.shape[0]
                total_correct += (pred.argmax(dim=1)==y).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print_freq = 1
        if n % print_freq == 0:
            val_acc = evaluate_model(model, dl_test, device=device)
            print(f"epoch {n+1}:")
            print(f'Validation-Accuracy={val_acc}')
            print("")

    return  model

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PytorchTraining')
    parser.add_argument('--architecture', type=str, default="", metavar='N', help='architecture of the neural network')
    parser.add_argument('--nn_config', type=str, default="", metavar='N', help='configurations and search space embeddings')
    parser.add_argument('--num_epochs', type=int, default=10, metavar='N', help='number of epoches that each child will be trained')
    #parser.add_argument('--num_gpus', type=int, default=1, metavar='N',
                    #help='number of GPU that used for training')

    args = parser.parse_args()
    # Get Algorithm Settings
    arch = args.architecture.replace("\'", "\"")
    print(">>> arch received by trial")
    print(arch)

    nn_config = args.nn_config.replace("\'", "\"")
    print(">>> nn_config received by trial")
    print(nn_config)

    num_epochs = args.num_epochs
    print(">>> num_epochs received by trial")
    print(num_epochs)

    #Using CPU for now
    device = torch.device("cpu")
    print(">>> Use CPU for Training <<<")

    # uncomment for local debugging purposes
    #arch = '[[1],[1, 0],[1, 1, 1]]'
    #nn_config = '{"input_sizes": [3072],"output_sizes": [10],"embedding": {"1": {"opt_type": "dense","opt_params": {"units": "10"}}}}'
    train_model(num_epochs, arch, nn_config)
	
	



