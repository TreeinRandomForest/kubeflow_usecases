import torch.nn as nn
import torch
import torch.optim as optim
import argparse
import json

import numpy as np

import torchvision.datasets as datasets
import torchvision.transforms as transforms


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
    input_size = nn_config['input_sizes']
    output_size = nn_config['output_sizes']

    unit_list = []
    act_list = [5]
    in_val = input_size[0]
    out_val = None
    carry = 0
    skip_cons = {0: []}

    for idx, l in enumerate(arch):
        layer_type = embedding[str(l[0])]
        if layer_type['opt_type']!='dense': raise ValueError("found non-dense layer")

        out_val = layer_type['opt_params']['units']
        act_list.append(out_val)
        print(f'Type for out_val is{type(out_val)} and for carry {type(carry)}')
        print(f'Type for out_val value is{out_val} and for carry value {carry}')
        print(f'Type for in_val value is {in_val} and for in_val tyle {type(in_val)}')
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

    unit_list.append((in_val + carry, output_size[0]))

    return unit_list, skip_cons


class Net(nn.Module):
    def __init__(self, arch, nn_config):
        super(Net, self).__init__()

        self.unit_list, self.skip_cons = parse_config(arch, nn_config)

        self.layer_list = nn.ModuleList()
        for l in self.unit_list:
            self.layer_list.append(nn.Linear(l[0], l[1]))

        self.activation = nn.ReLU()
        self.output_activation = nn.Softmax(dim=1)

    def forward(self, x):
        out = x
        self.buffer = [x]
        for idx, l in enumerate(self.layer_list):

            out = torch.cat([self.buffer[t] for t in self.skip_cons[idx]] + [out], dim=1)

            out = self.activation(l(out))

            self.buffer.append(out)

        return out

#def train_model(learningrate, numepochs, numhiddenlayers, numnodes,activation) -> int:
def train_model(numepochs, arch, nn_config) -> int:
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #TODO add gpu support
    device='cpu'
    #print(f'Device = {device}')
    #features_train = torch.from_numpy(read_from_store(bucket_name, 'features_train')).float()
    #target_train = torch.from_numpy(read_from_store(bucket_name, 'target_train')).float()
    #features_test = torch.from_numpy(read_from_store(bucket_name, 'features_test')).float()
    #target_test = torch.from_numpy(read_from_store(bucket_name, 'target_test')).float()
    #Get the training set
    cfarDatasets = datasets.CIFAR10
   # numberOfClasses = 10
   # numberOfInputs = 3
    # Do preprocessing TODO CHANGE
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]
    #might be taken out
    transf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]

    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]

    train_transform = transforms.Compose(transf + normalize)
    traininingdata = cfarDatasets(root="./trainingdata", train=True, download=True, transform=train_transform)
    
    #lr = float(conf.get('lr', 1e-2))
    learningrate=1e-2
    #N_epochs = int(conf.get('N_epochs', 1000))
    #num_hidden_layers = int(conf.get('num_hidden_layers', 1))
    #num_nodes = int(conf.get('num_nodes', 2))
    #activation = conf.get('activation', 'relu')

    #should be dependent on vars read from config
    #if activation=='relu':
     #   activation = nn.ReLU()
    #elif activation=='sigmoid':
    #    activation = nn.Sigmoid()
    arch_dict = json.loads(arch)
    nn_config_dict = json.loads(nn_config)
    model = Net(arch_dict, nn_config_dict)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate) #Adam optimizer
    model.train()    

    if device!='cpu':
        model = model.to(device)
        features_train = features_train.to(device)
        target_train = target_train.to(device)

    for epoch in range(numepochs): 
        features_shuffled = features_train
        target_shuffled = target_train

        out = model(features_shuffled) #predictions from model
        loss = criterion(out.squeeze(), target_shuffled.squeeze()) #loss between predictions and labels

        if epoch % 1000 == 0:
            print(f'epoch = {epoch} loss = {loss}')

        optimizer.zero_grad()
        loss.backward() #compute gradients
        optimizer.step() #update model

    out = model(features_shuffled) #predictions from model
    train_loss = criterion(out.squeeze(), target_shuffled.squeeze()) #loss between predictions and labels
    print(f'Train Loss : {train_loss}')




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

    #num_gpus = args.num_gpus
    #print(">>> num_gpus received by trial:")
    #print(num_gpus)

    #Using CPU for now
    device = torch.device("cpu")
    print(">>> Use CPU for Training <<<")

    #for not using relu
    activation = "relu"
    #train_model(learningrate, num_epochs, num_layers, num_nodes,activation)
    train_model(num_epochs, arch, nn_config)
	
	



