import torch.nn as nn
import torch
import torch.optim as optim
import argparse
import json

import numpy as np

import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Net(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=1, n_hidden_nodes=10, n_hidden_layers=1, activation=nn.ReLU(), output_activation=None):
        super(Net, self).__init__()

        self.layer_list = nn.ModuleList()

        for i in range(n_hidden_layers):
            if i==0:
                self.layer_list.append(nn.Linear(n_inputs, n_hidden_nodes))
            else:
                self.layer_list.append(nn.Linear(n_hidden_nodes, n_hidden_nodes))
        
        self.output_layer = nn.Linear(n_hidden_nodes, n_outputs)

        self.activation = activation
        self.output_activation = output_activation

    def forward(self, x):
        out = x

        for layer in self.layer_list:
            out = self.activation(layer(out))

        out = self.output_layer(out)
        if self.output_activation is not None:
            out = self.output_activation(out)

        return out

def train_model(learningrate, numepochs, numhiddenlayers, numnodes,activation) -> int:
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    #N_epochs = int(conf.get('N_epochs', 1000))
    #num_hidden_layers = int(conf.get('num_hidden_layers', 1))
    #num_nodes = int(conf.get('num_nodes', 2))
    #activation = conf.get('activation', 'relu')

    #should be dependent on vars read from config
    if activation=='relu':
        activation = nn.ReLU()
    elif activation=='sigmoid':
        activation = nn.Sigmoid()

    model = Net(n_inputs=3, n_outputs=10, n_hidden_nodes=numnodes, n_hidden_layers=numhiddenlayers, activation=activation, output_activation=nn.Sigmoid())
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate) #Adam optimizer
    model.train()    

    if device!='cpu':
        model = model.to(device)
        features_train = features_train.to(device)
        target_train = target_train.to(device)

    for epoch in range(N_epochs): #N_epochs = number of iterations over the full dataset
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
	
	



