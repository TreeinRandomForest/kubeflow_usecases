
import os, boto3, time, operator, requests
import numpy as np
import pandas as pd
import argparse

#Scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import precision_recall_curve,\
                            average_precision_score,\
                            roc_auc_score, roc_curve,\
                            confusion_matrix, classification_report
from sklearn.externals import joblib

#Plotting
import matplotlib.pylab as plt
import matplotlib.colors

#PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_architecture(df_train, N_hidden=128):
    '''Wrapper around the neural net architecture
    '''
    N_output = 1

    net = nn.Sequential(nn.Linear(df_train.shape[1]-1, N_hidden),
                        nn.ReLU(),
                        nn.Linear(N_hidden, N_output),
                        #nn.Sigmoid()
                       )
    
    return net


def get_criterion(weighted=False, pos_weight=torch.tensor([1,1])):
    '''Wrapper around criterion
    '''
    if not weighted:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) #NEEDS DEBUGGING
        
    return criterion
        
class CCDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.N_cols = df.shape[1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        x = np.array(self.df.iloc[ix])
        features = x[:(self.N_cols-1)] #exclude time, Class
        label = x[[-1]]

        #return {'features': torch.from_numpy(features), 'label': torch.from_numpy(label)}
        return (torch.from_numpy(features).float(), torch.from_numpy(label))
    

def train_model(train_dl, test_dl, model, criterion, N_epochs, print_freq, lr=1e-3, optimizer='adam'):
    '''Loop over dataset in batches, compute loss, backprop and update weights
    '''
    
    model.train() #switch to train model (for dropout, batch normalization etc.)
    
    model = model.to(device)
    if optimizer=='adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
        print("Using adam")
    elif optimizer=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
        print("Using sgd")
    else:
        raise ValueError("Please use either adam or sgd")
    
    avg_precision_dict, loss_dict = {}, {}
    for epoch in range(N_epochs): #loop over epochs i.e. sweeps over full data
        curr_loss = 0
        N = 0
        
        for idx, (features, labels) in enumerate(train_dl): #loop over batches
            features = features.to(device)
            labels = labels.to(device)
            
            preds = model(features)
            loss = criterion(preds.squeeze(), labels.squeeze().float())
            
            curr_loss += loss.item() #accumulate loss
            N += len(labels) #accumulate number of data points seen in this epoch
                
            #backprop and updates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch % print_freq == 0 or epoch==N_epochs-1:
            val_loss, val_avg_precision = validate(test_dl, model, criterion) #get model perf metrics from test set
            
            avg_precision_dict[epoch] = val_avg_precision
            loss_dict[epoch] = val_loss
            
            print(f'Iter = {epoch} Train Loss = {curr_loss / N} val_loss = {val_loss} val_avg_precision = {val_avg_precision}')
            
    return model, avg_precision_dict, loss_dict

def validate(test_dl, model, criterion):
    '''Loop over test dataset and compute loss and accuracy
    '''
    model.eval() #switch to eval model
    
    loss = 0
    N = 0

    preds_all, labels_all = torch.tensor([]), torch.tensor([])
    
    with torch.no_grad(): #no need to keep variables for backprop computations
        for idx, (features, labels) in enumerate(test_dl):
            features = features.to(device)
            labels = labels.to(device).float()
            
            preds = model(features)
            
            preds_all = torch.cat((preds_all, preds.to('cpu')), 0)
            labels_all = torch.cat((labels_all, labels.to('cpu')), 0)
            
            loss += criterion(preds.squeeze(), labels.squeeze()) #cumulative loss
            N += len(labels)
    
    avg_precision = average_precision_score(labels_all.squeeze().numpy(), preds_all.squeeze().numpy())
    
    return loss / N, avg_precision

def main():
    parser = argparse.ArgumentParser(description='Credit Card Fraud Detection')
    
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='batch size for training (default = 64)')

    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--n-hidden', type=int, default=16, metavar='N',
                        help='number of hidden layers')

    parser.add_argument('--optimizer', type=str, default='adam', metavar='N',
                        help='optimizer to use: "adam" or "sgd"')
    
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    
    parser.add_argument('--dir', default='logs', metavar='L',
                        help='directory where summary logs are stored')
  

    
    args = parser.parse_args()
    
    batch_size = args.batch_size
    N_epochs = args.epochs
    lr = args.lr
    N_print = args.log_interval
    N_hidden = args.n_hidden
    optimizer = args.optimizer
    
    #Read data and preprocessing
    df = pd.read_csv('creditcard.csv')
    df.drop('Time', inplace=True, axis=1)

    # Train-test split
    df_train, df_test = train_test_split(df, train_size=0.8)

    ds_torch_train = CCDataset(df_train)
    ds_torch_test = CCDataset(df_test)

    dl_torch_train = DataLoader(ds_torch_train, batch_size=batch_size, num_workers=0)
    dl_torch_test = DataLoader(ds_torch_test, batch_size=batch_size, num_workers=0)

    #Network architecture and criterion
    net = get_architecture(df_train, N_hidden=N_hidden)
    criterion = get_criterion()
    print(net)
    print(f"Learning rate: {lr}")
    
    net, avg_precision_dict, loss_dict = train_model(dl_torch_train, dl_torch_test, net, criterion, N_epochs, N_print, lr=lr, optimizer=optimizer)

if __name__=='__main__':
    main()
