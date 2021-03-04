
#PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class Net(nn.Module):
    def __init__(self, N_hidden=16):
        super(Net, self).__init__()
        
        #self.N_input = df_train.shape[1]-1
        self.N_input = 29
        self.N_output = 1
        
        self.layer1 = nn.Linear(self.N_input, N_hidden)
        self.layer2 = nn.Linear(N_hidden, self.N_output)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.layer2(self.act(self.layer1(x)))
        
        return x

def main():
    net = Net(N_hidden=16)
    print(net)

if __name__=='__main__':
    main()
