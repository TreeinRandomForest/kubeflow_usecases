import json
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

nn_config = {'input_sizes': 5, 'output_sizes': 7, 'embedding': {'1': {'opt_type': 'dense', 'opt_params': {'units': 10}}}}
arch = [[1],
        [1, 0],
        [1, 1, 1]
        ]

#arch = [[1], [1, 1], [1, 0, 0], [1, 1, 1, 1]]


def test(N_exp, max_length=3):
    nn_config = {'input_sizes': 5, 'output_sizes': 7, 'embedding': {'1': {'opt_type': 'dense', 'opt_params': {'units': 10}}}}
    arch = [[1],
            [1, 0],
            [1, 1, 1]
            ]

    for exp in range(N_exp):
        n_layers = np.random.randint(1, max_length)

        arch = []
        for i in range(n_layers):
            config = [1] + [np.random.randint(2) for k in range(i)] 
            arch.append(config)

        print("-------------")
        [print(config) for config in arch]
        net = Net(arch, nn_config)
        x = torch.rand(3, 5)
        try:
            y = net(x)
        except:
            import ipdb
            ipdb.set_trace()

        print(y.shape)

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
    in_val = input_size
    out_val = None
    carry = 0
    skip_cons = {0: []}

    for idx, l in enumerate(arch):
        layer_type = embedding[str(l[0])]
        if layer_type['opt_type']!='dense': raise ValueError("found non-dense layer")

        out_val = layer_type['opt_params']['units']
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
        self.output_activation = nn.Softmax(dim=1)

    def forward(self, x):
        out = x
        self.buffer = [x]
        for idx, l in enumerate(self.layer_list):
            
            out = torch.cat([self.buffer[t] for t in self.skip_cons[idx]] + [out], dim=1)

            out = self.activation(l(out))
            
            self.buffer.append(out)

        return out


