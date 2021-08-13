import json
import torch
import torch.nn as nn
import torch.optim as optim

nn_config = {'input_sizes': 5, 'output_sizes': 7, 'embedding': {'1': {'opt_params': {'units': 10}}}}
arch = [[1],
        [1, 0],
        [1, 1, 1]
        ]

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
    in_val = input_size
    out_val = None

    skip_cons = {}

    for idx, l in enumerate(arch): #loop over each layer (after input)
        #lookup layer type
        l_lookup_id = str(l[0])
        emb = embedding[l_lookup_id]

        #keep track of incoming skip connections
        skip_cons[idx] = []

        out_val = emb['opt_params']['units']

        #update dense layer in size based on skip connections
        for s_id in range(1, len(l)):
            s_bool = l[s_id]

            if s_bool==1:
                in_val += unit_list[s_id-1][0]
                skip_cons[idx].append(s_id-1) #refers to input of unit_list[s_id-1]

        unit_list.append((in_val, out_val))

        in_val = out_val

    unit_list.append((in_val, output_size))
    skip_cons[len(unit_list)-1] = []

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
        buffer = [x]
        for idx, l in enumerate(self.layer_list):
            
            out = torch.cat([buffer[t] for t in self.skip_cons[idx]] + [out], dim=1)

            out = self.activation(l(out))
            
            buffer.append(out)

        return out