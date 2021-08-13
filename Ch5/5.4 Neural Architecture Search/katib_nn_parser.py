import json
import torch
import torch.nn as nn
import torch.optim as optim

def create_net(arch, nn_config):
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

    layer_list = []

    for layer in arch:


def get_layer(type, params):
    if type=='convolution':
        
