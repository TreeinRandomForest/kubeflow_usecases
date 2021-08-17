import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import gym
import json
plt.ion()

class PolicyNet(nn.Module):
    def __init__(self, N_inputs, N_outputs, N_hidden_layers, N_hidden_nodes, activation, output_activation):
        super(PolicyNet, self).__init__()
        
        self.N_inputs = N_inputs
        self.N_outputs = N_outputs
        
        self.N_hidden_layers = N_hidden_layers
        self.N_hidden_nodes = N_hidden_nodes
        
        self.layer_list = nn.ModuleList([]) #use just as a python list
        for n in range(N_hidden_layers):
            if n==0:
                self.layer_list.append(nn.Linear(N_inputs, N_hidden_nodes))
            else:
                self.layer_list.append(nn.Linear(N_hidden_nodes, N_hidden_nodes))
        
        self.output_layer = nn.Linear(N_hidden_nodes, N_outputs)
        
        self.activation = activation
        self.output_activation = output_activation
        
    def forward(self, inp):
        out = inp
        for layer in self.layer_list:
            out = layer(out)
            out = self.activation(out)
            
        out = self.output_layer(out)
        if self.output_activation is not None:
            pred = self.output_activation(out)
        else:
            pred = out
        
        return pred

def plot_learning_curve(filename):
    d = json.load(open(filename, 'rb'))

    plt.figure()
    plt.plot(d.keys(), d.values(), label=filename)
    plt.xlabel('Time (s)')
    plt.ylabel('Average Reward in Current Batch')

    time_vals = np.array(list(d.keys()))
    ticks = np.arange(0, len(time_vals), 100)
    labels = [format(float(t), ".1f") for t in time_vals[ticks]]

    plt.xticks(ticks=ticks, labels=labels)

    plt.legend()
    
    return d

def animate(policy_filename):
    '''Can be combined with create trajectories above
    '''
    policy = torch.load(open(policy_filename, 'rb'))

    #env specific
    env = gym.make('CartPole-v1')
    action_space = np.arange(0, 2)

    state = env.reset()

    done = False
    while not done:
        env.render()

        action_probs = policy(torch.from_numpy(state).unsqueeze(0).float()).squeeze(0)

        action_selected_index = torch.multinomial(action_probs, 1)
        action_selected_prob = action_probs[action_selected_index]
        action_selected = action_space[action_selected_index]

        state, reward, done, info = env.step(action_selected.item())
