import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, rpc_async, rpc_sync, remote
import torch.multiprocessing as mp
import torch

import numpy as np
import time
import json
import os
import gym

#python run.py --nnode 1 --nproc_per_node 2 rpc_rl.py --local_world_size 2

'''
Architecture:
rank=0:
    Create Policy
        remote call: get action | state
        remote call: send reward (log prob local)
    Wrap in Coordinator:
        call: spin up K workers, init EpisodeRunner, run episode
        call: once episodes finish, run PG update
rank!=0:
    Create EpisodeRunner
        Init env + run episode + get actions through rpc calls + send rewards through rpc calls
'''

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

class Coordinator():
    def __init__(self, world_size, lr=1e-2, gamma=0.99):
        #init policy
        N_inputs = 4
        N_outputs = 2
        N_hidden_layers = 1
        N_hidden_nodes = 10
        activation = nn.ReLU()
        output_activation = nn.Softmax(dim=1)

        action_space = torch.arange(0, 2)

        self.world_size = world_size

        self.policy = PolicyNet(N_inputs, 
                                N_outputs, 
                                N_hidden_layers, 
                                N_hidden_nodes,
                                activation,
                                output_activation=output_activation)        
    

        #data structures to hold rewards and log probs
        #map worker id -> [] of log probs and rewards
        self.log_probs = {i:[] for i in range(1, world_size)}
        self.rewards = {i:[] for i in range(1, world_size)}

        #updating model
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

        #miscellaneous
        self.iter = 0
        self.log_interval = 10
        self.stats = {}

    def get_action(self, state, worker_id):
        #predict prob distribution on actions and sample
        action_probs = self.policy(torch.tensor(state).float().unsqueeze(0))[0]
        action_selected_index = torch.multinomial(action_probs, 1).item()
        
        #update log probs needed for policy gradient updates
        self.log_probs[worker_id].append(action_probs[[action_selected_index]].log())

        return action_selected_index

    def record_reward(self, reward, worker_id):
        self.rewards[worker_id].append(reward)

        #assert(len(self.rewards[worker_id]==self.log_probs[worker_id]))

    def update_model(self):
        #compute J = expected reward
        J = 0
        total_rewards_list = []
        for k in range(1, self.world_size):
            total_log_prob = torch.cat(self.log_probs[k]).sum()
            total_reward = np.sum(self.rewards[k]).sum()

            J += (total_log_prob)*(total_reward)

            total_rewards_list.append(total_reward)

        #compute rewards to go

        #baseline is just MC average of current episode


        #backprop and update policy
        self.optimizer.zero_grad()
        (-J).backward()
        self.optimizer.step()

        if self.iter % self.log_interval == 0:
            print(f"Average Reward: {np.mean(total_rewards_list)}")

        self.log_probs = {i:[] for i in range(1, world_size)}
        self.rewards = {i:[] for i in range(1, world_size)}
        self.stats[self.iter] = np.mean(total_rewards_list)
        self.iter += 1
        
    def run_training_loop(self, N_iter, coord_rref):
        self.iter = 0
        for i in range(N_iter):
            #create world_size-1 EpisodeRunner objects
            rref_list = [rpc.remote(f"rank{j}", EpisodeRunner, (j,)) for j in range(1, world_size)]
            
            #launch episodes
            fut_list = [r.rpc_async().run_episode(coord_rref) for r in rref_list]
            [fut.wait() for fut in fut_list]

            #update model
            self.update_model()

            #print stats

class EpisodeRunner:
    def __init__(self, rank):
        self.env = gym.make('CartPole-v1')
        self.action_space = np.arange(0, 2)
        self.rank = rank

    def run_episode(self, coord_rref):
        state = self.env.reset()

        done = False

        while not done:
            action_selected_index = coord_rref.rpc_sync().get_action(state, self.rank) #rpc call to receive action from worker = 0

            action = self.action_space[action_selected_index]

            state, reward, done, info = self.env.step(action)

            coord_rref.rpc_sync().record_reward(reward, self.rank) #rpc call to send reward to worker = 0


key_list = ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE']

def run(rank, world_size):
    print(f'rank = {rank} world_size = {world_size}')
    env_dict = {key: os.environ[key] for key in key_list}

    print(env_dict)

    if rank==0:
        rpc.init_rpc(f"rank{rank}", rank=rank, world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE)
    
        coordinator = Coordinator(world_size=world_size)
        coord_rref = RRef(coordinator)
        coordinator.run_training_loop(1000, coord_rref)

        torch.save(coordinator.policy, open(f'plots/policy_nworkers{world_size-1}.pt', 'wb'))
        json.dump(coordinator.stats, open(f'plots/stats_nworkers{world_size-1}.json', 'w'))

    else:
        rpc.init_rpc(f"rank{rank}", rank=rank, world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE)

    rpc.shutdown()

if __name__=='__main__':
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    run(rank, world_size)
        