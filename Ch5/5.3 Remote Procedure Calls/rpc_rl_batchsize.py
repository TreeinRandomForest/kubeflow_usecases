import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, rpc_async, rpc_sync, remote
import torch.multiprocessing as mp
import torch

import numpy as np
import itertools
import argparse
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
    def __init__(self, world_size, batch_size_multiple, lr=1e-2, gamma=0.99):
        #init policy
        #env_name = 'CartPole-v1'
        #N_inputs = 4
        #N_outputs = 2

        self.env_name = 'LunarLander-v2'
        N_inputs = 8
        N_outputs = 4

        N_hidden_layers = 1
        N_hidden_nodes = 10
        activation = nn.ReLU()
        output_activation = nn.Softmax(dim=1)

        action_space = torch.arange(0, 2)

        self.world_size = world_size
        self.batch_size_multiple = batch_size_multiple

        self.policy = PolicyNet(N_inputs, 
                                N_outputs, 
                                N_hidden_layers, 
                                N_hidden_nodes,
                                activation,
                                output_activation=output_activation)        
    

        #data structures to hold rewards and log probs
        #map worker id -> [] of log probs and rewards
        self.log_probs = {(i,j):[] for i,j in itertools.product(range(1, self.world_size), range(self.batch_size_multiple))}
        self.rewards = {(i,j):[] for i,j in itertools.product(range(1, self.world_size), range(self.batch_size_multiple))}

        #updating model
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

        #miscellaneous
        self.rref_list = []
        self.iter = 0
        self.log_interval = 10
        self.stats = {}
        self.start_time = time.time()
        self.exp_reward_factor = 0.90
        self.exp_reward_avg = 0

    def get_action(self, state, worker_id, batch_id):
        #predict prob distribution on actions and sample
        action_probs = self.policy(torch.tensor(state).float().unsqueeze(0))[0]
        action_selected_index = torch.multinomial(action_probs, 1).item()
        
        #update log probs needed for policy gradient updates
        self.log_probs[(worker_id, batch_id)].append(action_probs[[action_selected_index]].log())

        return action_selected_index

    def record_reward(self, reward, worker_id, batch_id):
        self.rewards[(worker_id, batch_id)].append(reward)

        #assert(len(self.rewards[(worker_id, batch_id)]==self.log_probs[(worker_id, batch_id)]))

    def update_model(self):
        #compute J = expected reward
        J = 0
        total_rewards_list = []
        '''
        for k in range(1, self.world_size):
            total_log_prob = torch.cat(self.log_probs[k]).sum()
            total_reward = np.sum(self.rewards[k])

            J += (total_log_prob)*(total_reward)

            total_rewards_list.append(total_reward)
        '''

        #compute rewards to go
        #import ipdb
        #ipdb.set_trace()

        for k in self.rewards.keys():
            r = np.array(self.rewards[k])
            episode_length = len(r)

            r_to_go = torch.tensor([np.sum(r[t:] * self.gamma**(np.arange(episode_length-t))) for t in range(episode_length)])

            J += (torch.cat(self.log_probs[k]) * r_to_go).sum()

            total_reward = np.sum(self.rewards[k])
            total_rewards_list.append(total_reward)

        #baseline is just MC average of current episode
        self.current_reward_avg = np.mean(total_rewards_list)
        self.exp_reward_avg = self.exp_reward_factor*self.exp_reward_avg + (1-self.exp_reward_factor)*self.current_reward_avg

        #backprop and update policy
        J /= len(self.rewards)
        self.optimizer.zero_grad()
        (-J).backward()
        self.optimizer.step()

        if self.iter % self.log_interval == 0:
            print(f"Current Average Reward: {self.current_reward_avg} Exponentially Average Reward: {self.exp_reward_avg}")

        self.log_probs = {(i,j):[] for i,j in itertools.product(range(1, self.world_size), range(self.batch_size_multiple))}
        self.rewards = {(i,j):[] for i,j in itertools.product(range(1, self.world_size), range(self.batch_size_multiple))}
        
        self.stats[time.time()-self.start_time] = (self.current_reward_avg, self.exp_reward_avg)
        self.iter += 1
        
    def run_training_loop(self, N_iter, coord_rref):
        self.iter = 0

        if len(self.rref_list)==0:
            self.rref_list = [rpc.remote(f"rank{j}", EpisodeRunner, (self.env_name, j,)) for j in range(1, world_size)]

        for i in range(N_iter):           
            #launch episodes
            for batch_id in range(self.batch_size_multiple):
                fut_list = [r.rpc_async().run_episode(coord_rref, batch_id) for r in self.rref_list]
                [fut.wait() for fut in fut_list]

            #update model
            self.update_model()

            #print stats

class EpisodeRunner:
    def __init__(self, env_name, rank):
        self.env = gym.make(env_name)
        self.action_space = np.arange(0, self.env.action_space.n)
        self.rank = rank

    def run_episode(self, coord_rref, batch_id):
        state = self.env.reset()

        done = False

        while not done:
            action_selected_index = coord_rref.rpc_sync().get_action(state, self.rank, batch_id) #rpc call to receive action from worker = 0

            action = self.action_space[action_selected_index]

            state, reward, done, info = self.env.step(action)

            coord_rref.rpc_sync().record_reward(reward, self.rank, batch_id) #rpc call to send reward to worker = 0


key_list = ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE']

def run(rank, world_size):
    print(f'rank = {rank} world_size = {world_size}')
    env_dict = {key: os.environ[key] for key in key_list}

    print(env_dict)

    if rank==0:
        rpc.init_rpc(f"rank{rank}", rank=rank, world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE)
        
        batch_size_multiple = 1
        n_iter = 10000 #number of updates

        coordinator = Coordinator(world_size, batch_size_multiple, lr=1e-3)
        coord_rref = RRef(coordinator)
        coordinator.run_training_loop(n_iter, coord_rref)

        torch.save(coordinator.policy, open(f'plots/{coordinator.env_name}_policy_nworkers{world_size-1}_batchsizemultiple{batch_size_multiple}.pt', 'wb'))
        json.dump(coordinator.stats, open(f'plots/{coordinator.env_name}_stats_nworkers{world_size-1}_batchsizemultiple{batch_size_multiple}.json', 'w'))

    else:
        rpc.init_rpc(f"rank{rank}", rank=rank, world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE)

    rpc.shutdown()

if __name__=='__main__':
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    run(rank, world_size)
        