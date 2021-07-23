import argparse
import os

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

#to run DDP on one local node:

#python launch.py --nnode 1 --nproc_per_node cpu pytorch_dist_test2.py --local_world_size 12
#python run.py --nnode 1 --nproc_per_node cpu pytorch_dist_test2.py --local_world_size 12


key_list = ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE']

def example(rank, world_size):
    print(f'rank = {rank} world_size = {world_size}')
    env_dict = {key: os.environ[key] for key in key_list}

    print(env_dict)

    dist.init_process_group(backend='gloo')

    if rank == 0:
        data = torch.zeros(world_size)
        print(f'Data: ', data)

        for i in range(1, world_size):
            dist.recv(tensor=data[i], src=i)
            
        print(f'Data: ', data)

    else:
        dist.send(tensor=torch.tensor(float(world_size-rank)), dst=0)

        print(f'Rank {rank} sending {torch.tensor(rank)} to 0')

    dist.destroy_process_group()

#def main(world_size):
#    mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)

if __name__=='__main__':
    #not needed if using run.py which is recommended
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--local_rank", type=int)
    #parser.add_argument("--local_world_size", type=int)
    #args = parser.parse_args()

    #launch.py
    #example(args.local_rank, args.local_world_size)

    #List all envs
    #print( '\n'.join([f'{k}: {v}' for k, v in sorted(os.environ.items())]) )
    #run.py
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    example(rank, world_size)
