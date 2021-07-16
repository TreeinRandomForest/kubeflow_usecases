import argparse
import os

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

#to run DDP on one local node:
#python launch.py --nnode 1 --node_rank=0 --nproc_per_node=3 pytorch_dist_test.py --local_world_size=3
#where launch.py is a link to `python -c "from os import path; import torch; print(path.join(path.dirname(torch.__file__), 'distributed', 'launch.py'))"`



key_list = ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE']

def example(rank, world_size):
    print(f'rank = {rank} world_size = {world_size}')
    env_dict = {key: os.environ[key] for key in key_list}

    print(env_dict)

    dist.init_process_group(backend='nccl')

    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        f"world_size = {dist.get_world_size()}"
    )

    dist.destroy_process_group()

#def main(world_size):
#    mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--local_world_size", type=int)
    args = parser.parse_args()

    example(args.local_rank, args.local_world_size)