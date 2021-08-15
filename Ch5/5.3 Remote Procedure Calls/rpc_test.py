import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, rpc_async, rpc_sync, remote
import torch.multiprocessing as mp
import torch

import os

#python run.py --nnode 1 --nproc_per_node 2 rpc_rl.py --local_world_size 2

key_list = ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE']

def run(rank, world_size):
    print(f'rank = {rank} world_size = {world_size}')
    env_dict = {key: os.environ[key] for key in key_list}

    print(env_dict)

    if rank==0:
        rpc.init_rpc(f"rank{rank}", rank=rank, world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE)
        
        #rref = rpc.rpc_async('rank1', torch.sum, args=(torch.ones(2),))
        rref = rpc.remote('rank1', torch.sum, args=(torch.ones(2),))
        print(rref)
        import ipdb
        ipdb.set_trace()
        

    elif rank==1:
        rpc.init_rpc(f"rank{rank}", rank=rank, world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE)
    else:
        rpc.init_rpc(f"rank{rank}", rank=rank, world_size=world_size, backend=rpc.BackendType.PROCESS_GROUP)


    rpc.shutdown()

if __name__=='__main__':
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    run(rank, world_size)