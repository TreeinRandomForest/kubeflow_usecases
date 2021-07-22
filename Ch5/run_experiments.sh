#!/bin/bash

for nproc in $(seq 1 1 12)
do
    echo $nproc
    python run.py --nnode 1 --nproc_per_node $nproc pytorch_dist_test3.py --local_world_size $nproc
done
