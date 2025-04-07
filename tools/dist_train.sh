#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}


gpu_id="0,1,2,3"
CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=24 \
python3 -m torch.distributed.launch \
--nproc_per_node=4 --master_port=12345 --node_rank=0 train.py \
configs/groupfree3d/dest3d_8x4_scannet-3d-18class-L6-O256.py \
--launcher pytorch