#!/bin/bash
data_path=/path/to/dataset/folder/
out_path=/path/to/output/FID.npz/file/like/cifar100_FID_stat.npz
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python3.6 -u ../cal_FID_stat.py \
    --data_path ${data_path} --out_path ${out_path} \
    --img_width 32 --ipt_dims 2048\
    --n_gpu 1 --dali --n_workers 1 --da \
    --batch_size 100 --n_samples 50000


