#!/bin/bash
load_path=/path/to/a/.pth/file/
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7 python3.6 -u ./eval_IS.py \
    --net sngan_cifar10 \
    --dim_z 128 \
    --img_c 3 \
    --fm_base 128 \
    --n_classes 10 \
    --bottom_width 4 \
    --img_width 32 \
    --n_gpu 1 --dali \
    --bn \
    --Z_dist normal --Z_p1 0 --Z_p2 1 \
    --load_path ${load_path} \
    --start 1 --end 400 --skip 1 \
    --batch_size 100 \
    --n_splits 10 \
    --n_samples 10000 --tf

