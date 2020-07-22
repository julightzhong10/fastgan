#!/bin/bash
load_path=/path/to/a/.pth/file/
stat_path=/path/to/a/pre-calculated/FID/.npz/file
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python3.6 -u ./eval_FID.py \
    --load_path ${load_path} --stat_path ${stat_path} \
    --net sngan_cifar10 \
    --dim_z 128 --img_c 3 --fm_base 128 --n_classes 10 \
    --bottom_width 4 --img_width 32 \
    --ipt_dims 2048\
    --n_gpu set_no._gpus --dali --n_workers 1 --da \
    --bn \
    --Z_dist normal --Z_p1 0 --Z_p2 1 \
    --start 1 --end 400 --skip 1 \
    --batch_size 100 --n_samples 5000



