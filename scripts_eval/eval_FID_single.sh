#!/bin/bash
load_path=/path/to/a/.pth/file/
stat_path=/path/to/a/pre-calculated/FID/.npz/file
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python3.6 -u ./eval_FID_single.py \
    --load_path ${load_path} --stat_path ${stat_path} \
    --net set_your_network \
    --dim_z set_your_latent_code_dim --img_c 3 --fm_base set_fm_base --n_classes set_classes_no. \
    --bottom_width 4 --img_width set_img_width \
    --ipt_dims 2048\
    --n_gpu set_no._gpus --dali --n_workers 1 --da \
    --bn \
    --Z_dist normal --Z_p1 0 --Z_p2 1 \
    --batch_size set_batch_size --n_samples set_samples_no._for_FID

