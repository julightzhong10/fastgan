save_path=/path/to/save/
data_path=/path/to/data/folder/
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0,1 python3.6 -u ./train.py \
    --random_seed 10 \
    --net sngan_catdog128 \
    --dim_z 128 \
    --img_c 3 \
    --fm_base_d 64 --fm_base_g 64 \
    --n_classes 143 \
    --num_bp 1 \
    --r_batch_size 64 --data_path ${data_path} \
    --g_batch_size 64 \
    --shuffle \
    --out_path ${save_path} \
    --bottom_width 4 \
    --img_width 128 \
    --itr_d 5 --lr_d 0.0002 --beta1_d 0.0 --beta2_d 0.9 \
    --itr_g 1 --lr_g 0.0002 --beta1_g 0.0 --beta2_g 0.9 \
    --anneal_lr linear --anneal_lr_p1 400000 \
    --n_gpu 2 --dali --n_workers 2 \
    --G_total_itrs 450001 --save_bias 2500 --start_itr 0 \
    --bn_g --sn_d \
    --b_metric hinge --c_metric hinge \
    --c_alph_f 1.0 --c_alph_g 1.0 \
    --pgd_type Linf --pgd_steps 0 --pgd_eps 0.0 --pgd_tau 0.0
