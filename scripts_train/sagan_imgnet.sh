save_path=/path/to/save/
data_path=/path/to/data/folder/
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0,1,2,3 python3.6 -u ./train.py \
    --random_seed 10 \
    --net sngan_imgnet128 \
    --dim_z 128 \
    --img_c 3 \
    --fm_base_d 64 --fm_base_g 64 \
    --n_classes 1000 \
    --num_bp 1 \
    --r_batch_size 256 --data_path ${data_path} \
    --g_batch_size 256 \
    --shuffle \
    --out_path ${save_path} \
    --bottom_width 4 \
    --img_width 128 \
    --itr_d 1 --lr_d 0.0004 --beta1_d 0.0 --beta2_d 0.9 \
    --itr_g 1 --lr_g 0.0001 --beta1_g 0.0 --beta2_g 0.9 \
    --anneal_lr nll \
    --n_gpu 4 --dali --n_workers 2 \
    --G_total_itrs 1000000 --save_bias 1000 --start_itr 0 \
    --bn_g --sa_d --sa_g --sn_d --sn_g --wide \
    --b_metric hinge --c_metric hinge \
    --c_alph_f 1.0 --c_alph_g 1.0 \
    --pgd_type Linf --pgd_steps 0 --pgd_eps 0.0 --pgd_tau 0.0
