import torch
from tools.loading import load_models_gen,load_datasets, load_inception_FID
import argparse
import random
import os
import numpy as np
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--out_path', type=str, required=True)
parser.add_argument('--img_width', type=int, required=True)
parser.add_argument('--ipt_dims', type=int, required=True)
parser.add_argument('--n_gpu', type=int, required=True)
parser.add_argument('--dali', action='store_true', default=False)
parser.add_argument('--da', action='store_true', default=False, help = 'data augmentation?')
parser.add_argument('--tf', action='store_true', default=False)
parser.add_argument('--n_workers', type=int, required=True)
parser.add_argument('--n_samples', type=int, required=50000)
parser.add_argument('--batch_size', type=int, required=50000)
opt = parser.parse_args()
print(opt)

def eval():
    # models
    datasets, class_to_idx = load_datasets(data_path=opt.data_path,img_width=opt.img_width,\
                        batch_size=opt.batch_size,n_workers=opt.n_workers,n_gpu=opt.n_gpu,shuffle=True,dali=opt.dali,seed=random.randint(0,9999),da=opt.da)

    if opt.tf:
        import tools.networks.inception_eval.inception_tf as inception_tf
        from tools.evalution import sample_imgs, generate_imgs
        total_iters = math.ceil(opt.n_samples/opt.batch_size)
        img_list = sample_imgs(train_loader=datasets,total=opt.n_samples,total_iters=total_iters,img_width=opt.img_width,dali=opt.dali)
        mu,sig = inception_tf.get_mean_and_cov(img_list)
    else:
        from tools.evalution import get_activations_stat_gen,get_activations_stat_orig, calculate_frechet_distance
        inception_model = load_inception_FID(opt.n_gpu,opt.ipt_dims)
        mu,sig = get_activations_stat_orig(datasets=datasets,n_samples=opt.n_samples,ipt_net=inception_model,ipt_dims=opt.ipt_dims,n_gpu=opt.n_gpu,dali=opt.dali)
    
    np.savez(opt.out_path,**{'mu':mu,'sig':sig})
if __name__ == "__main__":
    eval()
